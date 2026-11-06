"""CPU research prototype for aligning FM and refinement; not a Qwen backend.

Exact small AR laws check the objectives and candidate proposals. Optional
reports train a tiny drafter, including refinement on a fixed empirical bank.
"""

import itertools
import json
import sys
import unittest

import torch
import torch.nn.functional as F


DTYPE = torch.float64


def blocks(length):
    return torch.tensor(list(itertools.product(range(2), repeat=length)))


def markov_log_prob(ids, stay=0.95):
    same = ids[..., 1:] == ids[..., :-1]
    transitions = torch.where(same, torch.tensor(stay, dtype=DTYPE),
                              torch.tensor(1 - stay, dtype=DTYPE))
    return transitions.log().sum(-1) - torch.log(torch.tensor(2.0, dtype=DTYPE))


@torch.no_grad()
def posterior_from_candidates(x, t, ids, log_mass, noise_std=1.0):
    """log_mass is log P for enumeration, or log P - log proposal for particles.

    All endpoints are one-hot with the same length, so the endpoint-dependent
    Gaussian log likelihood is t*sum_j x[j, ids[j]] / (sigma^2*(1-t)^2).
    The omitted Gaussian terms are common to every candidate at this state.
    """
    if not 0 <= t < 1:
        raise ValueError("posterior requires 0 <= t < 1")
    selected = x[torch.arange(x.size(0))[None], ids].sum(-1)
    log_weights = log_mass + t * selected / (noise_std**2 * (1 - t)**2)
    weights = log_weights.softmax(0)
    endpoint = F.one_hot(ids, x.size(-1)).to(DTYPE)
    return (weights[:, None, None] * endpoint).sum(0), weights


def markov_posterior(x, t, stay=0.95):
    """Exact full-block posterior using forward/backward for this toy teacher."""
    emission = t * x / (1 - t)**2
    transition = torch.tensor([[stay, 1 - stay], [1 - stay, stay]], dtype=DTYPE).log()
    alpha = [emission[0] - torch.log(torch.tensor(2.0, dtype=DTYPE))]
    for j in range(1, x.size(0)):
        alpha.append(emission[j] + torch.logsumexp(alpha[-1][:, None] + transition, 0))
    beta = [torch.zeros(2, dtype=DTYPE)]
    for j in range(x.size(0) - 2, -1, -1):
        beta.append(torch.logsumexp(transition + emission[j + 1][None] + beta[-1][None], 1))
    return (torch.stack(alpha) + torch.stack(list(reversed(beta)))).softmax(-1)


def conditional_velocity_mse(logits, x, t, ids, posterior_weights):
    """Exact expectation over endpoints conditional on the SAME current x."""
    q = logits.softmax(-1)
    velocity = (q - x) / (1 - t)
    endpoints = F.one_hot(ids, x.size(-1)).to(DTYPE)
    conditional_velocity = (endpoints - x[None]) / (1 - t)
    errors = ((1 - t) * (velocity[None] - conditional_velocity)).square().sum(-1)
    return (posterior_weights[:, None] * errors).sum(0).mean()


def weighted_kl(target, logits, position_weights):
    errors = (target * (target.clamp_min(1e-300).log() - logits.log_softmax(-1))).sum(-1)
    return (errors * position_weights).sum() / position_weights.sum()


def sample_markov(count, length, rng, stay=0.95):
    first = torch.randint(2, (count, 1), generator=rng)
    flips = (torch.rand(count, length - 1, generator=rng) >= stay).long()
    return torch.cat([first, (first + flips.cumsum(-1)) % 2], -1)


def bank_component_weights(bank, x, t, floor=0.1):
    selected = x[torch.arange(x.size(0))[None], bank].sum(-1)
    evidence = (t * selected / (1 - t)**2).softmax(0)
    return (1 - floor) * evidence + floor / bank.size(0)


def bank_proposal_log_prob(ids, bank, component_weights, mutation_probs, mutation=0.15):
    """Known density of a mixture over whole-block parents, with local mutations.

    Duplicate parents are retained as mixture components. Marginal probabilities
    of an ODE sample are NOT substituted for this explicitly defined density.
    """
    chosen = mutation_probs[torch.arange(ids.size(1))[None], ids]
    per_token = (mutation * chosen[:, None]
                 + (1 - mutation) * (ids[:, None] == bank[None]).to(DTYPE))
    return torch.logsumexp(component_weights.log()[None] + per_token.log().sum(-1), 1)


def sample_bank_proposal(bank, component_weights, mutation_probs, count, rng, mutation=0.15):
    indices = torch.multinomial(component_weights, count, replacement=True, generator=rng)
    mutations = torch.multinomial(mutation_probs, count, replacement=True, generator=rng).T
    replace = torch.rand(count, bank.size(1), generator=rng) < mutation
    ids = torch.where(replace, mutations, bank[indices])
    return ids, bank_proposal_log_prob(ids, bank, component_weights, mutation_probs, mutation)


class ToyDrafter(torch.nn.Module):
    """Small CPU-only FM drafter, with no access to the exact posterior."""

    def __init__(self, length):
        super().__init__()
        self.length = length
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2 * length + 1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 2 * length),
        )

    def forward(self, x, t):
        times = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1).expand(x.size(0), 1)
        logits = self.network(torch.cat([x.flatten(1), times], -1)).view(-1, self.length, 2)
        return logits.softmax(-1)


@torch.no_grad()
def empirical_bank_posterior(x, t, bank):
    """Exact posterior of the fixed empirical endpoint law, not of the full AR."""
    times = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1)
    selected = x.gather(2, bank.T[None].expand(x.size(0), -1, -1)).sum(1)
    weights = (times * selected / (1 - times)**2).softmax(-1)
    return torch.einsum("bs,slv->blv", weights, F.one_hot(bank, x.size(-1)).to(x.dtype))


def batched_refinement_weights(draft, acceptance=0.8):
    length = draft.size(1)
    j = torch.arange(length, dtype=torch.float32)
    u = acceptance**j * (1 - acceptance**(length - j)) / (1 - acceptance)
    expected = torch.cat([torch.zeros(draft.size(0), 1, dtype=torch.long), draft[:, :-1]], -1)
    matched = draft == expected
    prefix = torch.cat([torch.ones(draft.size(0), 1, dtype=torch.bool),
                        matched[:, :-1].cumprod(-1).bool()], -1)
    coefficient = torch.where(prefix & ~matched, 1.0, 0.5)
    weights = u[None] * coefficient
    return weights / weights.sum()


def train_toy(length=31, steps=2000, seed=912, bank=None, ms_weight=0.0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    refinement_rng = torch.Generator().manual_seed(seed + 2)
    model = ToyDrafter(length)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    losses, fm_losses, ms_losses = [], [], []
    for _ in range(steps):
        ids = (sample_markov(64, length, rng) if bank is None
               else bank[torch.randint(bank.size(0), (64,), generator=rng)])
        endpoints = F.one_hot(ids, 2).float()
        noise = torch.randn(64, length, 2, generator=rng)
        t = torch.rand(64, 1, 1, generator=rng)
        x = (1 - t) * noise + t * endpoints
        q = model(x, t.flatten())
        fm = (q - endpoints).square().sum(-1).mean()
        ms = fm * 0.0
        if ms_weight:
            if bank is None:
                raise ValueError("aligned toy refinement requires the same fixed bank as FM")
            count = int(torch.randint(1, 5, (), generator=refinement_rng))
            state = torch.randn(64, length, 2, generator=refinement_rng)
            previous = None
            terms = []
            for i in range(count):
                time = i / count
                prediction = model(state, time)
                if i:
                    target = empirical_bank_posterior(state, time, bank)
                    weights = batched_refinement_weights(previous.argmax(-1))
                    errors = (target * (target.clamp_min(1e-30).log()
                                       - prediction.clamp_min(1e-30).log())).sum(-1)
                    terms.append((errors * weights).sum())
                state = (state + (prediction.detach() - state) / (count - i)).detach()
                previous = prediction.detach()
            if terms:
                ms = torch.stack(terms).mean()
        loss = fm + ms_weight * ms
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        fm_losses.append(float(fm.detach()))
        ms_losses.append(float(ms.detach()))
    return model.eval(), {"steps": steps, "seed": seed, "batch_size": 64,
                          "bank_size": None if bank is None else bank.size(0), "ms_weight": ms_weight,
                          "first_100_loss": sum(losses[:100]) / 100,
                          "last_100_loss": sum(losses[-100:]) / 100,
                          "last_100_fm": sum(fm_losses[-100:]) / 100,
                          "last_100_ms": sum(ms_losses[-100:]) / 100}


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_posterior_matches_full_gaussian_density_and_uses_future_evidence(self):
        ids = blocks(2)
        log_p = markov_log_prob(ids, 0.98)
        t = 0.6
        x = torch.tensor([[0.2, 0.2], [0.7, -0.4]], dtype=DTYPE)
        target, weights = posterior_from_candidates(x, t, ids, log_p)
        endpoints = F.one_hot(ids, 2).to(DTYPE)
        reference = (log_p - (x[None] - t * endpoints).square().sum((1, 2))
                     / (2 * (1 - t)**2)).softmax(0)
        torch.testing.assert_close(weights, reference)
        torch.testing.assert_close(target, markov_posterior(x, t, 0.98))
        # Local noise at position 0 is uninformative; position 1 changes its posterior.
        self.assertGreater(float(target[0, 0]), 0.95)
        torch.testing.assert_close(x[0].softmax(-1), torch.tensor([0.5, 0.5], dtype=DTYPE))

    def test_shared_optimum_for_fm_and_weighted_refinement(self):
        ids = blocks(3)
        x = torch.tensor([[0.7, -0.1], [0.3, 0.5], [-0.2, 0.6]], dtype=DTYPE)
        u_times_break = torch.tensor([1.22, 1.44, 0.32], dtype=DTYPE)
        for t in (0.0, 0.25, 0.5):
            with self.subTest(t=t):
                target, weights = posterior_from_candidates(x, t, ids, markov_log_prob(ids))
                logits = target.log().requires_grad_()
                fm = conditional_velocity_mse(logits, x, t, ids, weights)
                ms = weighted_kl(target, logits, u_times_break)
                g_fm, = torch.autograd.grad(fm, logits, retain_graph=True)
                g_ms, = torch.autograd.grad(ms, logits)
                self.assertLess(float(g_fm.abs().max()), 1e-12)
                self.assertLess(float(g_ms.abs().max()), 1e-12)

    def test_original_ar_target_conflicts_but_aligned_output_gradients_do_not(self):
        ids = blocks(2)
        x = torch.tensor([[0.7, -0.1], [0.3, 0.5]], dtype=DTYPE)
        target, weights = posterior_from_candidates(x, 0.5, ids, markov_log_prob(ids))
        position_weights = torch.tensor([1.22, 1.44], dtype=DTYPE)
        logits = target.log().requires_grad_()
        # AR response to the previous draft [0, 0], without Gaussian evidence.
        raw_ar = torch.tensor([[0.5, 0.5], [0.95, 0.05]], dtype=DTYPE)
        old_gradient, = torch.autograd.grad(weighted_kl(raw_ar, logits, position_weights), logits)
        self.assertGreater(float(old_gradient.norm()), 0.05)
        generator = torch.Generator().manual_seed(2301)
        for _ in range(30):
            logits = torch.randn(2, 2, generator=generator, dtype=DTYPE, requires_grad=True)
            fm = conditional_velocity_mse(logits, x, 0.5, ids, weights)
            ms = weighted_kl(target, logits, position_weights)
            g_fm, = torch.autograd.grad(fm, logits, retain_graph=True)
            g_ms, = torch.autograd.grad(ms, logits)
            self.assertGreaterEqual(float((g_fm * g_ms).sum()), -1e-12)

    def test_importance_posterior_requires_proposal_correction(self):
        ids = blocks(2)
        proposal = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DTYPE)
        # A finite stratified collection whose frequencies exactly equal proposal.
        indices = torch.arange(4).repeat_interleave(torch.tensor([1, 2, 3, 4]))
        x = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=DTYPE)
        log_p = markov_log_prob(ids)
        exact, _ = posterior_from_candidates(x, 0.4, ids, log_p)
        importance, _ = posterior_from_candidates(
            x, 0.4, ids[indices], log_p[indices] - proposal[indices].log()
        )
        biased, _ = posterior_from_candidates(x, 0.4, ids[indices], log_p[indices])
        torch.testing.assert_close(importance, exact)
        self.assertGreater(float((biased - exact).abs().max()), 0.05)

    def test_unnormalized_importance_fm_has_exact_ar_gradient_in_expectation(self):
        ids = blocks(3)
        probability = markov_log_prob(ids).exp()
        proposal = torch.arange(1, 9, dtype=DTYPE)
        proposal /= proposal.sum()
        endpoints = F.one_hot(ids, 2).to(DTYPE)
        noise = torch.tensor([[0.3, -0.2], [0.2, 0.7], [-0.4, 0.1]], dtype=DTYPE)
        t = 0.4
        states = (1 - t) * noise[None] + t * endpoints
        coefficient = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
        q = (coefficient * states).softmax(-1)
        velocity = (q - states) / (1 - t)
        error = ((1 - t) * (velocity - (endpoints - noise[None]))).square().sum(-1).mean(-1)
        exact = (probability * error).sum()
        importance_expectation = (proposal * (probability / proposal).detach() * error).sum()
        torch.testing.assert_close(exact, importance_expectation)
        g_exact, = torch.autograd.grad(exact, coefficient, retain_graph=True)
        g_importance, = torch.autograd.grad(importance_expectation, coefficient)
        torch.testing.assert_close(g_exact, g_importance)

    def test_finite_particle_normalization_can_be_biased_despite_high_ess(self):
        ids = blocks(2)
        x = torch.zeros(2, 2, dtype=DTYPE)
        log_p = markov_log_prob(ids)
        exact, _ = posterior_from_candidates(x, 0.0, ids, log_p)
        # Every sampled particle misses the other mode; ESS still equals M.
        missing_mode = torch.zeros(16, 2, dtype=torch.long)
        estimated, weights = posterior_from_candidates(
            x, 0.0, missing_mode, markov_log_prob(missing_mode)
        )
        self.assertAlmostEqual(float(1 / weights.square().sum()), 16.0)
        self.assertAlmostEqual(float((estimated - exact).abs().max()), 0.5)

    def test_structured_proposal_is_normalized_and_importance_recovers_posterior(self):
        ids = blocks(3)
        # A duplicate parent tests mixture mass accounting.
        bank = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
        components = torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE)
        mutation_probs = torch.tensor([[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]], dtype=DTYPE)
        log_r = bank_proposal_log_prob(ids, bank, components, mutation_probs)
        torch.testing.assert_close(log_r.exp().sum(), torch.tensor(1.0, dtype=DTYPE))
        self.assertTrue(bool(torch.isfinite(log_r).all()))
        x = torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.5, 0.6]], dtype=DTYPE)
        target, _ = posterior_from_candidates(x, 0.4, ids, markov_log_prob(ids))
        # Exact proposal expectation of the unnormalised importance measure.
        evidence = 0.4 * x[torch.arange(3)[None], ids].sum(-1) / 0.6**2
        mass = log_r.exp() * (markov_log_prob(ids) + evidence - log_r).exp()
        reconstructed = (mass[:, None, None] * F.one_hot(ids, 2)).sum(0) / mass.sum()
        torch.testing.assert_close(target, reconstructed)

    def test_structured_sampler_matches_its_declared_density(self):
        bank = torch.tensor([[0, 0], [1, 1]])
        components = torch.tensor([0.4, 0.6], dtype=DTYPE)
        mutation_probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=DTYPE)
        rng = torch.Generator().manual_seed(954)
        samples, _ = sample_bank_proposal(bank, components, mutation_probs, 50000, rng)
        index = 2 * samples[:, 0] + samples[:, 1]
        actual = torch.bincount(index, minlength=4).to(DTYPE) / samples.size(0)
        expected = bank_proposal_log_prob(blocks(2), bank, components, mutation_probs).exp()
        torch.testing.assert_close(actual, expected, atol=0.006, rtol=0)

    def test_empirical_bank_posterior_counts_duplicates_and_has_shared_optimum(self):
        bank = torch.tensor([[0, 0], [0, 0], [1, 1]])
        x = torch.tensor([[[0.3, 0.2], [0.1, 0.4]]], dtype=DTYPE)
        t = 0.4
        actual = empirical_bank_posterior(x, t, bank)[0]
        expected, weights = posterior_from_candidates(x[0], t, bank, torch.zeros(3, dtype=DTYPE))
        torch.testing.assert_close(actual, expected)
        logits = actual.log().requires_grad_()
        fm = conditional_velocity_mse(logits, x[0], t, bank, weights)
        ms = weighted_kl(actual, logits, torch.tensor([1.22, 1.44], dtype=DTYPE))
        g_fm, = torch.autograd.grad(fm, logits, retain_graph=True)
        g_ms, = torch.autograd.grad(ms, logits)
        self.assertLess(float(g_fm.abs().max()), 1e-12)
        self.assertLess(float(g_ms.abs().max()), 1e-12)
        self.assertFalse(actual.requires_grad)

    def test_exact_one_step_euler_returns_mean_not_joint_samples(self):
        bank = torch.tensor([[0, 0], [0, 0], [1, 1]])
        noise = torch.randn(8, 2, 2, dtype=DTYPE,
                            generator=torch.Generator().manual_seed(101))
        endpoint_mean = F.one_hot(bank, 2).to(DTYPE).mean(0)
        q = empirical_bank_posterior(noise, 0.0, bank)
        torch.testing.assert_close(q, endpoint_mean.expand_as(q))
        one_step = noise + (q - noise)
        torch.testing.assert_close(one_step, endpoint_mean.expand_as(q))
        # The empirical teacher assigns probability 1/3 to [1, 1].
        self.assertTrue(bool((one_step.argmax(-1) == 0).all()))


@torch.no_grad()
def report():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(4611)
    rows = []
    for length in (4, 8, 31):
        for t in (0.25, 0.75):
            # Teacher law is first token uniform, followed by a 95%-stay Markov chain.
            states = []
            for _ in range(100):
                endpoint = torch.empty(length, dtype=torch.long)
                endpoint[0] = torch.randint(2, (), generator=rng)
                for j in range(1, length):
                    endpoint[j] = endpoint[j - 1] if torch.rand((), generator=rng) < 0.95 else 1 - endpoint[j - 1]
                states.append((1 - t) * torch.randn(length, 2, generator=rng, dtype=DTYPE)
                              + t * F.one_hot(endpoint, 2))
            for count in (16, 128):
                errors, effective = [], []
                for x in states:
                    exact = markov_posterior(x, t)
                    # Optimistic factorized proposal, using exact marginals with a 10% floor.
                    proposal = 0.9 * exact + 0.1 / 2
                    candidates = torch.multinomial(proposal, count, replacement=True, generator=rng).T
                    log_r = proposal[torch.arange(length)[None], candidates].log().sum(-1)
                    estimated, weights = posterior_from_candidates(
                        x, t, candidates, markov_log_prob(candidates) - log_r
                    )
                    errors.append(float((estimated - exact).abs().mean()))
                    effective.append(float(1 / weights.square().sum()))
                rows.append({"length": length, "t": t, "particles": count,
                             "posterior_MAE": sum(errors) / len(errors),
                             "MAE_standard_error": float(torch.tensor(errors, dtype=DTYPE).std() / len(errors)**0.5),
                             "mean_ESS": sum(effective) / len(effective)})
    rho = {str(length): (2 * (0.95**2 + 0.05**2))**(length - 1) for length in (4, 8, 31)}
    return {"seed": 4611, "teacher": "binary Markov, stay=0.95", "trials_per_row": 100,
            "states": "AR-endpoint interpolants; same states for particle counts at each length/time",
            "proposal": "0.9 * exact posterior marginals + 0.1 * uniform; optimistic oracle",
            "results": rows, "E_uniform_proposal_weight_squared": rho}


def refinement_position_weights(draft, acceptance=0.8):
    length = draft.numel()
    j = torch.arange(length, dtype=DTYPE)
    survival = acceptance**j * (1 - acceptance**(length - j)) / (1 - acceptance)
    greedy_teacher = torch.cat([torch.zeros(1, dtype=torch.long), draft[:-1]])
    breaks = torch.nonzero(draft != greedy_teacher).flatten()
    coefficient = torch.full((length,), 0.5, dtype=DTYPE)
    if breaks.numel():
        coefficient[breaks[0]] = 1.0
    weights = survival * coefficient
    return weights / weights.sum()


def target_metrics(q, exact, estimated, weights):
    discrepancy = q - exact
    fm_gradient = 2 * q * (discrepancy - (q * discrepancy).sum(-1, keepdim=True)) / q.size(0)
    ms_gradient = weights[:, None] * (q - estimated)
    dot = float((fm_gradient * ms_gradient).sum())
    return float((estimated - exact).abs().mean()), dot < -1e-14


def proposal_report():
    """A trained toy's actual 4-step Euler states, independent fixed AR bank."""
    torch.set_num_threads(1)
    model, training = train_toy()
    rng = torch.Generator().manual_seed(9137)
    bank = sample_markov(256, 31, torch.Generator().manual_seed(7319))
    with torch.no_grad():
        x = torch.randn(100, 31, 2, generator=rng)
        states = []
        previous = None
        for i in range(4):
            t = i / 4
            q = model(x, t)
            if i:
                states.append((t, x.double(), previous.double(), q.double()))
            x = x + (0.25 / (1 - t)) * (q - x)
            previous = q
        rows = []
        for t, xs, previous_q, current_q in states:
            for count in (32, 128):
                for method in ("student_factorized", "oracle_factorized", "fresh_AR", "guided_bank"):
                    errors, effective, conflicts, student_errors = [], [], [], []
                    for x, prev, q in zip(xs, previous_q, current_q):
                        exact = markov_posterior(x, t)
                        probabilities = 0.9 * (exact if method == "oracle_factorized" else prev) + 0.05
                        if method in ("student_factorized", "oracle_factorized"):
                            candidates = torch.multinomial(probabilities, count, replacement=True, generator=rng).T
                            log_r = probabilities[torch.arange(31)[None], candidates].log().sum(-1)
                        elif method == "fresh_AR":
                            candidates = sample_markov(count, 31, rng)
                            log_r = markov_log_prob(candidates)
                        else:
                            components = bank_component_weights(bank, x, t)
                            candidates, log_r = sample_bank_proposal(bank, components, probabilities, count, rng)
                        estimated, weights = posterior_from_candidates(x, t, candidates,
                                                                       markov_log_prob(candidates) - log_r)
                        position_weights = refinement_position_weights(prev.argmax(-1))
                        error, conflict = target_metrics(q, exact, estimated, position_weights)
                        errors.append(error)
                        conflicts.append(conflict)
                        effective.append(float(1 / weights.square().sum()))
                        student_errors.append(float((q - exact).abs().mean()))
                    rows.append({"t": t, "particles": count, "proposal": method,
                                 "posterior_MAE": sum(errors) / len(errors),
                                 "MAE_standard_error": float(torch.tensor(errors, dtype=DTYPE).std() / len(errors)**0.5),
                                 "mean_ESS": sum(effective) / len(effective),
                                 "negative_FM_dot_MS_fraction": sum(conflicts) / len(conflicts),
                                 "student_MAE_to_exact_posterior": sum(student_errors) / len(student_errors)})
    return {"teacher": "unconditional binary Markov, stay=0.95", "length": 31,
            "training": training, "evaluation_seed": 9137, "trials_per_row": 100,
            "states": "actual four-step Euler states of FM-trained MLP; shared across proposals",
            "bank_size": 256, "bank_seed": 7319, "mutation": 0.15, "component_floor": 0.1,
            "cost_caveat": "fresh_AR needs sequential AR draws per query; bank needs 256 joint draws once per context",
            "results": rows}


def block_distribution_metrics(samples):
    unique, counts = samples.unique(dim=0, return_counts=True)
    empirical = counts.to(DTYPE) / samples.size(0)
    full_tv = 1 - torch.minimum(empirical, markov_log_prob(unique).exp()).sum()
    powers = torch.tensor([8, 4, 2, 1])
    prefix_index = (samples[:, :4] * powers).sum(-1)
    prefix_probability = torch.bincount(prefix_index, minlength=16).to(DTYPE) / samples.size(0)
    prefix_tv = 0.5 * (prefix_probability - markov_log_prob(blocks(4)).exp()).abs().sum()
    return {"first4_joint_TV": float(prefix_tv), "empirical_full_block_TV": float(full_tv),
            "switch_rate": float((samples[:, 1:] != samples[:, :-1]).double().mean()),
            "all_zero_probability": float((samples == 0).all(-1).double().mean()),
            "all_one_probability": float((samples == 1).all(-1).double().mean())}


def bank_training_report(seeds=(912,), include_full_ar=True):
    """Proof-of-concept training on one fixed conditional empirical AR law."""
    torch.set_num_threads(1)
    bank = sample_markov(256, 31, torch.Generator().manual_seed(7319))
    rows = []
    methods = [("FM_bank", bank, 0.0), ("FM_bank_plus_aligned_ms", bank, 1.0)]
    if include_full_ar:
        methods.insert(0, ("FM_full_AR", None, 0.0))
    for seed, (label, endpoints, ms_weight) in itertools.product(seeds, methods):
        model, training = train_toy(seed=seed, bank=endpoints, ms_weight=ms_weight)
        with torch.no_grad():
            # Same evaluation noise for all methods and step counts.
            noise = torch.randn(4096, 31, 2, generator=torch.Generator().manual_seed(8121))
            decode = []
            state_errors = []
            for count in (1, 4, 16):
                x = noise.clone()
                for i in range(count):
                    t = i / count
                    q = model(x, t)
                    if count == 4 and i:
                        exact_bank = empirical_bank_posterior(x[:100], t, bank)
                        exact_ar = torch.stack([markov_posterior(v.double(), t) for v in x[:100]])
                        state_errors.append({"t": t,
                                             "student_to_bank_MAE": float((q[:100] - exact_bank).abs().mean()),
                                             "student_to_AR_MAE": float((q[:100] - exact_ar).abs().mean()),
                                             "bank_to_AR_MAE": float((exact_bank - exact_ar).abs().mean())})
                    x = x + (q - x) / (count - i)
                decode.append({"steps": count, **block_distribution_metrics(x.argmax(-1))})
            rows.append({"method": label, "training": training, "own_state_errors": state_errors,
                         "raw_unverified_samples": decode})
    reference = sample_markov(4096, 31, torch.Generator().manual_seed(8122))
    return {"teacher": "unconditional binary Markov, stay=0.95", "length": 31,
            "bank_size": 256, "bank_seed": 7319, "training_seeds": list(seeds), "evaluation_seed": 8121,
            "samples": 4096, "comparison": "equal 2000 steps, NOT equal compute; ms adds 1-4 drafter passes",
            "bank_law_vs_AR": block_distribution_metrics(bank),
            "iid_AR_reference_sampling_floor": block_distribution_metrics(reference), "results": rows}


if __name__ == "__main__":
    if sys.argv[1:] == ["--report"]:
        print(json.dumps(report(), indent=2))
    elif sys.argv[1:] == ["--proposal-report"]:
        print(json.dumps(proposal_report(), indent=2))
    elif sys.argv[1:] == ["--bank-training-report"]:
        print(json.dumps(bank_training_report(), indent=2))
    elif sys.argv[1:] == ["--bank-seed-report"]:
        print(json.dumps(bank_training_report(seeds=(912, 913, 914), include_full_ar=False), indent=2))
    else:
        unittest.main()
