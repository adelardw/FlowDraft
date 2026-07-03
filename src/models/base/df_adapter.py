
import torch
import torch.nn as nn
from torch.func import functional_call
from src.models.base.fte import FlowTimeEmbedding


class OrthrusAttentionAdapter(nn.Module):
    """Wrap ANY HF causal LM with a frozen AR path + a trainable DF path.

    Orthrus weight init: every ``nn.Linear`` whose attribute name matches
    ``w_names`` (``q_proj`` / ``k_proj`` / ``v_proj`` — configurable per model
    family) gets a trainable twin (``W_Q -> W_Q_diff`` etc.), created as a copy
    of the frozen weight and registered on the adapter — so the twins are in
    the computational graph and in ``state_dict()``, while the backbone itself
    is never modified.

    Routing is stateless. The DF path runs the *same* backbone through
    :func:`torch.func.functional_call`, which substitutes the matched
    projection weights with their twins for exactly one forward — no module
    surgery, no mode flags, nothing to restore afterwards. Norms, MLP,
    ``o_proj``, embeddings and the LM head are therefore shared by
    construction.

    Shared KV cache (Orthrus contract): the cache holds AR-path K/V of
    *committed* tokens only. Pass ``past_key_values`` (a mutable HF ``Cache``,
    e.g. ``DynamicCache``) to either path — ``use_cache=True`` is set
    automatically. The AR path appends to it as usual (the decode loop commits
    accepted tokens and crops the rest). The DF path *reads* the committed
    prefix but is guaranteed to leave the cache unchanged: the drafted block's
    K/V — computed with DF weights — are cropped away right after the forward,
    so they can never pollute verification.

    The adapter is pure mechanism and carries no training policy: it never
    touches the backbone's ``requires_grad`` or train/eval mode. Freezing the
    AR path is the caller's decision (``build_model`` freezes and evals the
    backbone before wrapping — that is what makes decoding provably lossless);
    keep it unfrozen and the AR path trains like any model. The DF twins are
    fresh ``nn.Parameter``s and hence trainable from creation. Gradient flow
    on either path is governed by ``requires_grad`` alone — with a frozen
    backbone the AR path builds no autograd graph at all, and ``forward``
    needs no changes if you unfreeze it.
    """

    def __init__(self, model, w_names=("q_proj", "k_proj", "v_proj")):
        super().__init__()
        self.model = model
        self.w_names = tuple(w_names)
        self._df_names, self.df_weights, self.n_dual = self._clone_df_weights()
        embed_weight = self.model.get_input_embeddings().weight
        self.time_embed = FlowTimeEmbedding(self.model.config.hidden_size).to(
            device=embed_weight.device, dtype=embed_weight.dtype
        )

    def _clone_df_weights(self):
        """Collect the projections matched by name and clone their parameters."""
        matched = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
            and (name in self.w_names or name.rsplit(".", 1)[-1] in self.w_names)
        ]
        if not matched:
            linears = sorted(
                {n.rsplit(".", 1)[-1] for n, m in self.model.named_modules() if isinstance(m, nn.Linear)}
            )
            raise ValueError(
                f"No nn.Linear matched w_names={self.w_names}. "
                f"Linear attribute names in this model: {linears}"
            )
        names, twins = [], []
        for module_name, module in matched:
            for param_name, param in module.named_parameters():
                names.append(f"{module_name}.{param_name}")
                twins.append(nn.Parameter(param.detach().clone()))
        return tuple(names), nn.ParameterList(twins), len(matched)

    def df_parameters(self):
        """Trainable parameters of the DF path (feed these to the optimizer)."""
        yield from self.df_weights
        yield from self.time_embed.parameters()

    def _df_inputs_embeds(self, input_ids):
        """Simplex point ``[B, T, V]`` (or int ids) -> ``inputs_embeds``.

        A point on the simplex is embedded as its barycentric mix of token
        embeddings: ``x @ E``. For a one-hot vertex this equals the ordinary
        embedding lookup, so int ids are also accepted.
        """
        embed = self.model.get_input_embeddings()
        if not input_ids.is_floating_point():
            return embed(input_ids)
        if input_ids.dim() != 3 or input_ids.size(-1) != embed.num_embeddings:
            raise ValueError(
                f"DF path expects simplex input [B, T, V={embed.num_embeddings}], "
                f"got {tuple(input_ids.shape)}"
            )
        return input_ids.to(embed.weight.dtype) @ embed.weight

    @staticmethod
    def _df_no_mask(attention_mask, batch, q_len, kv_len, dtype, device):
        """Express "NO mask" for the CFM path in the only way HF accepts.

        There is no attention mask in the diffusion/CFM path: every position
        attends to everything — the whole committed prefix (KV cache) and the
        whole block, future included. But ``attention_mask=None`` does NOT
        mean that to a HF model: with ``None`` it silently builds its default
        *causal* mask. The all-zeros additive 4D mask ``[B, 1, q_len, kv_len]``
        returned here (0 = attend everywhere) is the API's way of saying
        "don't mask anything", and the model uses it verbatim.

        The only thing ever masked out is padding (from a 2D ``[B, kv_len]``
        padding mask, if given) — pad rows are off the simplex and must never
        enter a softmax. A caller-supplied 4D mask passes through untouched.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        mask = torch.zeros(batch, 1, q_len, kv_len, dtype=dtype, device=device)
        if attention_mask is not None:  # [B, kv_len] padding mask
            padding = attention_mask[:, None, None, :].to(device) == 0
            mask = mask.masked_fill(padding, torch.finfo(dtype).min)
        return mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *,
        use_df: bool = False,
        s=None,
        t=None,
        past_key_values=None,
        **kwargs,
    ):
        ## input_ids - AR IDS or OHE IDS FOR DF!
        if past_key_values is not None:
            kwargs.update(past_key_values=past_key_values, use_cache=True)
        if use_df:
            # DF PATH: same backbone, Q/K/V substituted by their trainable
            # twins for this single call. The drafter reads the committed AR
            # prefix from the cache; its own block K/V (DF weights) are
            # cropped away so the shared cache stays AR-only.
            committed_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_embeds = self._df_inputs_embeds(input_ids)
            batch, q_len = inputs_embeds.shape[:2]
            if (s is None) != (t is None):
                raise ValueError("flow-map conditioning needs both s and t (or neither)")
            if s is not None:
                # Flow-map time conditioning, added to every block position.
                s = torch.as_tensor(s, device=inputs_embeds.device).reshape(-1).expand(batch)
                t = torch.as_tensor(t, device=inputs_embeds.device).reshape(-1).expand(batch)
                inputs_embeds = inputs_embeds + self.time_embed(s, t)[:, None, :].to(inputs_embeds.dtype)
            out = functional_call(
                self.model,
                dict(zip(self._df_names, self.df_weights)),
                kwargs=dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=self._df_no_mask(
                        attention_mask,
                        batch,
                        q_len,
                        committed_len + q_len,
                        inputs_embeds.dtype,
                        inputs_embeds.device,
                    ),
                    **kwargs,
                ),
            )
            if past_key_values is not None:
                past_key_values.crop(committed_len)
            return out
        # AR PATH. No no_grad here on purpose: with the backbone frozen,
        # autograd builds no graph anyway, and unfreezing parameters is all it
        # takes to fine-tune this path.
        if s is not None or t is not None:
            raise ValueError("s/t are flow-map (DF) conditioning; the AR path takes none")
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
