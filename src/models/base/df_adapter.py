import torch.nn as nn
from torch.func import functional_call


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
    construction, and ``past_key_values`` passed via ``**kwargs`` gives the
    shared KV cache on both paths.

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
        return iter(self.df_weights)

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

    def forward(self, input_ids=None, attention_mask=None, *, use_df: bool = False, **kwargs):
        ## input_ids - AR IDS or OHE IDS FOR DF!
        if use_df:
            # DF PATH: same backbone, Q/K/V substituted by their trainable
            # twins for this single call.
            return functional_call(
                self.model,
                dict(zip(self._df_names, self.df_weights)),
                kwargs=dict(
                    inputs_embeds=self._df_inputs_embeds(input_ids),
                    attention_mask=attention_mask,
                    **kwargs,
                ),
            )
        # AR PATH. No no_grad here on purpose: with the backbone frozen,
        # autograd builds no graph anyway, and unfreezing parameters is all it
        # takes to fine-tune this path.
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
