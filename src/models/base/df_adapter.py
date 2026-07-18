
import torch
import torch.nn as nn
from torch.func import functional_call
from src.models.base.fte import FlowTimeEmbedding


def _make_dual_pass_block_mask(batch, heads, q_len, ar_len, block_size, causal_limit):
    """Paper's sparse AR-cache + independent diffusion-block mask.

    Based on the MIT-licensed official Orthrus implementation
    (chiennv2000/orthrus, ``generate_dual_pass_mask``).  Import lazily: the
    CPU/macOS development build does not ship FlexAttention, while the H100
    training environment does.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Paper sparse baseline needs PyTorch FlexAttention. Install a CUDA "
            "PyTorch build that provides torch.nn.attention.flex_attention."
        ) from exc

    def mask_fn(b, h, q_idx, kv_idx):
        is_ar = kv_idx < ar_len
        allow_ar = is_ar & (kv_idx <= causal_limit[b, q_idx])
        allow_block = (~is_ar) & ((q_idx // block_size) == ((kv_idx - ar_len) // block_size))
        return allow_ar | allow_block

    return create_block_mask(
        mask_fn, B=batch, H=heads, Q_LEN=q_len, KV_LEN=ar_len + q_len, BLOCK_SIZE=128
    )


def _install_qwen3_flex_df_attention(module):
    """Patch Qwen3 attention with Orthrus's sparse diffusion-only path.

    The ordinary AR path remains the upstream HF implementation. During a DF
    call ``functional_call`` has already substituted the diffusion-attention
    modules with their trainable twins; this function reads frozen AR K/V from the cache and
    applies the official FlexAttention block mask without ever appending DF
    K/V to that cache.
    """
    if getattr(module, "_flowdraft_flex_df_installed", False):
        return
    try:
        from torch.nn.attention.flex_attention import flex_attention
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError:
        # Kept importable on non-CUDA development hosts. A paper-baseline run
        # will raise a direct, actionable error from _make_dual_pass_block_mask.
        return

    compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    original_forward = module.forward

    def forward(hidden_states, position_embeddings, attention_mask, past_key_values=None,
                use_df=False, ar_seq_len=None, flex_block_mask=None, **kwargs):
        if not use_df:
            return original_forward(hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            # DF K/V are transient. Read the committed AR cache directly
            # instead of calling Cache.update(), which would pollute it.
            cache_layer = past_key_values.layers[module.layer_idx]
            cache_len = ar_seq_len if ar_seq_len is not None else cache_layer.keys.shape[2]
            k = torch.cat([cache_layer.keys[:, :, :cache_len], k], dim=2)
            v = torch.cat([cache_layer.values[:, :, :cache_len], v], dim=2)
        if flex_block_mask is None:
            # Decoding, validation decode, and the non-paper variants retain
            # the original dense bidirectional DF behavior. FlexAttention is
            # only needed for paper training's many isolated blocks.
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                module.config._attn_implementation, eager_attention_forward
            )
            out, weights = attention_interface(
                module, q, k, v, attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                is_causal=False,
                **kwargs,
            )
            out = out.reshape(*input_shape, -1).contiguous()
            return module.o_proj(out), weights
        if past_key_values is None or ar_seq_len is None:
            raise ValueError("sparse DF FlexAttention requires an AR cache and ar_seq_len")
        # FA4's CuTeDSL FLASH backend is Hopper/Blackwell-only. Keep the
        # same sparse block semantics on older CUDA GPUs through FlexAttention
        # Triton, which is slower but avoids falling back to dense SDPA.
        is_hopper_or_newer = q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 9
        q_bs, kv_bs = flex_block_mask.BLOCK_SIZE
        kernel_options = (
            {"BACKEND": "FLASH", "sparse_block_size": (int(q_bs), int(kv_bs))}
            if is_hopper_or_newer
            else {"BACKEND": "TRITON"}
        )
        out = compiled(
            q, k, v, block_mask=flex_block_mask, enable_gqa=True,
            kernel_options=kernel_options,
        )
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return module.o_proj(out), None

    module.forward = forward
    module._flowdraft_flex_df_installed = True


class OrthrusAttentionAdapter(nn.Module):
    """Wrap ANY HF causal LM with a frozen AR path + a trainable DF path.

    Orthrus weight init: every ``nn.Linear`` whose attribute name matches
    ``w_names`` (for Qwen3: Q/K/V/O projections and Q/K norms) gets a
    trainable twin, created as a copy of the frozen parameter and registered
    on the adapter — so the twins are in the computational graph and in
    ``state_dict()``, while the backbone itself is never modified.

    Routing is stateless. The DF path runs the *same* backbone through
    :func:`torch.func.functional_call`, which substitutes the matched
    projection weights with their twins for exactly one forward — no module
    surgery, no mode flags, nothing to restore afterwards. Norms, MLP,
    ``o_proj``, embeddings and the LM head are therefore shared by
    construction. The official Orthrus Qwen implementation does *not* share
    the attention output projection or Q/K normalization with AR, so those
    modules belong in Qwen3's default ``w_names`` too.

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

    def __init__(self, model, w_names=("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")):
        super().__init__()
        self.model = model
        # Kept outside ``nn.Module`` registration. A compiled wrapper holds a
        # reference to ``self.model``; registering both would duplicate model
        # paths in state_dict/DDP traversal. The original remains the sole
        # registered backbone and the DF path always calls it directly.
        self.__dict__["_compiled_ar_model"] = None
        self.w_names = tuple(w_names)
        self._df_names, self.df_weights, self.n_dual = self._clone_df_weights()
        embed_weight = self.model.get_input_embeddings().weight
        # Orthrus feeds a dedicated mask embedding to the diffusion view.  It
        # must not mutate the frozen token embedding table, so keep it beside
        # the DF projections.  Initializing at the vocabulary mean is a
        # neutral starting point while allowing distillation to specialize it.
        self.mask_embedding = nn.Parameter(embed_weight.detach().mean(0, keepdim=True))
        self.time_embed = FlowTimeEmbedding(self.model.config.hidden_size).to(
            device=embed_weight.device, dtype=embed_weight.dtype
        )
        # The paper's training kernel is Qwen3-specific. Patch only Qwen3
        # attention modules; other model families retain the portable SDPA DF
        # implementation below.
        if self.model.config.model_type == "qwen3":
            for layer in self.model.model.layers:
                _install_qwen3_flex_df_attention(layer.self_attn)

    def enable_ar_compile(self, *, mode: str = "default", dynamic: bool = False) -> None:
        """Compile the fixed-shape frozen AR teacher forward.

        The baseline explicitly opts its packed ``[B, 2048]`` teacher call in
        through ``use_compiled_ar=True``. Generation and validation decoding
        retain variable cache lengths and stay eager; compiling them would
        exhaust Dynamo's shape-specialization cache. The DF path also remains
        eager because it uses :func:`torch.func.functional_call` to substitute
        the trainable diffusion projections for one forward. Keeping the
        compiled callable unregistered avoids a second backbone path in DDP
        and checkpoints.
        """
        self.__dict__["_compiled_ar_model"] = torch.compile(
            self.model, mode=mode, fullgraph=False, dynamic=dynamic
        )

    def _clone_df_weights(self):
        """Clone parameters of every configured diffusion-attention module.

        Attention projections are ``nn.Linear``, but Qwen3's per-head Q/K
        RMSNorms are custom modules. Matching modules rather than only Linear
        layers is what lets the DF path faithfully use ``q_norm_diff`` and
        ``k_norm_diff`` as in the official Orthrus implementation.
        """
        matched = [
            (name, module)
            for name, module in self.model.named_modules()
            if name and (name in self.w_names or name.rsplit(".", 1)[-1] in self.w_names)
            and any(True for _ in module.named_parameters(recurse=False))
        ]
        if not matched:
            parameterized_modules = sorted(
                {
                    n.rsplit(".", 1)[-1]
                    for n, m in self.model.named_modules()
                    if any(True for _ in m.named_parameters(recurse=False))
                }
            )
            raise ValueError(
                f"No parameterized module matched w_names={self.w_names}. "
                f"Parameterized module attribute names in this model: {parameterized_modules}"
            )
        names, twins = [], []
        for module_name, module in matched:
            for param_name, param in module.named_parameters():
                names.append(f"{module_name}.{param_name}")
                twins.append(nn.Parameter(param.detach().clone()))
        return tuple(names), nn.ParameterList(twins), len(matched)

    def df_parameters(self):
        """Trainable parameters of the DF path (feed these to the optimizer)."""
        yield from (parameter for parameter in self.df_weights if parameter.requires_grad)
        if self.mask_embedding.requires_grad:
            yield self.mask_embedding
        yield from (parameter for parameter in self.time_embed.parameters() if parameter.requires_grad)

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
        inputs_embeds=None,
        s=None,
        t=None,
        past_key_values=None,
        causal_limit=None,
        diffusion_block_size=None,
        use_compiled_ar: bool = False,
        **kwargs,
    ):
        ## input_ids - AR IDS or OHE IDS FOR DF!
        if past_key_values is not None:
            kwargs.update(past_key_values=past_key_values, use_cache=True)
        if use_df:
            # DF PATH: same backbone, diffusion-attention modules substituted
            # by their trainable twins for this single call. The drafter reads the committed AR
            # prefix from the cache; its own block K/V (DF weights) are
            # cropped away so the shared cache stays AR-only.
            committed_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            if inputs_embeds is None:
                inputs_embeds = self._df_inputs_embeds(input_ids)
            elif inputs_embeds.dim() != 3:
                raise ValueError(
                    "DF inputs_embeds must have shape [B, T, H], got "
                    f"{tuple(inputs_embeds.shape)}"
                )
            else:
                inputs_embeds = inputs_embeds.to(
                    dtype=self.model.get_input_embeddings().weight.dtype,
                    device=self.model.get_input_embeddings().weight.device,
                )
            batch, q_len = inputs_embeds.shape[:2]
            if (s is None) != (t is None):
                raise ValueError("flow-map conditioning needs both s and t (or neither)")
            if s is not None:
                # Flow-map time conditioning, added to every block position.
                s = torch.as_tensor(s, device=inputs_embeds.device).reshape(-1).expand(batch)
                t = torch.as_tensor(t, device=inputs_embeds.device).reshape(-1).expand(batch)
                inputs_embeds = inputs_embeds + self.time_embed(s, t)[:, None, :].to(inputs_embeds.dtype)
            flex_block_mask = None
            if causal_limit is not None:
                if diffusion_block_size is None:
                    raise ValueError("diffusion_block_size is required with causal_limit")
                if past_key_values is None:
                    raise ValueError("paper FlexAttention DF path requires an AR cache")
                flex_block_mask = _make_dual_pass_block_mask(
                    batch, self.model.config.num_attention_heads, q_len, committed_len,
                    int(diffusion_block_size), causal_limit,
                )
            # Qwen normally materializes a causal 4-D mask before calling
            # each attention layer. The patched DF attention ignores that
            # mask and consumes ``flex_block_mask`` instead, so pass an
            # explicit mapping of Nones to prevent recreating the dense mask.
            model_attention_mask = (
                {kind: None for kind in set(self.model.config.layer_types)}
                if flex_block_mask is not None
                else self._df_no_mask(
                    attention_mask, batch, q_len, committed_len + q_len,
                    inputs_embeds.dtype, inputs_embeds.device,
                )
            )
            out = functional_call(
                self.model,
                dict(zip(self._df_names, self.df_weights)),
                kwargs=dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=model_attention_mask,
                    use_df=use_df,
                    ar_seq_len=committed_len if flex_block_mask is not None else None,
                    flex_block_mask=flex_block_mask,
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
        ar_model = self.__dict__["_compiled_ar_model"]
        if ar_model is None or not use_compiled_ar:
            ar_model = self.model
        return ar_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
