import torch
import torch.nn as nn

from src.models.flowdraft_block_wise import FlowDraftBlockWise
from src.models.orthrus import Orthrus

PARTS = ("attn", "mlp", "attn+mlp", "layer")


def parse_plan(entries):
    """``["3:layer", "7:mlp", ...]`` -> ``{3: "layer", 7: "mlp"}``."""
    plan = {}
    for entry in entries or ():
        layer, part = str(entry).split(":")
        layer = int(layer)
        if part not in PARTS:
            raise ValueError(f"linearize: unknown part {part!r} in {entry!r} (one of {PARTS})")
        if layer in plan:
            raise ValueError(f"linearize: layer {layer} listed twice")
        plan[layer] = part
    return plan


class LinearizedDraftMixin:
    """A hybrid diffusion view: in the planned layers a component is replaced by
    a linear map, everywhere else the drafter is unchanged.

    ``train.linearize`` lists ``"<layer>:<part>"`` with part one of
      attn      the attention sublayer, x -> W x + b on its normed input;
      mlp       the MLP, x -> W x + b on its normed input;
      attn+mlp  both, as two separate maps;
      layer     the whole decoder layer, h -> h + W·LN(h) + b.
    The AR path is never touched, so decoding stays lossless.

    Where attention is replaced, that layer's diffusion twins are frozen at
    their AR copies. The map is then distilled towards what the frozen layer
    computes on the drafter's own input — for attention exactly
    Softmax(Q_ar K_arᵀ/√d) V_ar over the AR cache and the block:

        KL input:  W x + b, x detached — the term trains W and b only
        KL target: the frozen component on the same input, under no_grad

    It is a relative squared error, scale-free across layers, weighted by
    ``train.linear_distill_weight``. The drafter's own loss still flows through
    W x + b with x attached, so the rest of the network adapts to the hybrid.

    Outside training the frozen component is not computed at all: a replaced
    layer is genuinely skipped, which is what makes the speed measurable.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._plan = parse_plan(cfg.train.get("linearize", []))
        if not self._plan:
            raise ValueError("a linearized variant needs train.linearize")
        backbone = self.orthrus.model
        if getattr(backbone, "is_gradient_checkpointing", False):
            # Recomputation in backward runs outside the diffusion forward, so
            # it would take the AR branch and disagree with the forward pass.
            raise ValueError("train.linearize is incompatible with gradient checkpointing")
        layers = backbone.model.layers
        d = backbone.config.hidden_size
        for layer in self._plan:
            if not 0 <= layer < len(layers):
                raise ValueError(f"linearize: no layer {layer} in a {len(layers)}-layer model")

        surrogates = {}
        for layer, part in self._plan.items():
            for piece in ("attn", "mlp") if part == "attn+mlp" else (part,):
                surrogates[f"{layer}_{piece}"] = nn.Linear(d, d)
        for module in surrogates.values():
            # Zero start: a replaced component contributes nothing until the
            # distillation term has fitted it, a convex problem it solves fast.
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        device = backbone.get_input_embeddings().weight.device
        self.orthrus.linear_surrogates = nn.ModuleDict(surrogates).to(device)

        frozen = {
            f"model.layers.{layer}.self_attn."
            for layer, part in self._plan.items() if part in ("attn", "attn+mlp", "layer")
        }
        for twin, name in zip(self.orthrus.df_weights, self.orthrus._df_names):
            if any(name.startswith(prefix) for prefix in frozen):
                twin.requires_grad_(False)

        base_df_parameters = self.orthrus.df_parameters
        adapter = self.orthrus

        def df_parameters():
            yield from base_df_parameters()
            yield from adapter.linear_surrogates.parameters()

        self.orthrus.df_parameters = df_parameters

        self._df_active = False
        self._distill_terms = []
        adapter_forward = self.orthrus.forward

        def flagged_forward(*args, **kwargs):
            previous, self._df_active = self._df_active, bool(kwargs.get("use_df", False))
            try:
                return adapter_forward(*args, **kwargs)
            finally:
                self._df_active = previous

        self.orthrus.forward = flagged_forward

        for layer, part in self._plan.items():
            block = layers[layer]
            if part in ("attn", "attn+mlp"):
                self._wrap(block.self_attn, self._attn_forward(f"{layer}_attn"))
            if part in ("mlp", "attn+mlp"):
                self._wrap(block.mlp, self._mlp_forward(f"{layer}_mlp"))
            if part == "layer":
                self._wrap(block, self._layer_forward(f"{layer}_layer", block))

    @staticmethod
    def _wrap(module, make):
        """Install a diffusion-only forward around the module's own.

        Accelerate keeps the model's implementation in ``_old_forward`` and its
        device-dispatch wrapper in ``forward``; patch the implementation so
        the dispatch survives, as the adapter does for attention.
        """
        hooked = getattr(module, "_hf_hook", None) is not None and hasattr(module, "_old_forward")
        original = module._old_forward if hooked else module.forward
        replacement = make(original)
        if hooked:
            module._old_forward = replacement
        else:
            module.forward = replacement

    def _surrogate(self, key):
        return self.orthrus.linear_surrogates[key]

    def _linear_map(self, key, x):
        """W x + b in the map's own dtype, returned in the residual stream's.

        The backbone may run in bf16 without autocast; the map's parameters
        stay fp32 like every other optimizer-owned tensor here.
        """
        surrogate = self._surrogate(key)
        return surrogate(x.to(surrogate.weight.dtype)).to(x.dtype)

    def _distill(self, key, x, teacher):
        prediction = self._linear_map(key, x.detach())
        teacher = teacher.detach().to(prediction.dtype)
        error = (prediction - teacher).pow(2).mean()
        self._distill_terms.append(error / teacher.pow(2).mean().clamp_min(1e-12))

    def _attn_forward(self, key):
        def make(original):
            def forward(*args, **kwargs):
                if not self._df_active:
                    return original(*args, **kwargs)
                x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                if self.training:
                    with torch.no_grad():
                        if "hidden_states" in kwargs:
                            teacher = original(*args, **{**kwargs, "hidden_states": x.detach()})[0]
                        else:
                            teacher = original(x.detach(), *args[1:], **kwargs)[0]
                    self._distill(key, x, teacher)
                return self._linear_map(key, x), None
            return forward
        return make

    def _mlp_forward(self, key):
        def make(original):
            def forward(x):
                if not self._df_active:
                    return original(x)
                if self.training:
                    with torch.no_grad():
                        teacher = original(x.detach())
                    self._distill(key, x, teacher)
                return self._linear_map(key, x)
            return forward
        return make

    def _layer_forward(self, key, block):
        def make(original):
            def forward(*args, **kwargs):
                if not self._df_active:
                    return original(*args, **kwargs)
                h = args[0] if args else kwargs["hidden_states"]
                normed = block.input_layernorm(h)
                if self.training:
                    with torch.no_grad():
                        rest = dict(kwargs)
                        if args:
                            out = original(h.detach(), *args[1:], **rest)
                        else:
                            rest["hidden_states"] = h.detach()
                            out = original(**rest)
                        out = out[0] if isinstance(out, tuple) else out
                    self._distill(key, normed, out - h.detach())
                return h + self._linear_map(key, normed).to(h.dtype)
            return forward
        return make

    def training_step(self, batch, batch_idx):
        self._distill_terms = []
        loss = super().training_step(batch, batch_idx)
        if self._distill_terms:
            distill = torch.stack(self._distill_terms).mean()
        else:
            distill = torch.zeros((), device=loss.device)
        # Logged on every rank in the same order, empty block or not.
        self.log("loss/linear_distill", distill, on_step=True, on_epoch=False, sync_dist=True)
        self._distill_terms = []
        return loss + float(self.cfg.train.get("linear_distill_weight", 1.0)) * distill


class LinearOrthrus(LinearizedDraftMixin, Orthrus):
    """Orthrus with a hybrid diffusion view (``train.linearize``)."""


class LinearFlowDraft(LinearizedDraftMixin, FlowDraftBlockWise):
    """The block-wise flow map with a hybrid diffusion view (``train.linearize``)."""
