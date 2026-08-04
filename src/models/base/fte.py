import math

import torch
import torch.nn as nn

class FlowTimeEmbedding(nn.Module):
    """Flow-map time conditioning: ``(s, t) -> [B, hidden]`` vector.

    ``s`` (start) and ``t`` (target) in ``[0, 1]`` are each encoded with
    sinusoidal features (scaled by 1000, the standard trick for continuous
    diffusion time), concatenated and passed through a small MLP. The output
    layer is ZERO-initialized (adaLN-Zero style): at init the conditioning is
    exactly zero, so the DF path still reproduces the AR weights' behaviour
    and training starts from the Orthrus operating point.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: float = 10_000.0,
                 parameterisation: str = "pair"):
        super().__init__()
        if parameterisation not in ("pair", "jump"):
            raise ValueError(
                f"unknown parameterisation='{parameterisation}' (pair | jump)"
            )
        self.freq_dim = freq_dim
        self.max_period = max_period
        # ``pair`` encodes (s, t); ``jump`` encodes (s, t - s), i.e. the start
        # and the LENGTH of the jump. The reference categorical-flow-map
        # implementation uses the second, with a separate embedder per
        # argument. It is the easier target: under ``pair`` the identity jump
        # is the whole diagonal t = s, a line the network has to learn to
        # recognise, whereas under ``jump`` it is the single input value 0.
        # The deployed pair (0, 1) is likewise one point in both, but every
        # intermediate step of a multi-jump schedule shares its length with
        # steps taken from different starts, so the second form pools them.
        self.parameterisation = parameterisation
        self.mlp = nn.Sequential(
            nn.Linear(2 * freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _sinusoidal(self, x):
        """``[B]`` times in [0, 1] -> ``[B, freq_dim]`` features (fp32)."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = x.to(torch.float32)[:, None] * 1000.0 * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, s, t):
        second = t if self.parameterisation == "pair" else (t - s).clamp_min(0.0)
        features = torch.cat([self._sinusoidal(s), self._sinusoidal(second)], dim=-1)
        return self.mlp(features.to(self.mlp[0].weight.dtype))
