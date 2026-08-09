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
                 parameterisation: str = "pair", gated: bool = False):
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
        # Гейт: множитель на входе, а не слагаемое к нему.
        #
        # `verify_kl` считается при s = 0 на случайном розыгрыше приора, и его
        # таргет от розыгрыша не зависит: минимум требует, чтобы карта вход
        # ИГНОРИРОВАЛА, то есть d(pi)/dx = 0. Член самокоррекции при s >= 0.5
        # требует обратного. Различать режимы сеть может только через s, а s
        # входит ПРИБАВЛЕНИЕМ, и сложение якобиан не гейтит: замер показал, что
        # отклик на вход гасится лишь сдвигами в 81-325 крат медианной нормы
        # строки эмбеддинга, а в реальном диапазоне падает только до 0.43.
        # Умножение обнуляет якобиан по построению.
        #
        # `g = 1 + h(s,t)` с zero-init `h`: на старте гейт РОВНО единица, то
        # есть путь побитово совпадает с негейтованным, и обучение начинается
        # из той же рабочей точки.
        self.gate = None
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(2 * freq_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def _sinusoidal(self, x):
        """Times in [0, 1] -> sinusoidal features (fp32), one extra last axis.

        ``[B] -> [B, freq_dim]`` and ``[B, L] -> [B, L, freq_dim]``. The second
        form is what lets different POSITIONS of the same block carry different
        times, which is the only way to say "this position is confirmed clean,
        that one is a rejected guess, that one is fresh" — the state a decode
        cycle actually hands to the next one.
        """
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = x.to(torch.float32).unsqueeze(-1) * 1000.0 * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, s, t):
        """``(s, t)`` shaped ``[B]`` -> ``[B, hidden]``; ``[B, L]`` -> ``[B, L, hidden]``.

        Scalar-per-sequence times go through unchanged, bit for bit, so every
        existing checkpoint keeps loading and every existing run keeps its
        behaviour: the parameters are identical, only the batching of the input
        differs.
        """
        if s.shape != t.shape:
            raise ValueError(f"s and t must share a shape, got {tuple(s.shape)} and {tuple(t.shape)}")
        features = self._features(s, t)
        return self.mlp(features)

    def _features(self, s, t):
        second = t if self.parameterisation == "pair" else (t - s).clamp_min(0.0)
        features = torch.cat([self._sinusoidal(s), self._sinusoidal(second)], dim=-1)
        return features.to(self.mlp[0].weight.dtype)

    def input_gate(self, s, t):
        """Множитель на вход, `1 + h(s,t)`. `None`, когда гейт выключен.

        На старте ровно единица (zero-init `h`), поэтому включённый, но ещё не
        обученный гейт даёт побитово тот же прогон, что и выключенный.
        """
        if self.gate is None:
            return None
        if s.shape != t.shape:
            raise ValueError(
                f"s and t must share a shape, got {tuple(s.shape)} and {tuple(t.shape)}"
            )
        return 1.0 + self.gate(self._features(s, t))
