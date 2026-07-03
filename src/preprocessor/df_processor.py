import torch
import torch.nn.functional as F


class DiffusionProcessor:
    """Tokenize text and build the diffusion-path input in one call.

    Returns (as a HF ``BatchEncoding``):
      - ``input_ids`` / ``attention_mask`` — for the frozen AR path (verification)
      - ``simplex`` — one-hot vocab vertices, the flow-map drafter's clean
        endpoint ``x_1`` on the simplex (only when ``return_simplex=True``)

    Notes
    -----
    * **Special tokens** (BOS/EOS/…): ordinary integer ids, one-hot like any
      other token — no special handling.
    * **Padding**: with ``zero_pad=True`` pad rows are zeroed via
      ``attention_mask`` (an explicit "ignore me" sentinel that is *off* the
      simplex). Correctness relies on ``attention_mask`` masking pad in both
      attention and the flow-map loss — never feed a zeroed row unmasked into a
      softmax / cross-entropy / corruption step. With ``zero_pad=False`` pad
      stays a valid ``one_hot(pad_id)`` vertex.
    * **vocab_size** must equal the model's LM-head width (``config.vocab_size``),
      NOT ``len(tokenizer)``: Llama pads its output matrix past the tokenizer
      vocab, and the simplex has to live in the same space the AR head produces,
      or the endpoint and the verifier won't align. Use ``from_model``.
    * **Memory**: ``simplex`` is ``[B, T, V]`` — with V≈128k this is large
      (B=8,T=512 → ~2 GB fp32). In the decode loop you typically only need the
      simplex for the drafted block (K positions), i.e. slice ids to the block
      before calling, giving ``[B, K, V]``.
    """

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    @classmethod
    def from_model(cls, tokenizer, model):
        """Take vocab_size straight from the model's LM head — the safe source."""
        return cls(tokenizer, vocab_size=model.config.vocab_size)

    def __call__(
        self,
        text,
        *,
        return_simplex: bool = True,
        simplex_dtype: torch.dtype = torch.float32,
        zero_pad: bool = True,
        **tokenizer_kwargs,
    ):
        tokenizer_kwargs.setdefault("return_tensors", "pt")
        tokenizer_kwargs.setdefault("padding", True)
        enc = self.tokenizer(text, **tokenizer_kwargs)

        if return_simplex:
            enc["simplex"] = self.to_simplex(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask") if zero_pad else None,
                dtype=simplex_dtype,
            )
        return enc

    def to_simplex(self, input_ids, attention_mask=None, dtype: torch.dtype = torch.float32):
        """``[*, T]`` int ids  ->  ``[*, T, V]`` one-hot on the simplex.

        If ``attention_mask`` is given, pad rows (mask == 0) are zeroed out.
        """
        simplex = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype)
        if attention_mask is not None:
            simplex = simplex * attention_mask.unsqueeze(-1).to(dtype)
        return simplex
