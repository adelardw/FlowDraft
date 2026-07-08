import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask




class RoPE(nn.Module):
    def __call__(self, num_tokens: int, d_model: int, base: int = 10000):
        theta = 1. / base ** (torch.arange(0, d_model,  2) / d_model)
        ids = torch.arange(num_tokens)
        # ids[[:, None] --> [num_tokens, 1]
        # theta[None, :] --> [1, dmodel // 2]
        ids_theta = ids[:, None] * theta[None, :]
        pe = torch.stack([ids_theta.cos(), ids_theta.sin()])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        bs, s, n, d = x.shape()
        # self.pe.shape --> [s, d // 2]
        rope_reshape = self.pe.reshape(1, s, )


class OrthrusAttnCFMBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, mask, use_df: bool):
        if use_df:
            return ...

        return ...

class OrthrusCFM(nn.Module):
    def __init__(self, num_attn_blocks: int,
                       attn_implementation: str,
                       use_cache: bool,
                       vocab_size: int,
                       d_model: int):
        super().__init__()
