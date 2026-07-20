import torch
import torch.nn.functional as F

from src.models.lit_orthrus_block_wise import FlowMapOrthrusBlockWise


class FlowMapOrthrusBaselineBlockWise(FlowMapOrthrusBlockWise):
    """The Orthrus baseline in the brief's BLOCK-CAUSAL geometry.

    Causal to the cached context, bidirectional within the block, ONE DF
    forward that reconstructs a fully-masked block (every fresh position is
    the barycenter — the simplex-native ``[MASK]``), clean in-block anchor —
    exactly the configuration the drafter decodes in. This removes the
    training-geometry confound of the full-sequence baseline: both the
    baseline and the flow drafter now train in the decode configuration and
    the comparison is geometry-for-geometry.

    No time conditioning — ``time_embed`` stays dark. Window math, cache and
    anchor come from the parent's :meth:`_prepare_block`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Block-wise baseline also never supplies flow-map times.
        self.orthrus.time_embed.requires_grad_(False)

    def _df_forward(self, x_block, anchor, ctx_mask, cache, s=None, t=None):
        """Masked Orthrus output alignment in block-wise geometry.

        Unlike a flow-map endpoint predictor, a causal-LM diffusion head uses
        the output at the clean anchor to predict the first fresh token.  For
        ``[anchor, mask_1, ..., mask_K]``, rows ``[:-1]`` therefore align with
        the K teacher distributions and the final mask row is unused.
        """
        x_in = torch.cat([anchor, x_block], dim=1)
        logits = self.orthrus(
            x_in, ctx_mask, use_df=True, past_key_values=cache
        ).logits
        return logits[:, :-1]

    def _masked_step(self, batch):
        teacher_logits, block_ids, ctx_mask, block_mask, cache, anchor = self._prepare_block(batch)
        vocab = self.df_processor.vocab_size
        barycenter = torch.full(
            (block_ids.size(0), block_ids.size(1), vocab), 1.0 / vocab, device=self.device
        )
        df_logits = self._df_forward(barycenter, anchor, ctx_mask, cache, s=None, t=None)
        loss = self._masked_kl(
            F.log_softmax(teacher_logits.float(), -1),
            F.log_softmax(df_logits.float(), -1),
            block_mask.bool(),
        )
        return loss, df_logits, teacher_logits, block_mask

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._masked_step(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, df_logits, teacher_logits, block_mask = self._masked_step(batch)
        live = block_mask.bool()
        agree = (df_logits.argmax(-1) == teacher_logits.argmax(-1))[live]
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/teacher_agreement", agree.float().mean(), sync_dist=True)
        self._maybe_decode_val(batch, batch_idx)
        return loss

    def _draft_block(self, cache, block_size, times, sample: bool = False, anchor_token=None):
        """Single-step barycenter draft — same step as the full-sequence
        baseline (`lit_orthrus_baseline`), kept in sync by the shared tests."""
        if len(times) != 2:
            raise ValueError("the masked baseline drafts in exactly one step: use jumps=1")
        vocab = self.df_processor.vocab_size
        x_in = torch.full((1, block_size, vocab), 1.0 / vocab, device=self.device)
        if anchor_token is not None:
            anchor = F.one_hot(anchor_token.view(1, 1), vocab).to(x_in.dtype)
            x_in = torch.cat([anchor, x_in], dim=1)
        mask = torch.ones(1, cache.get_seq_length() + x_in.size(1), dtype=torch.long, device=self.device)
        logits = self.orthrus(x_in, mask, use_df=True, past_key_values=cache).logits
        q = (logits[:, :-1] if anchor_token is not None else logits).float().softmax(-1)
        ids = torch.multinomial(q[0], 1).view(1, -1) if sample else q.argmax(-1)
        return ids, q
