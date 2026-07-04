import torch
import torch.nn.functional as F

from src.models.lit_orthrus import FlowMapOrthrus


class FlowMapOrthrusBaseline(FlowMapOrthrus):
    """Orthrus masked-diffusion baseline — the comparison row FlowDraft must beat.

    Single-step masked diffusion drafter: no time conditioning, ONE DF
    forward per draft, block positions conditionally independent — the
    acceptance ceiling the flow-map drafter is built to raise. The "masked"
    input on the simplex is the barycenter (uniform over V): a
    zero-information point, the simplex analogue of a [MASK] token.

    Inherits everything else from the fixed variant (adapter, checkpoints,
    generate/ar_generate loop, eval compatibility); overrides only the
    training objective and the draft step. ``time_embed`` is never used and
    receives no gradients.
    """

    # --- training: mask-and-reconstruct distillation ---------------------------

    def _masked_step(self, batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        live = mask.bool()
        x1 = batch.get("simplex")
        if x1 is None:
            x1 = self.df_processor.to_simplex(ids, attention_mask=mask)

        # mask rate ~ U(0, 1] per sample — the single-step masked-diffusion
        # recipe; masked positions get the barycenter, the rest stay one-hot
        rate = torch.rand(ids.size(0), 1, device=ids.device).clamp(min=0.05)
        masked = (torch.rand_like(ids, dtype=torch.float) < rate) & live
        barycenter = torch.full_like(x1, 1.0 / x1.size(-1))
        x_in = torch.where(masked[..., None], barycenter, x1)
        x_in = x_in * mask[..., None].to(x_in.dtype)  # pads stay the zero sentinel

        with torch.no_grad():
            teacher_logits = self.orthrus(ids, mask).logits
        df_logits = self.orthrus(x_in, mask, use_df=True).logits  # no (s, t)

        # df position i reconstructs token i; teacher_logits[:, i-1] is the
        # AR distribution of token i — hence the shift; only masked
        # positions carry loss (the others were given away clean)
        loss = self._masked_kl(
            F.log_softmax(teacher_logits[:, :-1].float(), -1),
            F.log_softmax(df_logits[:, 1:].float(), -1),
            masked[:, 1:] & live[:, 1:],
        )
        return loss, df_logits, teacher_logits, masked

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._masked_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, df_logits, teacher_logits, masked = self._masked_step(batch)
        live = batch["attention_mask"][:, 1:].bool() & masked[:, 1:]
        agree = (df_logits[:, 1:].argmax(-1) == teacher_logits[:, :-1].argmax(-1))[live]
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/teacher_agreement", agree.float().mean())
        return loss

    # --- generation: one forward per draft --------------------------------------

    def _draft_block(self, cache, block_size, times):
        """Barycenter block -> ONE DF forward -> argmax.

        Only ``jumps=1`` is meaningful: a single-step drafter has no jump
        schedule (that limitation is the baseline's whole point).
        """
        if len(times) != 2:
            raise ValueError("the masked baseline drafts in exactly one step: use jumps=1")
        vocab = self.df_processor.vocab_size
        x_in = torch.full((1, block_size, vocab), 1.0 / vocab, device=self.device)
        mask = torch.ones(1, cache.get_seq_length() + block_size, dtype=torch.long, device=self.device)
        logits = self.orthrus(x_in, mask, use_df=True, past_key_values=cache).logits
        return logits.argmax(-1)
