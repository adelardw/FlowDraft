import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from src.models.lit_orthrus_block_wise import FlowMapOrthrusBlockWise


class FlowMapOrthrusBaseline(FlowMapOrthrusBlockWise):
    """Paper-faithful Orthrus masked-block baseline.

    A frozen AR pass builds the clean cache.  The diffusion pass then contains
    ``anchors_per_sequence`` independent blocks of size ``K`` (one visible
    anchor plus ``K - 1`` masks). Its
    4-D mask lets a block attend only to the AR prefix before its anchor and
    to itself bidirectionally; it can never read future clean tokens or any
    other block.  This is the dual-pass block-masking geometry of Orthrus,
    rather than full-sequence random token masking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orthrus.time_embed.requires_grad_(False)

    def _sample_anchors(self, attention_mask, block: int, count: int):
        """Uniform anchor positions shared across the micro-batch.

        The paper uses a micro-batch of one; sharing positions also keeps a
        larger local batch compatible with one cache and one sparse mask.
        """
        min_prefix = int(self.cfg.train.get("min_prefix", 1))
        true_min = int(attention_mask.sum(dim=1).min())
        # ``block`` includes the visible anchor; only the remaining K-1
        # positions are drafted.
        high = true_min - (block - 1)
        if high <= min_prefix:
            raise ValueError(
                f"sequence length {true_min} cannot fit a block of size {block}"
            )
        return torch.randint(min_prefix, high, (count,), device=attention_mask.device)


    def _masked_step(self, batch):
        ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        block = int(self.cfg.train.get("block_size", 32))
        count = int(self.cfg.train.get("anchors_per_sequence", 256))
        anchors = self._sample_anchors(attention_mask, block, count)

        # One clean AR pass supplies both exact teacher rows and the shared
        # AR-only KV cache.  The DF forward below never appends to it.
        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad():
            teacher_full = self.orthrus(ids, attention_mask, past_key_values=cache).logits

        width = block
        drafted = block - 1
        if drafted <= 0:
            raise ValueError("baseline block_size must be at least 2 (anchor + one mask)")
        anchor_ids = ids[:, anchors]  # [B, A]
        embed = self.orthrus.model.get_input_embeddings()
        anchor_embeds = embed(anchor_ids).unsqueeze(2)
        mask_embed = self.orthrus.mask_embedding.to(anchor_embeds.dtype)[None, None]
        masked_embeds = mask_embed.expand(ids.size(0), count, drafted, -1)
        x_in = torch.cat([anchor_embeds, masked_embeds], dim=2).flatten(1, 2)

        # Each synthetic block keeps the original absolute positions, even
        # though all blocks are concatenated for one efficient DF forward.
        offsets = torch.arange(width, device=ids.device)
        position_ids = (anchors[:, None] + offsets).flatten()[None].expand(ids.size(0), -1)
        # The official FlexAttention mask uses one causal limit per synthetic
        # query. It represents this sparse relation directly, instead of a
        # dense [8192, 10240] SDPA mask.
        causal_limit = anchors.repeat_interleave(width)[None].expand(ids.size(0), -1)
        df_all = self.orthrus(
            inputs_embeds=x_in,
            use_df=True,
            past_key_values=cache,
            position_ids=position_ids,
            causal_limit=causal_limit,
            diffusion_block_size=width,
        ).logits.view(ids.size(0), count, width, -1)
        df_logits = df_all[:, :, 1:]

        # AR row p predicts token p+1, exactly the first drafted position.
        gather = anchors[:, None] + torch.arange(drafted, device=ids.device)
        teacher_logits = teacher_full[:, gather]
        live = attention_mask[:, gather + 1].bool()
        loss = self._masked_kl_chunked(teacher_logits, df_logits, live)
        return loss, df_logits, teacher_logits, live

    def _masked_kl_chunked(self, teacher_logits, draft_logits, live):
        vocab = draft_logits.size(-1)
        teacher_flat = teacher_logits.reshape(-1, vocab)
        draft_flat = draft_logits.reshape(-1, vocab)
        live_flat = live.reshape(-1).float()
        n_live = live_flat.sum()

        def term(draft_chunk, teacher_chunk, live_chunk):
            log_draft = F.log_softmax(draft_chunk.float(), -1)
            log_teacher = F.log_softmax(teacher_chunk.float(), -1)
            kl = (log_teacher.exp() * (log_teacher - log_draft)).sum(-1)
            return (kl * live_chunk).sum()

        chunk = int(self.cfg.train.get("kl_chunk", 4096)) 
        total = draft_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, draft_flat.size(0), chunk):
            stop = start + chunk
            total = total + checkpoint(
                term,
                draft_flat[start:stop],
                teacher_flat[start:stop],
                live_flat[start:stop],
                use_reentrant=False,
            )
        return total / n_live.clamp(min=1.0)

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._masked_step(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, df_logits, teacher_logits, live = self._masked_step(batch)
        agree = (df_logits.argmax(-1) == teacher_logits.argmax(-1))[live]
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/teacher_agreement", agree.float().mean(), sync_dist=True)
        self._maybe_decode_val(batch, batch_idx)
        return loss

    def _draft_block(self, cache, block_size, times, sample: bool = False, anchor_token=None):
        if len(times) != 2:
            raise ValueError("the masked baseline drafts in exactly one step: use jumps=1")
        embed = self.orthrus.model.get_input_embeddings()
        # Orthrus defines K as the complete parallel block, including its
        # clean anchor.  Thus a K=32 baseline proposal contains 31 fresh
        # tokens.
        drafted = block_size - 1
        if drafted <= 0:
            raise ValueError("baseline block_size must be at least 2 (anchor + one mask)")
        masks = self.orthrus.mask_embedding.to(self.device).expand(1, drafted, -1)
        if anchor_token is not None:
            anchor = embed(anchor_token.view(1, 1))
            x_in = torch.cat([anchor, masks], dim=1)
        else:
            x_in = masks
        mask = torch.ones(1, cache.get_seq_length() + x_in.size(1), dtype=torch.long, device=self.device)
        logits = self.orthrus(
            attention_mask=mask, inputs_embeds=x_in, use_df=True, past_key_values=cache
        ).logits
        q = (logits[:, 1:] if anchor_token is not None else logits).float().softmax(-1)
        ids = torch.multinomial(q[0], 1).view(1, -1) if sample else q.argmax(-1)
        return ids, q
