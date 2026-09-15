from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .cnn_encoders import CNNEncoder
from .decoders import DecoderOutput
from .feature_adapter import ImageFeatureAdapter


class CaptionModel(nn.Module):
    """图像描述统一模型：视觉 encoder + 特征适配器 + 文本 decoder。"""

    def __init__(
        self,
        encoder: CNNEncoder,
        adapter: ImageFeatureAdapter,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder

    @property
    def decoder_type(self) -> str:
        return self.decoder.decoder_type

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.encoder(images)
        return self.adapter(feature_map)

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        pad_id: int | None = None,
    ) -> DecoderOutput:
        visual_tokens = self.encode_images(images)
        return self.decoder(visual_tokens, captions, pad_id=pad_id)

    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    def unfreeze_encoder_last_stage(self) -> None:
        self.encoder.unfreeze_last_stage()

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> list[list[int]]:
        self.eval()
        visual_tokens = self.encode_images(images)
        batch_size = images.size(0)
        device = images.device

        if self.decoder_type == "transformer":
            sequences = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                output = self.decoder(visual_tokens, sequences)
                next_token = output.logits[:, -1].argmax(dim=-1)
                next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
                sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(eos_id)
                if finished.all():
                    break
            return sequences.cpu().tolist()

        state = self.decoder.init_state(visual_tokens)
        prev_tokens = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        sequences = [prev_tokens]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits, state, _ = self.decoder.step(prev_tokens, state, visual_tokens)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            sequences.append(next_token)
            finished |= next_token.eq(eos_id)
            prev_tokens = next_token
            if finished.all():
                break
        return torch.stack(sequences, dim=1).cpu().tolist()

    def sample_decode(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample captions and keep token log probabilities for SCST."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        def sample_next(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scaled_logits = logits / temperature
            if 0 < top_k < scaled_logits.size(-1):
                top_values, top_ids = torch.topk(scaled_logits, k=top_k, dim=-1)
                filtered_logits = torch.full_like(scaled_logits, float("-inf"))
                scaled_logits = filtered_logits.scatter(dim=-1, index=top_ids, src=top_values)
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = log_probs.exp()
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_log_prob = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            return next_token, next_log_prob

        visual_tokens = self.encode_images(images)
        batch_size = images.size(0)
        device = images.device

        if self.decoder_type == "transformer":
            sequences = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            log_prob_steps: list[torch.Tensor] = []
            mask_steps: list[torch.Tensor] = []
            for _ in range(max_len - 1):
                output = self.decoder(visual_tokens, sequences)
                next_token, next_log_prob = sample_next(output.logits[:, -1])
                active = ~finished
                next_token = torch.where(active, next_token, torch.full_like(next_token, eos_id))
                log_prob_steps.append(torch.where(active, next_log_prob, torch.zeros_like(next_log_prob)))
                mask_steps.append(active.float())
                sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(eos_id)
                if finished.all():
                    break
            log_probs = torch.stack(log_prob_steps, dim=1)
            mask = torch.stack(mask_steps, dim=1)
            return sequences, log_probs, mask

        state = self.decoder.init_state(visual_tokens)
        prev_tokens = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        sequences = [prev_tokens]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        log_prob_steps: list[torch.Tensor] = []
        mask_steps: list[torch.Tensor] = []
        for _ in range(max_len - 1):
            logits, state, _ = self.decoder.step(prev_tokens, state, visual_tokens)
            next_token, next_log_prob = sample_next(logits)
            active = ~finished
            next_token = torch.where(active, next_token, torch.full_like(next_token, eos_id))
            log_prob_steps.append(torch.where(active, next_log_prob, torch.zeros_like(next_log_prob)))
            mask_steps.append(active.float())
            sequences.append(next_token)
            finished |= next_token.eq(eos_id)
            prev_tokens = next_token
            if finished.all():
                break
        log_probs = torch.stack(log_prob_steps, dim=1)
        mask = torch.stack(mask_steps, dim=1)
        return torch.stack(sequences, dim=1), log_probs, mask

    @torch.no_grad()
    def beam_search_one(
        self,
        image: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
        length_penalty: float,
    ) -> list[int]:
        """单图 beam search。为了统一支持 LSTM/Transformer，这里按前缀重算 logits。"""

        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        visual_tokens = self.encode_images(image)
        beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]

        for _ in range(max_len - 1):
            candidates: list[tuple[list[int], float, bool]] = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, True))
                    continue
                prefix = torch.tensor(seq, dtype=torch.long, device=image.device).unsqueeze(0)
                output = self.decoder(visual_tokens, prefix)
                log_probs = F.log_softmax(output.logits[0, -1], dim=-1)
                top_scores, top_ids = torch.topk(log_probs, k=beam_size)
                for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                    new_seq = seq + [int(token_id)]
                    candidates.append((new_seq, score + float(token_score), token_id == eos_id))

            beams = sorted(
                candidates,
                key=lambda item: item[1] / (len(item[0]) ** length_penalty),
                reverse=True,
            )[:beam_size]
            if all(item[2] for item in beams):
                break

        best = max(beams, key=lambda item: item[1] / (len(item[0]) ** length_penalty))
        return best[0]

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int = 1,
        length_penalty: float = 0.8,
    ) -> list[list[int]]:
        if beam_size <= 1:
            return self.greedy_decode(images, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
        return [
            self.beam_search_one(
                image=images[i : i + 1],
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
            for i in range(images.size(0))
        ]
