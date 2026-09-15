from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.vocab import Vocabulary
from src.engine.train_loop import caption_loss
from src.metrics import compute_caption_metrics, compute_official_caption_metrics
from src.metrics.reward import CiderReward


def _decode_sequences(vocab: Vocabulary, sequences: torch.Tensor | list[list[int]]) -> list[str]:
    if isinstance(sequences, torch.Tensor):
        seq_list = sequences.detach().cpu().tolist()
    else:
        seq_list = sequences
    return [vocab.decode(token_ids) for token_ids in seq_list]


def _normalize_advantage(advantage: torch.Tensor) -> torch.Tensor:
    if advantage.numel() <= 1:
        return advantage
    std = advantage.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return advantage - advantage.mean()
    return (advantage - advantage.mean()) / (std + 1e-8)


def train_scst_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    reward_fn: CiderReward,
    vocab: Vocabulary,
    device: torch.device,
    max_len: int,
    pad_id: int,
    ce_weight: float,
    label_smoothing: float,
    grad_clip: float,
    use_amp: bool,
    num_samples: int,
    temperature: float,
    top_k: int,
    normalize_advantage: bool,
    length_normalize_logprob: bool,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_scst = 0.0
    running_ce = 0.0
    running_sample_reward = 0.0
    running_greedy_reward = 0.0
    running_advantage = 0.0
    seen = 0

    progress = tqdm(dataloader, desc="scst-train", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["images"].to(device, non_blocking=True)
        captions = batch["captions"].to(device, non_blocking=True)
        image_names = batch["image_names"]
        batch_size = images.size(0)

        with torch.no_grad():
            greedy_sequences = model.greedy_decode(
                images,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_len=max_len,
            )
        model.train()

        greedy_captions = _decode_sequences(vocab, greedy_sequences)
        greedy_rewards = torch.tensor(
            reward_fn.score_many(image_names, greedy_captions),
            dtype=torch.float32,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        sample_losses: list[torch.Tensor] = []
        sample_reward_values: list[torch.Tensor] = []
        advantage_values: list[torch.Tensor] = []

        for _ in range(num_samples):
            with autocast(enabled=use_amp):
                sample_sequences, sample_log_probs, sample_mask = model.sample_decode(
                    images,
                    bos_id=vocab.bos_id,
                    eos_id=vocab.eos_id,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k,
                )

            sample_captions = _decode_sequences(vocab, sample_sequences)
            sample_rewards = torch.tensor(
                reward_fn.score_many(image_names, sample_captions),
                dtype=torch.float32,
                device=device,
            )
            raw_advantage = sample_rewards - greedy_rewards
            advantage = (
                _normalize_advantage(raw_advantage)
                if normalize_advantage
                else raw_advantage
            )
            sequence_log_probs = (sample_log_probs * sample_mask).sum(dim=1)
            if length_normalize_logprob:
                token_counts = sample_mask.sum(dim=1).clamp_min(1.0)
                sequence_log_probs = sequence_log_probs / token_counts
            sample_losses.append(-(advantage.detach() * sequence_log_probs).mean())
            sample_reward_values.append(sample_rewards.detach())
            advantage_values.append(raw_advantage.detach())

        scst_loss = torch.stack(sample_losses).mean()
        ce_loss = torch.zeros((), dtype=scst_loss.dtype, device=device)
        if ce_weight > 0:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            with autocast(enabled=use_amp):
                output = model(images, inputs, pad_id=pad_id)
                ce_loss = caption_loss(output.logits, targets, pad_id, label_smoothing)

        loss = scst_loss + ce_weight * ce_loss
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        mean_sample_reward = torch.stack(sample_reward_values).mean()
        mean_advantage = torch.stack(advantage_values).mean()
        running_loss += float(loss.item()) * batch_size
        running_scst += float(scst_loss.item()) * batch_size
        running_ce += float(ce_loss.item()) * batch_size
        running_sample_reward += float(mean_sample_reward.item()) * batch_size
        running_greedy_reward += float(greedy_rewards.mean().item()) * batch_size
        running_advantage += float(mean_advantage.item()) * batch_size
        seen += batch_size

        if step % log_interval == 0:
            progress.set_postfix(
                loss=running_loss / max(1, seen),
                adv=running_advantage / max(1, seen),
            )

    denom = max(1, seen)
    return {
        "train_loss": running_loss / denom,
        "scst_loss": running_scst / denom,
        "ce_loss": running_ce / denom,
        "sample_reward": running_sample_reward / denom,
        "greedy_reward": running_greedy_reward / denom,
        "advantage": running_advantage / denom,
    }


@torch.no_grad()
def evaluate_generation_metrics(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    vocab: Vocabulary,
    device: torch.device,
    max_len: int,
    beam_size: int,
    length_penalty: float,
    metrics_backend: str,
) -> dict[str, float]:
    model.eval()
    hypotheses: dict[str, str] = {}
    references: dict[str, list[str]] = {}

    for batch in tqdm(dataloader, desc="scst-val", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        generated = model.generate(
            images,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        for image_name, token_ids, refs in zip(
            batch["image_names"],
            generated,
            batch["references"],
        ):
            hypotheses[image_name] = vocab.decode(token_ids)
            references[image_name] = refs

    if metrics_backend == "official":
        return compute_official_caption_metrics(hypotheses, references)
    if metrics_backend == "lightweight":
        return compute_caption_metrics(hypotheses, references)
    raise ValueError(f"Unsupported metrics backend: {metrics_backend}")
