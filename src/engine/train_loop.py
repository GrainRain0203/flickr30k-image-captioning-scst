from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def create_optimizer(
    model: torch.nn.Module,
    lr_decoder: float,
    lr_encoder: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """为视觉 encoder 和其余模块设置不同学习率。"""

    encoder_params: list[torch.nn.Parameter] = []
    decoder_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups: list[dict[str, object]] = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr_encoder})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def caption_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler,
    device: torch.device,
    pad_id: int,
    label_smoothing: float,
    grad_clip: float,
    use_amp: bool,
    log_interval: int,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["images"].to(device, non_blocking=True)
        captions = batch["captions"].to(device, non_blocking=True)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            output = model(images, inputs, pad_id=pad_id)
            loss = caption_loss(output.logits, targets, pad_id, label_smoothing)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size
        if step % log_interval == 0:
            progress.set_postfix(loss=running_loss / max(1, seen))

    return running_loss / max(1, seen)


@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    pad_id: int,
    label_smoothing: float,
    use_amp: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    seen = 0

    for batch in tqdm(dataloader, desc="val", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        captions = batch["captions"].to(device, non_blocking=True)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        with autocast(enabled=use_amp):
            output = model(images, inputs, pad_id=pad_id)
            loss = caption_loss(output.logits, targets, pad_id, label_smoothing)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(1, seen)

