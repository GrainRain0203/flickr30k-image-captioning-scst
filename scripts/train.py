from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import DataConfig, EvalConfig, ModelConfig, ProjectConfig, TrainConfig, save_config
from src.data.dataset import FlickrCaptionDataset, build_transforms, caption_collate_fn
from src.data.splits import create_splits, load_captions, load_splits, save_splits
from src.data.vocab import Vocabulary, build_vocab
from src.engine import create_optimizer, train_one_epoch, validate_loss
from src.engine.scheduler import build_warmup_cosine_scheduler
from src.models import MODEL_SPECS, build_caption_model
from src.utils import save_checkpoint, set_seed
from src.utils.csv_logger import CSVLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Flickr30k 图像描述模型")
    parser.add_argument("--model", default="m4", choices=sorted(MODEL_SPECS))
    parser.add_argument("--data-root", default="Flickr30k")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=12)
    parser.add_argument("--epochs-stage2", type=int, default=8)
    parser.add_argument("--lr-decoder", type=float, default=3e-4)
    parser.add_argument("--lr-encoder", type=float, default=3e-5)
    parser.add_argument("--lr-decoder-stage2", type=float, default=5e-5)
    parser.add_argument("--lr-encoder-stage2", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--encoder-weights", default="", help="离线 ImageNet 预训练权重 .pth 路径")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def ensure_prepared_data(data_cfg: DataConfig) -> tuple[dict[str, list[str]], dict[str, list[str]], Vocabulary]:
    captions_by_image = load_captions(data_cfg.captions_path)
    existing_images = {path.name for path in data_cfg.image_path.glob("*.jpg")}
    captions_by_image = {
        image: captions
        for image, captions in captions_by_image.items()
        if image in existing_images
    }

    if data_cfg.split_path.exists():
        splits = load_splits(data_cfg.split_path)
    else:
        splits = create_splits(
            sorted(captions_by_image),
            train_ratio=data_cfg.train_ratio,
            val_ratio=data_cfg.val_ratio,
            seed=data_cfg.seed,
        )
        save_splits(splits, data_cfg.split_path)

    if data_cfg.vocab_path.exists():
        vocab = Vocabulary.load(data_cfg.vocab_path)
    else:
        train_captions = [
            caption
            for image_name in splits["train"]
            for caption in captions_by_image[image_name]
        ]
        vocab = build_vocab(
            train_captions,
            min_freq=data_cfg.min_freq,
            max_vocab_size=data_cfg.max_vocab_size,
        )
        vocab.save(data_cfg.vocab_path)

    return captions_by_image, splits, vocab


def make_dataloaders(
    data_cfg: DataConfig,
    captions_by_image: dict[str, list[str]],
    splits: dict[str, list[str]],
    vocab: Vocabulary,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_set = FlickrCaptionDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=splits["train"],
        vocab=vocab,
        max_len=data_cfg.max_len,
        transform=build_transforms(data_cfg.image_size, train=True),
    )
    val_set = FlickrCaptionDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=splits["val"],
        vocab=vocab,
        max_len=data_cfg.max_len,
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    collate_fn = partial(caption_collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def run_stage(
    stage_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    project_cfg: ProjectConfig,
    run_dir: Path,
    logger: CSVLogger,
    device: torch.device,
    pad_id: int,
    start_epoch: int,
    num_epochs: int,
    best_val_loss: float,
    lr_decoder: float,
    lr_encoder: float,
) -> tuple[int, float]:
    optimizer = create_optimizer(
        model,
        lr_decoder=lr_decoder,
        lr_encoder=lr_encoder,
        weight_decay=train_cfg.weight_decay,
    )
    total_steps = max(1, len(train_loader) * num_epochs)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, train_cfg.warmup_steps)
    scaler = GradScaler(enabled=train_cfg.amp and device.type == "cuda")
    use_amp = train_cfg.amp and device.type == "cuda"
    epoch = start_epoch

    for _ in range(num_epochs):
        epoch += 1
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            pad_id=pad_id,
            label_smoothing=train_cfg.label_smoothing,
            grad_clip=train_cfg.grad_clip,
            use_amp=use_amp,
            log_interval=train_cfg.log_interval,
        )
        val_loss = validate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            pad_id=pad_id,
            label_smoothing=train_cfg.label_smoothing,
            use_amp=use_amp,
        )
        lr = optimizer.param_groups[0]["lr"]
        encoder_lr = ""
        if len(optimizer.param_groups) > 1:
            encoder_lr = f", encoder_lr={optimizer.param_groups[1]['lr']:.2e}"
        logger.log(
            {
                "epoch": epoch,
                "stage": stage_name,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "lr": f"{lr:.8f}",
            }
        )
        print(
            f"[{stage_name}] epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"decoder_lr={lr:.2e}{encoder_lr}"
        )

        extra = {
            "config": asdict(project_cfg),
            "stage": stage_name,
        }
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            extra=extra,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                extra=extra,
            )
            print(f"保存 best checkpoint：val_loss={best_val_loss:.4f}")

    return epoch, best_val_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_cfg = DataConfig(
        data_root=args.data_root,
        image_size=args.image_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model_cfg = ModelConfig(
        model_name=args.model,
        pretrained_encoder=not args.no_pretrained,
        freeze_encoder=True,
        encoder_weights_path=args.encoder_weights,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_decoder=args.lr_decoder,
        lr_encoder=args.lr_encoder,
        lr_decoder_stage2=args.lr_decoder_stage2,
        lr_encoder_stage2=args.lr_encoder_stage2,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        seed=args.seed,
    )
    eval_cfg = EvalConfig(max_len=args.max_len)
    project_cfg = ProjectConfig(data=data_cfg, model=model_cfg, train=train_cfg, eval=eval_cfg)

    run_name = args.run_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(project_cfg, run_dir / "config.json")
    log_path = run_dir / "train_log.csv"
    if log_path.exists():
        backup_path = run_dir / f"train_log.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_path.replace(backup_path)
        print(f"检测到已有训练日志，已备份到：{backup_path}")

    captions_by_image, splits, vocab = ensure_prepared_data(data_cfg)

    train_loader, val_loader = make_dataloaders(
        data_cfg,
        captions_by_image,
        splits,
        vocab,
        batch_size=train_cfg.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_caption_model(model_cfg, vocab_size=len(vocab), max_len=data_cfg.max_len).to(device)
    print(MODEL_SPECS[args.model].description)
    print(f"训练设备：{device}")
    print(f"词表大小：{len(vocab)}，训练样本：{len(train_loader.dataset)}，验证样本：{len(val_loader.dataset)}")

    logger = CSVLogger(log_path, ["epoch", "stage", "train_loss", "val_loss", "lr"])
    epoch = 0
    best_val_loss = float("inf")

    if train_cfg.epochs_stage1 > 0:
        model.freeze_encoder()
        epoch, best_val_loss = run_stage(
            stage_name="stage1_frozen_encoder",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            project_cfg=project_cfg,
            run_dir=run_dir,
            logger=logger,
            device=device,
            pad_id=vocab.pad_id,
            start_epoch=epoch,
            num_epochs=train_cfg.epochs_stage1,
            best_val_loss=best_val_loss,
            lr_decoder=train_cfg.lr_decoder,
            lr_encoder=train_cfg.lr_encoder,
        )

    if train_cfg.epochs_stage2 > 0:
        model.unfreeze_encoder_last_stage()
        lr_decoder_stage2 = (
            train_cfg.lr_decoder_stage2
            if train_cfg.lr_decoder_stage2 is not None
            else train_cfg.lr_decoder
        )
        lr_encoder_stage2 = (
            train_cfg.lr_encoder_stage2
            if train_cfg.lr_encoder_stage2 is not None
            else train_cfg.lr_encoder
        )
        epoch, best_val_loss = run_stage(
            stage_name="stage2_finetune_last_stage",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            project_cfg=project_cfg,
            run_dir=run_dir,
            logger=logger,
            device=device,
            pad_id=vocab.pad_id,
            start_epoch=epoch,
            num_epochs=train_cfg.epochs_stage2,
            best_val_loss=best_val_loss,
            lr_decoder=lr_decoder_stage2,
            lr_encoder=lr_encoder_stage2,
        )

    print(f"训练完成。best checkpoint：{run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
