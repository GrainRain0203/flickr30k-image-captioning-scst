from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
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

from src.config import DataConfig, ModelConfig
from src.data.dataset import (
    FlickrEvalDataset,
    FlickrSCSTDataset,
    build_transforms,
    eval_collate_fn,
    scst_collate_fn,
)
from src.data.splits import load_captions, load_splits
from src.data.vocab import Vocabulary
from src.engine import create_optimizer
from src.engine.scst_loop import evaluate_generation_metrics, train_scst_one_epoch
from src.metrics import validate_official_metrics_environment
from src.metrics.reward import CiderReward
from src.models import build_caption_model
from src.utils import load_checkpoint, save_checkpoint, set_seed
from src.utils.csv_logger import CSVLogger


METRIC_NAMES = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCST fine-tuning for image captioning")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--lr-decoder", type=float, default=1e-5)
    parser.add_argument("--lr-encoder", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--no-length-normalize-logprob", action="store_true")
    parser.add_argument("--unfreeze-encoder-last-stage", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--beam-size", type=int, default=7)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument(
        "--metrics-backend",
        default="official",
        choices=["official", "lightweight"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def _config_from_checkpoint(checkpoint: dict) -> tuple[DataConfig, ModelConfig, dict]:
    extra = checkpoint.get("extra", {})
    cfg = extra.get("config", {})
    if cfg:
        data_cfg = DataConfig(**cfg["data"])
        model_cfg = ModelConfig(**cfg["model"])
    elif extra.get("scst_config"):
        scst_cfg = extra["scst_config"]
        data_cfg = DataConfig(**scst_cfg["data"])
        model_cfg = ModelConfig(**scst_cfg["model"])
        cfg = {
            "data": scst_cfg["data"],
            "model": scst_cfg["model"],
        }
    elif extra.get("base_project_config"):
        cfg = extra["base_project_config"]
        data_cfg = DataConfig(**cfg["data"])
        model_cfg = ModelConfig(**cfg["model"])
    else:
        data_cfg = DataConfig()
        model_cfg = ModelConfig()
    return data_cfg, model_cfg, cfg


def _make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        base_name = Path(args.checkpoint).resolve().parent.name
        run_name = f"{base_name}_scst_cider"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.metrics_backend == "official":
        validate_official_metrics_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    data_cfg, model_cfg, base_project_config = _config_from_checkpoint(checkpoint)
    if args.data_root:
        data_cfg.data_root = args.data_root
    data_cfg.max_len = args.max_len
    if args.num_workers >= 0:
        data_cfg.num_workers = args.num_workers

    model_cfg.pretrained_encoder = False
    model_cfg.freeze_encoder = False
    model_cfg.encoder_weights_path = ""

    run_dir = _make_run_dir(args)
    scst_config = {
        "base_checkpoint": str(Path(args.checkpoint)),
        "args": vars(args),
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
    }
    _write_json(run_dir / "scst_config.json", scst_config)

    captions_by_image = load_captions(data_cfg.captions_path)
    splits = load_splits(data_cfg.split_path)
    vocab = Vocabulary.load(data_cfg.vocab_path)

    train_set = FlickrSCSTDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=splits["train"],
        vocab=vocab,
        max_len=data_cfg.max_len,
        transform=build_transforms(data_cfg.image_size, train=True),
    )
    val_set = FlickrEvalDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=splits["val"],
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
        collate_fn=partial(scst_collate_fn, pad_id=vocab.pad_id),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
        collate_fn=eval_collate_fn,
    )

    train_references = {
        image_name: captions_by_image[image_name]
        for image_name in splits["train"]
        if image_name in captions_by_image
    }
    reward_fn = CiderReward(train_references)

    model = build_caption_model(model_cfg, vocab_size=len(vocab), max_len=data_cfg.max_len).to(device)
    model.load_state_dict(checkpoint["model"])
    model.freeze_encoder()
    if args.unfreeze_encoder_last_stage:
        model.unfreeze_encoder_last_stage()

    optimizer = create_optimizer(
        model,
        lr_decoder=args.lr_decoder,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
    )
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    log_path = run_dir / "scst_log.csv"
    if log_path.exists():
        backup_path = run_dir / f"scst_log.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_path.replace(backup_path)
        print(f"Existing SCST log backed up to: {backup_path}")
    logger = CSVLogger(
        log_path,
        [
            "epoch",
            "train_loss",
            "scst_loss",
            "ce_loss",
            "sample_reward",
            "greedy_reward",
            "advantage",
            *METRIC_NAMES,
            "lr",
        ],
    )

    print(f"SCST device: {device}")
    print(f"Train images: {len(train_set)}, val images: {len(val_set)}, vocab: {len(vocab)}")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Run dir: {run_dir}")

    best_val_cider = float("-inf")
    if not args.skip_initial_eval:
        initial_metrics = evaluate_generation_metrics(
            model=model,
            dataloader=val_loader,
            vocab=vocab,
            device=device,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            metrics_backend=args.metrics_backend,
        )
        _write_json(run_dir / f"metrics_val_epoch00_{args.metrics_backend}.json", initial_metrics)
        initial_row = {
            "epoch": 0,
            "train_loss": "",
            "scst_loss": "",
            "ce_loss": "",
            "sample_reward": "",
            "greedy_reward": "",
            "advantage": "",
            **{key: f"{initial_metrics.get(key, 0.0):.6f}" for key in METRIC_NAMES},
            "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
        }
        logger.log(initial_row)
        best_val_cider = float(initial_metrics.get("CIDEr", 0.0))
        initial_extra = {
            "base_checkpoint": str(Path(args.checkpoint)),
            "base_project_config": base_project_config,
            "scst_config": scst_config,
            "stage": "scst_initial",
            "epoch_metrics": initial_metrics,
            "best_val_cider": best_val_cider,
        }
        for checkpoint_name in ("initial.pt", "best.pt"):
            save_checkpoint(
                run_dir / checkpoint_name,
                model=model,
                optimizer=optimizer if args.save_optimizer else None,
                scheduler=None,
                epoch=0,
                best_val_loss=-best_val_cider,
                extra=initial_extra,
            )
        print(f"[epoch 0] initial val CIDEr={best_val_cider:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_scst_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            reward_fn=reward_fn,
            vocab=vocab,
            device=device,
            max_len=args.max_len,
            pad_id=vocab.pad_id,
            ce_weight=args.ce_weight,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            normalize_advantage=args.normalize_advantage,
            length_normalize_logprob=not args.no_length_normalize_logprob,
            log_interval=args.log_interval,
        )
        metrics = evaluate_generation_metrics(
            model=model,
            dataloader=val_loader,
            vocab=vocab,
            device=device,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            metrics_backend=args.metrics_backend,
        )
        _write_json(run_dir / f"metrics_val_epoch{epoch:02d}_{args.metrics_backend}.json", metrics)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            **{key: f"{value:.6f}" for key, value in train_stats.items()},
            **{key: f"{metrics.get(key, 0.0):.6f}" for key in METRIC_NAMES},
            "lr": f"{lr:.8f}",
        }
        logger.log(row)

        val_cider = float(metrics.get("CIDEr", 0.0))
        extra = {
            "base_checkpoint": str(Path(args.checkpoint)),
            "base_project_config": base_project_config,
            "scst_config": scst_config,
            "stage": "scst_cider",
            "epoch_metrics": metrics,
            "best_val_cider": max(best_val_cider, val_cider),
        }
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer if args.save_optimizer else None,
            scheduler=None,
            epoch=epoch,
            best_val_loss=-max(best_val_cider, val_cider),
            extra=extra,
        )
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer if args.save_optimizer else None,
                scheduler=None,
                epoch=epoch,
                best_val_loss=-best_val_cider,
                extra=extra,
            )
            print(f"[epoch {epoch}] saved best SCST checkpoint: val CIDEr={best_val_cider:.4f}")

        print(
            f"[epoch {epoch}] loss={train_stats['train_loss']:.4f}, "
            f"adv={train_stats['advantage']:.4f}, val CIDEr={val_cider:.4f}"
        )

    print(f"SCST finished. Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
