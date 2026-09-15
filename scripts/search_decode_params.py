from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DataConfig, ModelConfig
from src.data.dataset import FlickrEvalDataset, build_transforms, eval_collate_fn
from src.data.splits import load_captions, load_splits
from src.data.vocab import Vocabulary
from src.metrics import (
    compute_caption_metrics,
    compute_official_caption_metrics,
    validate_official_metrics_environment,
)
from src.models import build_caption_model
from src.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证集解码参数搜索")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[7])
    parser.add_argument(
        "--length-penalties",
        nargs="+",
        type=float,
        default=[0.8],
    )
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--metrics-backend",
        default="official",
        choices=["official", "lightweight"],
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _config_from_checkpoint(checkpoint: dict) -> tuple[DataConfig, ModelConfig]:
    extra = checkpoint.get("extra", {})
    cfg = extra.get("config", {})
    if cfg:
        data_cfg = DataConfig(**cfg["data"])
        model_cfg = ModelConfig(**cfg["model"])
    elif extra.get("scst_config"):
        scst_cfg = extra["scst_config"]
        data_cfg = DataConfig(**scst_cfg["data"])
        model_cfg = ModelConfig(**scst_cfg["model"])
    elif extra.get("base_project_config"):
        cfg = extra["base_project_config"]
        data_cfg = DataConfig(**cfg["data"])
        model_cfg = ModelConfig(**cfg["model"])
    else:
        data_cfg = DataConfig()
        model_cfg = ModelConfig()
    return data_cfg, model_cfg


def _decode_tag(beam_size: int, length_penalty: float) -> str:
    lp = str(length_penalty).replace(".", "p")
    return f"beam{beam_size}_lp{lp}"


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["image", "hypothesis", "ref1", "ref2", "ref3", "ref4", "ref5"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_metrics(
    hypotheses: dict[str, str],
    references: dict[str, list[str]],
    backend: str,
) -> dict[str, float]:
    if backend == "official":
        return compute_official_caption_metrics(hypotheses, references)
    return compute_caption_metrics(hypotheses, references)


def main() -> None:
    args = parse_args()
    if args.metrics_backend == "official":
        validate_official_metrics_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    data_cfg, model_cfg = _config_from_checkpoint(checkpoint)
    if args.data_root:
        data_cfg.data_root = args.data_root
    data_cfg.max_len = args.max_len
    model_cfg.pretrained_encoder = False
    model_cfg.freeze_encoder = False
    model_cfg.encoder_weights_path = ""

    captions_by_image = load_captions(data_cfg.captions_path)
    splits = load_splits(data_cfg.split_path)
    vocab = Vocabulary.load(data_cfg.vocab_path)

    dataset = FlickrEvalDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=splits[args.split],
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
        collate_fn=eval_collate_fn,
    )

    model = build_caption_model(model_cfg, vocab_size=len(vocab), max_len=data_cfg.max_len).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for beam_size in args.beam_sizes:
        for length_penalty in args.length_penalties:
            tag = _decode_tag(beam_size, length_penalty)
            hypotheses: dict[str, str] = {}
            references: dict[str, list[str]] = {}
            prediction_rows: list[dict[str, str]] = []

            desc = f"{args.split}-{tag}"
            for batch in tqdm(loader, desc=desc):
                images = batch["images"].to(device, non_blocking=True)
                generated = model.generate(
                    images,
                    bos_id=vocab.bos_id,
                    eos_id=vocab.eos_id,
                    max_len=args.max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
                for image_name, token_ids, refs in zip(
                    batch["image_names"],
                    generated,
                    batch["references"],
                ):
                    caption = vocab.decode(token_ids)
                    hypotheses[image_name] = caption
                    references[image_name] = refs
                    row = {"image": image_name, "hypothesis": caption}
                    for idx, ref in enumerate(refs, start=1):
                        row[f"ref{idx}"] = ref
                    prediction_rows.append(row)

            if args.save_predictions:
                _write_predictions(output_dir / f"predictions_{args.split}_{tag}.csv", prediction_rows)

            metrics = _compute_metrics(hypotheses, references, args.metrics_backend)
            metrics_path = output_dir / f"metrics_{args.split}_{tag}_{args.metrics_backend}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            row: dict[str, object] = {
                "beam_size": beam_size,
                "length_penalty": length_penalty,
                **metrics,
            }
            summary_rows.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

    summary_path = output_dir / f"decode_search_{args.split}_{args.metrics_backend}.csv"
    metric_names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["beam_size", "length_penalty", *metric_names])
        writer.writeheader()
        writer.writerows(summary_rows)

    best = max(summary_rows, key=lambda row: float(row.get("CIDEr", 0.0)))
    print(f"搜索完成：{summary_path}")
    print(f"按 CIDEr 选择的最优参数：{best}")


if __name__ == "__main__":
    main()
