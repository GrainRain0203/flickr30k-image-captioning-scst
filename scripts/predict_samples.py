from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, ModelConfig
from src.data.dataset import FlickrEvalDataset, build_transforms, eval_collate_fn
from src.data.splits import load_captions, load_splits
from src.data.vocab import Vocabulary
from src.models import build_caption_model
from src.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出测试图片描述样例")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=7)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--vocab-file", default="", help="可选：覆盖配置中的词表文件路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _config_from_checkpoint(checkpoint: dict) -> tuple[DataConfig, ModelConfig]:
    extra = checkpoint.get("extra", {})
    cfg = extra.get("config", {})
    if cfg:
        return DataConfig(**cfg["data"]), ModelConfig(**cfg["model"])
    if extra.get("scst_config"):
        scst_cfg = extra["scst_config"]
        return DataConfig(**scst_cfg["data"]), ModelConfig(**scst_cfg["model"])
    if extra.get("base_project_config"):
        cfg = extra["base_project_config"]
        return DataConfig(**cfg["data"]), ModelConfig(**cfg["model"])
    return DataConfig(), ModelConfig()


def _decode_tag(beam_size: int, length_penalty: float) -> str:
    lp = str(length_penalty).replace(".", "p")
    return f"beam{beam_size}_lp{lp}"


def main() -> None:
    args = parse_args()
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
    vocab_path = Path(args.vocab_file) if args.vocab_file else data_cfg.vocab_path
    vocab = Vocabulary.load(vocab_path)

    rng = random.Random(args.seed)
    image_names = list(splits[args.split])
    rng.shuffle(image_names)
    image_names = image_names[: args.num_samples]

    dataset = FlickrEvalDataset(
        image_dir=data_cfg.image_path,
        captions_by_image=captions_by_image,
        image_names=image_names,
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    model = build_caption_model(model_cfg, vocab_size=len(vocab), max_len=data_cfg.max_len).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, str]] = []
    for batch in loader:
        image_name = batch["image_names"][0]
        images = batch["images"].to(device)
        token_ids = model.generate(
            images,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )[0]
        hypothesis = vocab.decode(token_ids)
        refs = batch["references"][0]
        row = {
            "image": image_name,
            "image_path": str((data_cfg.image_path / image_name).resolve()),
            "hypothesis": hypothesis,
        }
        for idx, ref in enumerate(refs, start=1):
            row[f"ref{idx}"] = ref
        rows.append(row)

    output_dir = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _decode_tag(args.beam_size, args.length_penalty)
    csv_path = output_dir / f"samples_{args.split}_{tag}.csv"
    md_path = output_dir / f"samples_{args.split}_{tag}.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["image", "image_path", "hypothesis", "ref1", "ref2", "ref3", "ref4", "ref5"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 图像描述样例\n\n")
        for idx, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"]).as_posix()
            f.write(f"## 样例 {idx}: {row['image']}\n\n")
            f.write(f"![{row['image']}]({image_path})\n\n")
            f.write(f"**模型生成：** {row['hypothesis']}\n\n")
            f.write("**参考描述：**\n\n")
            for ref_idx in range(1, 6):
                f.write(f"- {row[f'ref{ref_idx}']}\n")
            f.write("\n")

    print(f"样例 CSV：{csv_path}")
    print(f"样例 Markdown：{md_path}")


if __name__ == "__main__":
    main()
