from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import DataConfig
from src.data.splits import create_splits, load_captions, save_splits
from src.data.vocab import build_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 Flickr30k 划分文件和词表")
    parser.add_argument("--data-root", default="Flickr30k")
    parser.add_argument("--image-dir", default="Flickr30k_Images")
    parser.add_argument("--captions-file", default="captions.txt")
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-vocab-size", type=int, default=12000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DataConfig(
        data_root=args.data_root,
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    captions_by_image = load_captions(cfg.captions_path)
    existing_images = {path.name for path in cfg.image_path.glob("*.jpg")}
    captions_by_image = {
        image: captions
        for image, captions in captions_by_image.items()
        if image in existing_images
    }
    if not captions_by_image:
        raise RuntimeError("caption 文件中的图片名与图片目录没有交集")

    splits = create_splits(
        sorted(captions_by_image),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    save_splits(splits, cfg.split_path)

    train_captions = [
        caption
        for image_name in splits["train"]
        for caption in captions_by_image[image_name]
    ]
    vocab = build_vocab(
        train_captions,
        min_freq=cfg.min_freq,
        max_vocab_size=cfg.max_vocab_size,
    )
    vocab.save(cfg.vocab_path)

    print(f"图片数：{len(captions_by_image)}")
    print(f"train/val/test：{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"训练 caption 数：{len(train_captions)}")
    print(f"词表大小：{len(vocab)}")
    print(f"划分文件：{cfg.split_path}")
    print(f"词表文件：{cfg.vocab_path}")


if __name__ == "__main__":
    main()
