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

from src.metrics import (
    compute_caption_metrics,
    compute_official_caption_metrics,
    validate_official_metrics_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 predictions CSV 重新计算图像描述指标")
    parser.add_argument("--predictions", required=True, help="predictions_*.csv 路径")
    parser.add_argument(
        "--metrics-backend",
        default="official",
        choices=["official", "lightweight"],
    )
    parser.add_argument("--output", default="", help="指标 JSON 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_backend == "official":
        validate_official_metrics_environment()

    predictions_path = Path(args.predictions)
    hypotheses: dict[str, str] = {}
    references: dict[str, list[str]] = {}
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image"]
            hypotheses[image_name] = row["hypothesis"]
            references[image_name] = [
                row[f"ref{idx}"]
                for idx in range(1, 6)
                if row.get(f"ref{idx}")
            ]

    if args.metrics_backend == "official":
        metrics = compute_official_caption_metrics(hypotheses, references)
    else:
        metrics = compute_caption_metrics(hypotheses, references)

    output_path = Path(args.output) if args.output else predictions_path.with_name(
        predictions_path.stem.replace("predictions", "metrics") + f"_{args.metrics_backend}.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"指标文件：{output_path}")


if __name__ == "__main__":
    main()
