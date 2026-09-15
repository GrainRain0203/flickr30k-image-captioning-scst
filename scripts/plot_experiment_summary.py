from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_RUNS = [
    "m1_resnet50_attlstm",
    "m2_resnet101_attlstm",
    "m3_resnet101_transformer",
    "m4_convnext_tiny_transformer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 M1-M4 指标并绘制对比图")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    metric_names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    rows: list[dict[str, object]] = []

    for run in args.runs:
        metric_candidates = [
            outputs_dir / run / "metrics_test_beam7_lp0p8_official.json",
            outputs_dir / run / "metrics_test_beam7_lp0p8.json",
            outputs_dir / run / "metrics_test_beam5_lp0p7_official.json",
            outputs_dir / run / "metrics_test_beam5_lp0p7.json",
            outputs_dir / run / "metrics_test_beam5.json",
        ]
        metric_path = next((path for path in metric_candidates if path.exists()), metric_candidates[0])
        log_path = outputs_dir / run / "train_log.csv"
        if not metric_path.exists():
            print(f"跳过缺少指标文件的实验：{run}")
            continue
        with metric_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        best_epoch = ""
        best_val_loss = ""
        if log_path.exists():
            with log_path.open("r", encoding="utf-8-sig", newline="") as f:
                log_rows = list(csv.DictReader(f))
            if log_rows:
                best = min(log_rows, key=lambda r: float(r["val_loss"]))
                best_epoch = best["epoch"]
                best_val_loss = best["val_loss"]

        row: dict[str, object] = {
            "model": run,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
        row.update({name: metrics[name] for name in metric_names})
        rows.append(row)

    if not rows:
        raise RuntimeError("没有找到可汇总的实验指标")

    summary_path = outputs_dir / "metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "best_epoch", "best_val_loss", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["model"]).split("_")[0].upper() for row in rows]
    x = range(len(rows))

    plt.figure(figsize=(10, 5.5), dpi=160)
    for metric in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
        plt.plot(x, [float(row[metric]) for row in rows], marker="o", linewidth=2, label=metric)
    plt.xticks(list(x), labels)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Caption Metrics Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure_path = outputs_dir / "metrics_comparison.png"
    plt.savefig(figure_path)

    print(f"保存指标总表：{summary_path}")
    print(f"保存指标对比图：{figure_path}")


if __name__ == "__main__":
    main()
