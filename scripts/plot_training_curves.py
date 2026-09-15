from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制训练/验证 loss 曲线")
    parser.add_argument("--log", required=True, help="train_log.csv 路径")
    parser.add_argument("--output", default="", help="输出图片路径，默认保存到日志同目录")
    return parser.parse_args()


def select_last_run(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """同名 run 重复训练时，日志会追加多段；画图默认使用最后一段。"""

    if not rows:
        raise ValueError("训练日志为空")

    segments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    last_epoch = 0
    for row in rows:
        epoch = int(row["epoch"])
        if current and epoch <= last_epoch:
            segments.append(current)
            current = []
        current.append(row)
        last_epoch = epoch
    if current:
        segments.append(current)

    if len(segments) > 1:
        print(f"检测到 {len(segments)} 段训练日志，默认绘制最后一段。")
    return segments[-1]


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)

    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = select_last_run(list(csv.DictReader(f)))

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]
    stages = [row["stage"] for row in rows]

    output_path = Path(args.output) if args.output else log_path.with_name("loss_curve.png")

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, val_losses, marker="s", label="val loss")

    seen_stage = set()
    for epoch, stage in zip(epochs, stages):
        if stage not in seen_stage:
            plt.axvline(epoch, linestyle="--", linewidth=1, alpha=0.4)
            plt.text(epoch, max(val_losses), stage, rotation=90, va="top", fontsize=8)
            seen_stage.add(stage)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"保存曲线：{output_path}")


if __name__ == "__main__":
    main()
