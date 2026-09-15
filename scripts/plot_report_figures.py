from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


METRIC_VARIANTS = [
    ("m1_resnet50_attlstm", "M1"),
    ("m2_resnet101_attlstm", "M2"),
    ("m3_resnet101_transformer", "M3"),
    ("m4_convnext_tiny_transformer", "M4"),
    ("m4_convnext_tiny_transformer_lrB", "M4-lrB"),
    ("m5_convnext_base_lra", "M5"),
    ("m5_convnext_base_lra_scst_safe", "M5+SCST"),
]


LOSS_RUNS = [
    ("M1", "m1_clean_train_log.csv", "m1_resnet50_attlstm"),
    ("M2", "m2_resnet101_attlstm/train_log.csv", "m2_resnet101_attlstm"),
    ("M3", "m3_resnet101_transformer/train_log.csv", "m3_resnet101_transformer"),
    ("M4", "m4_convnext_tiny_transformer/train_log.csv", "m4_convnext_tiny_transformer"),
    ("M4-lrB", "m4_convnext_tiny_transformer_lrB/train_log.csv", "m4_convnext_tiny_transformer_lrB"),
    ("M5", "m5_convnext_base_resize384_lra/train_log.csv", "m5_convnext_base_lra"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the three report figures from experiment CSV/JSON files."
    )
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--keep-duplicated-log-segments",
        action="store_true",
        help=(
            "Plot every row in train_log.csv. By default, duplicated/restarted logs "
            "are reduced to the segment matching metrics_summary.csv."
        ),
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_epoch_segments(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    if not rows:
        raise ValueError("empty train log")

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
    return segments


def select_segment_matching_summary(
    rows: list[dict[str, str]], best_epoch: int | None, best_val_loss: float | None
) -> list[dict[str, str]]:
    segments = split_epoch_segments(rows)
    if best_epoch is None or best_val_loss is None:
        return segments[-1]

    for segment in segments:
        for row in segment:
            same_epoch = int(row["epoch"]) == best_epoch
            same_loss = abs(float(row["val_loss"]) - best_val_loss) < 1e-6
            if same_epoch and same_loss:
                return segment
    return segments[-1]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )


def plot_metric_comparison(outputs_dir: Path) -> Path:
    summary_path = outputs_dir / "metrics_summary.csv"
    rows = {row["model"]: row for row in read_csv_dicts(summary_path)}

    labels: list[str] = []
    values: dict[str, list[float]] = {
        "BLEU-4": [],
        "METEOR": [],
        "ROUGE-L": [],
        "CIDEr": [],
    }
    for model_name, label in METRIC_VARIANTS:
        if model_name not in rows:
            raise FileNotFoundError(f"{summary_path} is missing row: {model_name}")
        labels.append(label)
        row = rows[model_name]
        for metric in values:
            values[metric].append(float(row[metric]))

    x = list(range(len(labels)))
    plt.figure(figsize=(8.8, 4.8), dpi=180)
    for metric, metric_values in values.items():
        plt.plot(x, metric_values, marker="o", linewidth=2.0, label=metric)
    plt.xticks(x, labels)
    plt.xlabel("Model variant")
    plt.ylabel("Score")
    plt.title("Official Test Metrics")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()

    output_path = outputs_dir / "report_metrics_comparison.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_validation_losses(outputs_dir: Path, keep_duplicated_log_segments: bool) -> Path:
    summary_rows = {
        row["model"]: row for row in read_csv_dicts(outputs_dir / "metrics_summary.csv")
    }
    plt.figure(figsize=(8.5, 4.594), dpi=180)

    for label, log_rel_path, summary_model in LOSS_RUNS:
        log_path = outputs_dir / log_rel_path
        rows = read_csv_dicts(log_path)
        if not keep_duplicated_log_segments:
            summary = summary_rows.get(summary_model, {})
            best_epoch = int(summary["best_epoch"]) if summary.get("best_epoch") else None
            best_val_loss = (
                float(summary["best_val_loss"]) if summary.get("best_val_loss") else None
            )
            rows = select_segment_matching_summary(rows, best_epoch, best_val_loss)
        epochs = [int(row["epoch"]) for row in rows]
        val_losses = [float(row["val_loss"]) for row in rows]
        plt.plot(epochs, val_losses, linewidth=2.0, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation CE loss")
    plt.title("Validation Loss Curves")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()

    output_path = outputs_dir / "report_val_loss_curves.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def scst_epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cannot parse SCST epoch from {path.name}")
    return int(match.group(1))


def plot_scst_validation_metrics(outputs_dir: Path) -> Path:
    scst_dir = outputs_dir / "m5_convnext_base_resize384_lra_scst_cider_safe"
    metric_paths = sorted(scst_dir.glob("metrics_val_epoch*_official.json"), key=scst_epoch)
    if not metric_paths:
        raise FileNotFoundError(f"no SCST validation metrics found in {scst_dir}")

    epochs: list[int] = []
    cider: list[float] = []
    bleu4: list[float] = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        epochs.append(scst_epoch(path))
        cider.append(float(metrics["CIDEr"]))
        bleu4.append(float(metrics["BLEU-4"]))

    plt.figure(figsize=(6.8, 4.2), dpi=180)
    plt.plot(epochs, cider, marker="o", linewidth=2.0, label="CIDEr")
    plt.plot(epochs, bleu4, marker="s", linewidth=2.0, label="BLEU-4")
    plt.xticks(epochs)
    plt.xlabel("SCST epoch")
    plt.ylabel("Validation score")
    plt.title("SCST Validation Metrics")
    plt.grid(alpha=0.25)
    plt.legend(loc="center right")
    plt.tight_layout()

    output_path = outputs_dir / "report_scst_val_curve.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    configure_matplotlib()

    figure_paths = [
        plot_metric_comparison(outputs_dir),
        plot_validation_losses(outputs_dir, args.keep_duplicated_log_segments),
        plot_scst_validation_metrics(outputs_dir),
    ]
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
