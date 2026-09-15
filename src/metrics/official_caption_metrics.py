from __future__ import annotations

import shutil
from typing import Any


References = dict[str, list[str]]
Hypotheses = dict[str, str]


def _require_pycocoevalcap() -> tuple[Any, Any, Any, Any]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "缺少官方 caption 评估依赖 pycocoevalcap。请先安装：\n"
            "  pip install pycocoevalcap\n"
            "METEOR 还需要服务器安装 Java 运行环境。"
        ) from exc
    return Bleu, Meteor, Rouge, Cider


def validate_official_metrics_environment() -> None:
    _require_pycocoevalcap()
    if shutil.which("java") is None:
        raise RuntimeError(
            "当前评估进程找不到 java。请先执行：\n"
            "  export JAVA_HOME=/data/wyr/apps/java17\n"
            "  export PATH=\"$JAVA_HOME/bin:$PATH\"\n"
            "  java -version\n"
            "再重新运行 evaluate.py。"
        )


def _to_coco_format(
    hypotheses: Hypotheses,
    references: References,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    common_images = sorted(set(hypotheses) & set(references))
    if not common_images:
        raise ValueError("hypotheses 与 references 没有可对齐的图片")

    # pycocoevalcap 的 scorer 直接接收 image_id -> list[str]。
    # COCOEvalCap 外层 API 才会使用 {"caption": "..."} 这种 annotation 字典。
    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for idx, image_name in enumerate(common_images):
        gts[idx] = references[image_name]
        res[idx] = [hypotheses[image_name]]
    return gts, res


def compute_official_caption_metrics(
    hypotheses: Hypotheses,
    references: References,
) -> dict[str, float]:
    """使用 pycocoevalcap 计算 BLEU、METEOR、ROUGE-L、CIDEr。

    这是图像描述论文中常用 COCO caption evaluation 工具链的 Python 封装，
    比项目内轻量实现更适合正式报告和论文指标对比。
    """

    Bleu, Meteor, Rouge, Cider = _require_pycocoevalcap()
    gts, res = _to_coco_format(hypotheses, references)

    metrics: dict[str, float] = {}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "官方 METEOR 评估需要 Java。请在服务器安装 Java 后重试，"
                "例如 Ubuntu 上执行：sudo apt install default-jre"
            ) from exc
        finally:
            # Meteor 会启动一个 Java 子进程，评估后需要关闭。
            if hasattr(scorer, "close"):
                scorer.close()

        if isinstance(method, list):
            for name, value in zip(method, score):
                metrics[name] = float(value)
        else:
            metrics[method] = float(score)

    return metrics
