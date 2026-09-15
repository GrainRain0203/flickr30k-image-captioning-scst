from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


class CSVLogger:
    """轻量 CSV 日志，方便后续画 loss/指标曲线。"""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})

