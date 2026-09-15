from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，保证数据划分和训练过程尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

