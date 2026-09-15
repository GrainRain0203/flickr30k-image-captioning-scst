from __future__ import annotations

import torch
from torch import nn


class ImageFeatureAdapter(nn.Module):
    """将 CNN 特征图转换为 decoder 可使用的视觉 token 序列。"""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        dropout: float = 0.1,
        max_grid_size: int = 32,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_position = use_position
        self.max_grid_size = max_grid_size
        if use_position:
            # 行/列位置编码比固定展平位置更贴合二维图像特征图。
            self.row_embed = nn.Embedding(max_grid_size, d_model)
            self.col_embed = nn.Embedding(max_grid_size, d_model)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        tokens = feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        tokens = self.proj(tokens)

        if self.use_position:
            if height > self.max_grid_size or width > self.max_grid_size:
                raise ValueError(
                    f"特征图尺寸 {height}x{width} 超过位置编码上限 {self.max_grid_size}"
                )
            rows = torch.arange(height, device=feature_map.device)
            cols = torch.arange(width, device=feature_map.device)
            row_pos = self.row_embed(rows).unsqueeze(1)
            col_pos = self.col_embed(cols).unsqueeze(0)
            pos = (row_pos + col_pos).reshape(1, height * width, -1)
            tokens = tokens + pos

        return self.dropout(tokens)

