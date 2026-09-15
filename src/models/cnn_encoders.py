from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CNNEncoder(nn.Module):
    """通用视觉编码器，输出空间特征图 B x C x H x W。"""

    def __init__(
        self,
        arch: str,
        pretrained: bool = True,
        weights_path: str = "",
    ) -> None:
        super().__init__()
        self.arch = arch.lower()
        self.backbone, self.out_channels = self._build_backbone(
            self.arch,
            pretrained,
            weights_path,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_stage(self) -> None:
        """只解冻最后一个视觉 stage，用于 stage2 小学习率微调。"""

        self.freeze()
        if self.arch.startswith("resnet"):
            last_stage = self.backbone[-1]
        elif self.arch.startswith("convnext"):
            last_stage = self.backbone[-1]
        else:
            raise ValueError(f"未知视觉编码器：{self.arch}")
        for param in last_stage.parameters():
            param.requires_grad = True

    @classmethod
    def _build_backbone(
        cls,
        arch: str,
        pretrained: bool,
        weights_path: str,
    ) -> tuple[nn.Module, int]:
        if arch == "resnet50":
            weights = None if weights_path else (models.ResNet50_Weights.DEFAULT if pretrained else None)
            model = models.resnet50(weights=weights)
            cls._load_local_weights(model, weights_path)
            return nn.Sequential(*list(model.children())[:-2]), 2048

        if arch == "resnet101":
            weights = None if weights_path else (models.ResNet101_Weights.DEFAULT if pretrained else None)
            model = models.resnet101(weights=weights)
            cls._load_local_weights(model, weights_path)
            return nn.Sequential(*list(model.children())[:-2]), 2048

        if arch == "convnext_tiny":
            weights = None if weights_path else (models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            model = models.convnext_tiny(weights=weights)
            cls._load_local_weights(model, weights_path)
            return model.features, 768

        if arch == "convnext_small":
            weights = None if weights_path else (models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
            model = models.convnext_small(weights=weights)
            cls._load_local_weights(model, weights_path)
            return model.features, 768

        if arch == "convnext_base":
            weights = None if weights_path else (models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
            model = models.convnext_base(weights=weights)
            cls._load_local_weights(model, weights_path)
            return model.features, 1024

        raise ValueError(f"不支持的视觉编码器：{arch}")

    @staticmethod
    def _load_local_weights(model: nn.Module, weights_path: str) -> None:
        if not weights_path:
            return
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            raise ValueError(f"本地权重文件格式不正确：{weights_path}")

        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"本地权重 {weights_path} 与当前视觉骨干结构不匹配"
            ) from exc
