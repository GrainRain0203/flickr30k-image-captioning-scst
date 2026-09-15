from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class DataConfig:
    """数据相关配置。路径默认贴合本项目目录。"""

    data_root: str = "Flickr30k"
    image_dir: str = "Flickr30k_Images"
    captions_file: str = "captions.txt"
    split_file: str = "splits.json"
    vocab_file: str = "vocab.json"
    min_freq: int = 3
    max_vocab_size: int = 12000
    max_len: int = 32
    image_size: int = 384
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 4

    @property
    def root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def image_path(self) -> Path:
        return self.root_path / self.image_dir

    @property
    def captions_path(self) -> Path:
        return self.root_path / self.captions_file

    @property
    def split_path(self) -> Path:
        return self.root_path / self.split_file

    @property
    def vocab_path(self) -> Path:
        return self.root_path / self.vocab_file


@dataclass
class ModelConfig:
    """模型超参数。M0-M4 的网络类型由 model_name 决定。"""

    model_name: str = "m4"
    d_model: int = 512
    hidden_dim: int = 512
    embed_dim: int = 512
    decoder_layers: int = 4
    attention_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pretrained_encoder: bool = True
    freeze_encoder: bool = True
    encoder_weights_path: str = ""


@dataclass
class TrainConfig:
    """训练配置。stage1 冻结视觉编码器，stage2 解冻最后 stage 微调。"""

    output_dir: str = "outputs"
    batch_size: int = 32
    epochs_stage1: int = 12
    epochs_stage2: int = 8
    lr_decoder: float = 3e-4
    lr_encoder: float = 3e-5
    lr_decoder_stage2: float | None = 5e-5
    lr_encoder_stage2: float | None = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True
    log_interval: int = 50
    seed: int = 42


@dataclass
class EvalConfig:
    """评估与生成配置。"""

    split: str = "test"
    batch_size: int = 32
    max_len: int = 32
    beam_size: int = 7
    length_penalty: float = 0.8
    use_beam: bool = True


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig


def save_config(config: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return ProjectConfig(
        data=DataConfig(**obj["data"]),
        model=ModelConfig(**obj["model"]),
        train=TrainConfig(**obj["train"]),
        eval=EvalConfig(**obj["eval"]),
    )
