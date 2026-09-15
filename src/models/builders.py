from __future__ import annotations

from dataclasses import dataclass

from .caption_model import CaptionModel
from .cnn_encoders import CNNEncoder
from .decoders import AttentionLSTMDecoder, LSTMDecoder, TransformerCaptionDecoder
from .feature_adapter import ImageFeatureAdapter
from src.config import ModelConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder_arch: str
    decoder_type: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "m0": ModelSpec(
        name="m0",
        encoder_arch="resnet50",
        decoder_type="lstm",
        description="参考模型：ResNet50 + LSTM",
    ),
    "m1": ModelSpec(
        name="m1",
        encoder_arch="resnet50",
        decoder_type="attention_lstm",
        description="正式 baseline：ResNet50 + Attention LSTM",
    ),
    "m2": ModelSpec(
        name="m2",
        encoder_arch="resnet101",
        decoder_type="attention_lstm",
        description="改进 CNN：ResNet101 + Attention LSTM",
    ),
    "m3": ModelSpec(
        name="m3",
        encoder_arch="resnet101",
        decoder_type="transformer",
        description="改进 decoder：ResNet101 + Transformer Decoder",
    ),
    "m4": ModelSpec(
        name="m4",
        encoder_arch="convnext_tiny",
        decoder_type="transformer",
        description="最终增强模型：ConvNeXt-Tiny + Transformer Decoder",
    ),
    "m5": ModelSpec(
        name="m5",
        encoder_arch="convnext_base",
        decoder_type="transformer",
        description="ConvNeXt-Base + Transformer Decoder",
    ),
}


def build_caption_model(
    model_config: ModelConfig,
    vocab_size: int,
    max_len: int,
) -> CaptionModel:
    model_name = model_config.model_name.lower()
    if model_name not in MODEL_SPECS:
        valid = ", ".join(MODEL_SPECS)
        raise ValueError(f"未知模型 {model_name}，可选：{valid}")

    spec = MODEL_SPECS[model_name]
    encoder = CNNEncoder(
        arch=spec.encoder_arch,
        pretrained=model_config.pretrained_encoder,
        weights_path=model_config.encoder_weights_path,
    )
    adapter = ImageFeatureAdapter(
        in_channels=encoder.out_channels,
        d_model=model_config.d_model,
        dropout=model_config.dropout,
    )

    if spec.decoder_type == "lstm":
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=model_config.embed_dim,
            d_model=model_config.d_model,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
        )
    elif spec.decoder_type == "attention_lstm":
        decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=model_config.embed_dim,
            d_model=model_config.d_model,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
        )
    elif spec.decoder_type == "transformer":
        decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            d_model=model_config.d_model,
            num_layers=model_config.decoder_layers,
            nhead=model_config.attention_heads,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            max_len=max_len,
        )
    else:
        raise ValueError(f"不支持的 decoder 类型：{spec.decoder_type}")

    model = CaptionModel(encoder=encoder, adapter=adapter, decoder=decoder)
    if model_config.freeze_encoder:
        model.freeze_encoder()
    return model
