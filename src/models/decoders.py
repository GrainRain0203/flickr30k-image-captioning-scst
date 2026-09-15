from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    attention: torch.Tensor | None = None


class LSTMDecoder(nn.Module):
    """M0 使用的普通 LSTM decoder，只使用全局图像特征初始化隐状态。"""

    decoder_type = "lstm"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.init_h = nn.Linear(d_model, hidden_dim)
        self.init_c = nn.Linear(d_model, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = visual_tokens.mean(dim=1)
        h = torch.tanh(self.init_h(pooled)).unsqueeze(0)
        c = torch.tanh(self.init_c(pooled)).unsqueeze(0)
        return h, c

    def forward(
        self,
        visual_tokens: torch.Tensor,
        captions: torch.Tensor,
        pad_id: int | None = None,
    ) -> DecoderOutput:
        del pad_id
        state = self.init_state(visual_tokens)
        embedded = self.dropout(self.embedding(captions))
        output, _ = self.lstm(embedded, state)
        logits = self.classifier(self.dropout(output))
        return DecoderOutput(logits=logits)

    def step(
        self,
        prev_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        del visual_tokens
        embedded = self.embedding(prev_tokens).unsqueeze(1)
        output, new_state = self.lstm(embedded, state)
        logits = self.classifier(output.squeeze(1))
        return logits, new_state, None


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention，用于 M1/M2。"""

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected_features = self.feature_proj(features)
        projected_hidden = self.hidden_proj(hidden).unsqueeze(1)
        energy = torch.tanh(projected_features + projected_hidden)
        scores = self.score(energy).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        context = torch.sum(features * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class AttentionLSTMDecoder(nn.Module):
    """M1/M2 使用的 Attention LSTM decoder。"""

    decoder_type = "attention_lstm"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AdditiveAttention(d_model, hidden_dim, attention_dim=hidden_dim)
        self.init_h = nn.Linear(d_model, hidden_dim)
        self.init_c = nn.Linear(d_model, hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, d_model)
        self.lstm_cell = nn.LSTMCell(embed_dim + d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = visual_tokens.mean(dim=1)
        h = torch.tanh(self.init_h(pooled))
        c = torch.tanh(self.init_c(pooled))
        return h, c

    def forward(
        self,
        visual_tokens: torch.Tensor,
        captions: torch.Tensor,
        pad_id: int | None = None,
    ) -> DecoderOutput:
        del pad_id
        batch_size, seq_len = captions.shape
        h, c = self.init_state(visual_tokens)
        logits_steps: list[torch.Tensor] = []
        attention_steps: list[torch.Tensor] = []

        embeddings = self.dropout(self.embedding(captions))
        for t in range(seq_len):
            context, alpha = self.attention(visual_tokens, h)
            gate = torch.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([embeddings[:, t], context], dim=-1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits_steps.append(self.classifier(self.dropout(h)))
            attention_steps.append(alpha)

        logits = torch.stack(logits_steps, dim=1)
        attention = torch.stack(attention_steps, dim=1).reshape(batch_size, seq_len, -1)
        return DecoderOutput(logits=logits, attention=attention)

    def step(
        self,
        prev_tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        visual_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        h, c = state
        embedded = self.embedding(prev_tokens)
        context, alpha = self.attention(visual_tokens, h)
        gate = torch.sigmoid(self.f_beta(h))
        context = gate * context
        h, c = self.lstm_cell(torch.cat([embedded, context], dim=-1), (h, c))
        logits = self.classifier(self.dropout(h))
        return logits, (h, c), alpha


class TransformerCaptionDecoder(nn.Module):
    """M3/M4 使用的 Transformer decoder。"""

    decoder_type = "transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len + 8, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        captions: torch.Tensor,
        pad_id: int | None = None,
    ) -> DecoderOutput:
        batch_size, seq_len = captions.shape
        positions = torch.arange(seq_len, device=captions.device).unsqueeze(0).expand(batch_size, -1)
        target = self.embedding(captions) * math.sqrt(self.d_model)
        target = self.dropout(target + self.position_embedding(positions))

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=captions.device),
            diagonal=1,
        )
        key_padding_mask = captions.eq(pad_id) if pad_id is not None else None
        decoded = self.decoder(
            tgt=target,
            memory=visual_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )
        logits = self.classifier(decoded)
        return DecoderOutput(logits=logits)

