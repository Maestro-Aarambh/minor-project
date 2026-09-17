"""
Multimodal self-attention encoder: byte-patch tokens + static-feature token.

Why it exists:
The core hypothesis of the project — self-attention relates distant positions
directly, so it should degrade less on evasive samples than CNNs (local
windows) or LSTMs (sequential bottleneck). This module also fuses the EMBER
static vector (what LightGBM sees) with the byte sequence (what CNN/LSTM see),
so the transformer is not handicapped by a weaker input representation.

Token layout fed to the encoder:
    [CLS] [STATIC] [P1] [P2] ... [P256]
where each Pi is a patch of `patch_size` consecutive byte embeddings and
STATIC is the projected (standardized) EMBER static feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer.positional_encoding import SinusoidalPositionalEncoding


class BytePatchEmbedding(nn.Module):
    """
    Bytes (B, seq_len) -> patch tokens (B, seq_len // patch_size, d_model).

    Patching keeps the token count small (4096 bytes / 16 = 256 tokens), which
    keeps self-attention memory quadratic in 256 — trainable on a 4 GB GPU.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        byte_embed_dim: int = 32,
        patch_size: int = 16,
        d_model: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Embedding(vocab_size, byte_embed_dim)
        self.proj = nn.Linear(patch_size * byte_embed_dim, d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, seq_len) int64
        emb = self.embedding(x_seq)  # (B, seq_len, byte_embed_dim)
        bsz, seq_len, edim = emb.shape
        n_patches = seq_len // self.patch_size
        emb = emb[:, : n_patches * self.patch_size, :]
        emb = emb.reshape(bsz, n_patches, self.patch_size * edim)
        return self.proj(emb)  # (B, n_patches, d_model)


class MultimodalTransformer(nn.Module):
    """CLS + optional static/byte tokens -> transformer encoder -> logit."""

    def __init__(
        self,
        static_dim: int,
        seq_len: int = 4096,
        patch_size: int = 16,
        byte_embed_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.2,
        modality: str = "full",
    ):
        super().__init__()
        self.static_dim = static_dim
        self.seq_len = seq_len
        self.modality = modality.lower()
        if self.modality not in ("full", "static_only", "bytes_only"):
            raise ValueError(f"modality must be full|static_only|bytes_only, got {modality!r}")

        n_patches = seq_len // patch_size
        self.byte_embed = BytePatchEmbedding(
            byte_embed_dim=byte_embed_dim, patch_size=patch_size, d_model=d_model
        )
        self.static_norm = nn.LayerNorm(static_dim)
        self.static_proj = nn.Linear(static_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        n_extra = 0
        if self.modality in ("full", "static_only"):
            n_extra += 1  # STATIC token
        if self.modality in ("full", "bytes_only"):
            n_extra += n_patches
        n_tokens = 1 + n_extra  # CLS + extras

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=n_tokens, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_static: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        # x_static: (B, static_dim) float32 (already standardized by the classifier)
        # x_seq:    (B, seq_len) int64
        cls = self.cls_token.expand(x_seq.size(0), -1, -1)
        parts: list[torch.Tensor] = [cls]

        if self.modality in ("full", "static_only"):
            static_token = self.static_proj(self.static_norm(x_static)).unsqueeze(1)
            parts.append(static_token)
        if self.modality in ("full", "bytes_only"):
            parts.append(self.byte_embed(x_seq))

        tokens = torch.cat(parts, dim=1)
        tokens = self.pos_encoding(tokens)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])  # logit from CLS token: (B, 1)
