from __future__ import annotations

import torch
from torch import nn

from .models import SEResNet


class Stage1V3MorphologyFusion(nn.Module):
    """Full-resolution MI router with safe ECG morphology fusion."""

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 0,
        width: int = 64,
        block_dropout: float = 0.15,
        fusion_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("Stage1-v3 requires non-empty morphology features")
        self.ecg = SEResNet(
            in_channels=in_channels,
            classes=2,
            width=width,
            dropout=block_dropout,
        )
        self.feature_dim = int(feature_dim)
        self.raw_projection = nn.Sequential(
            nn.Linear(self.ecg.embedding_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(self.ecg.embedding_dim + 128, 128),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(fusion_dropout),
        )
        self.mi_head = nn.Linear(256, 2)
        self.subtype_head = nn.Linear(256, 3)

    def encode(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        raw_z = self.ecg.encode(x)
        raw_repr = self.raw_projection(raw_z)
        morphology_z = self.feature_encoder(features)
        gate = 0.5 + self.feature_gate(torch.cat([raw_z, morphology_z], dim=1))
        return self.fusion(torch.cat([raw_repr, morphology_z * gate], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features is None:
            raise ValueError("Stage1-v3 requires morphology features")
        z = self.encode(x, features)
        return {
            "class_logits": self.mi_head(z),
            "subtype_logits": self.subtype_head(z),
        }
