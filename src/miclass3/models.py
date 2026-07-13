from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1), nn.SiLU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 9, stride=stride, padding=4, bias=False), nn.BatchNorm1d(out_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False), nn.BatchNorm1d(out_channels), SEBlock(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class SEResNet(nn.Module):
    def __init__(self, in_channels: int = 12, classes: int = 3, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, 15, padding=7, bias=False), nn.BatchNorm1d(width), nn.SiLU())
        self.encoder = nn.Sequential(
            ResidualSEBlock(width, width), ResidualSEBlock(width, width * 2, 2), ResidualSEBlock(width * 2, width * 2),
            ResidualSEBlock(width * 2, width * 4, 2), ResidualSEBlock(width * 4, width * 4), ResidualSEBlock(width * 4, width * 8, 2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = width * 8
        self.head = nn.Linear(self.embedding_dim, classes)

    def encode(self, x):
        return self.pool(self.encoder(self.stem(x))).squeeze(-1)

    def forward(self, x, features=None):
        return {"class_logits": self.head(self.encode(x))}


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        bottleneck = min(32, in_channels)
        branch = out_channels // 4
        self.reduce = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.paths = nn.ModuleList([nn.Conv1d(bottleneck, branch, k, padding=k // 2, bias=False) for k in (9, 19, 39)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_channels, branch, 1, bias=False))
        self.norm = nn.BatchNorm1d(branch * 4)
        self.act = nn.SiLU()

    def forward(self, x):
        z = self.reduce(x)
        return self.act(self.norm(torch.cat([*(p(z) for p in self.paths), self.pool(x)], dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_channels: int = 12, classes: int = 3, channels: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(InceptionBlock(in_channels, channels), InceptionBlock(channels, channels), InceptionBlock(channels, channels), InceptionBlock(channels, channels), InceptionBlock(channels, channels), InceptionBlock(channels, channels))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(channels, classes))

    def forward(self, x, features=None):
        return {"class_logits": self.head(self.blocks(x))}


class MorphologyFusion(nn.Module):
    """Primary model with waveform/morphology fusion and auxiliary ECG tasks.

    The LBBB head reads the waveform embedding (before morphology fusion) so
    its optional auxiliary loss encourages the raw ECG encoder to represent
    conduction abnormality, instead of merely copying the LBBB input feature.
    """
    def __init__(self, in_channels: int = 12, feature_dim: int = 0, classes: int = 3):
        super().__init__()
        self.ecg = SEResNet(in_channels, classes=classes)
        self.feature_dim = feature_dim
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.15)) if feature_dim else None
        total = self.ecg.embedding_dim + (64 if feature_dim else 0)
        self.classifier = nn.Sequential(nn.Linear(total, 256), nn.SiLU(), nn.Dropout(0.25), nn.Linear(256, classes))
        self.stemi_head = nn.Linear(total, 1)
        self.lbbb_head = nn.Linear(self.ecg.embedding_dim, 1)

    def forward(self, x, features=None):
        raw_z = self.ecg.encode(x)
        z = raw_z
        if self.feature_encoder is not None:
            if features is None:
                features = torch.zeros((x.shape[0], self.feature_dim), device=x.device)
            z = torch.cat([z, self.feature_encoder(features)], dim=1)
        return {
            "class_logits": self.classifier(z),
            "stemi_logits": self.stemi_head(z).squeeze(1),
            "lbbb_logits": self.lbbb_head(raw_z).squeeze(1),
        }


def make_model(name: str, feature_dim: int = 0, in_channels: int = 12) -> nn.Module:
    if name == "seresnet": return SEResNet(in_channels)
    if name == "inceptiontime": return InceptionTime(in_channels)
    if name == "morphology_fusion": return MorphologyFusion(in_channels, feature_dim)
    raise ValueError(f"Unknown model: {name}")
