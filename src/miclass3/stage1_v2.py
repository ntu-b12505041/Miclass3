from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .models import ResidualSEBlock, SEBlock


class MultiScaleSEStem(nn.Module):
    """125 Hz multi-scale stem for narrow QRS and slower ST/T morphology."""

    def __init__(
        self,
        in_channels: int = 64,
        width: int = 64,
        kernels: Sequence[int] = (11, 25, 49),
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if len(kernels) != 3:
            raise ValueError("stage1-v2 expects exactly three temporal kernels")
        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("multi-scale temporal kernels must be positive odd integers")
        if width % 4 != 0:
            raise ValueError("width must be divisible by 4 for the four inception branches")

        branch_channels = width // 4
        reduced = min(int(bottleneck), width)
        self.reduce = nn.Sequential(
            nn.Conv1d(in_channels, reduced, 1, bias=False),
            nn.BatchNorm1d(reduced),
            nn.SiLU(),
        )
        self.temporal_paths = nn.ModuleList(
            [
                nn.Conv1d(
                    reduced,
                    branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernels
            ]
        )
        self.pool_path = nn.Sequential(
            nn.MaxPool1d(7, stride=1, padding=3),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(width)
        self.act = nn.SiLU()
        self.se = SEBlock(width)
        self.kernels = kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branches = [path(reduced) for path in self.temporal_paths]
        branches.append(self.pool_path(x))
        return self.se(self.act(self.norm(torch.cat(branches, dim=1))))


class Stage1V2MultiScaleSE(nn.Module):
    """MI router with NSTEMI-aware three-class auxiliary supervision.

    ``class_logits`` is the primary MI-vs-non-MI output used for routing.
    ``subtype_logits`` is training-only auxiliary supervision over the three
    ECG proxy classes. It forces the shared waveform representation to retain
    information useful for NSTEMI-proxy even when classic STEMI morphology is
    absent.
    """

    def __init__(
        self,
        in_channels: int = 12,
        width: int = 64,
        kernels: Sequence[int] = (11, 25, 49),
        block_dropout: float = 0.10,
        head_dropout: float = 0.20,
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        # Reduce 500 Hz to 125 Hz before the expensive parallel convolutions.
        # This preserves about 8 ms temporal resolution while making the three
        # kernels correspond to roughly 88, 200 and 392 ms receptive fields.
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.stem = MultiScaleSEStem(
            in_channels=width,
            width=width,
            kernels=kernels,
            bottleneck=bottleneck,
        )
        self.encoder = nn.Sequential(
            ResidualSEBlock(width, width, dropout=block_dropout),
            ResidualSEBlock(width, width * 2, 2, dropout=block_dropout),
            ResidualSEBlock(width * 2, width * 2, dropout=block_dropout),
            ResidualSEBlock(width * 2, width * 4, 2, dropout=block_dropout),
            ResidualSEBlock(width * 4, width * 4, dropout=block_dropout),
            ResidualSEBlock(width * 4, width * 8, 2, dropout=block_dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = width * 8
        self.dropout = nn.Dropout(head_dropout)
        self.mi_head = nn.Linear(self.embedding_dim, 2)
        self.subtype_head = nn.Linear(self.embedding_dim, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(self.stem(self.input_projection(x)))).squeeze(-1)

    def forward(self, x: torch.Tensor, features=None) -> dict[str, torch.Tensor]:
        embedding = self.dropout(self.encode(x))
        return {
            "class_logits": self.mi_head(embedding),
            "subtype_logits": self.subtype_head(embedding),
        }
