"""Pixel-shuffle + MLP connector (the "bridge" the placeholder did as one Linear).

Two jobs:
  1. **Compress** visual tokens. Folding each ``s x s`` block of patches into the
     channel dimension (``torch.nn.functional.pixel_unshuffle``) cuts the token
     count by ``s^2`` while growing the feature dim by ``s^2``. Keeps full-page
     token counts tractable for a small decoder.
  2. **Project** the compressed features into the LLM embedding dim.

Operates on a spatial map ``[B, C, H, W]`` so it handles **rectangular** grids
(native aspect-preserving mode) as well as square tiles. The LLM hidden size is
passed in from the loaded model config, never hardcoded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, scale: int = 2,
                 depth: int = 2, activation: str = "gelu"):
        super().__init__()
        self.scale = scale
        in_dim = vision_dim * scale * scale
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, llm_dim)]
        for _ in range(max(0, depth - 1)):
            layers += [act(), nn.Linear(llm_dim, llm_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W] -> [B, (H/s)*(W/s), llm_dim]``."""
        b, c, h, w = feature_map.shape
        if h % self.scale or w % self.scale:
            raise ValueError(f"feature grid {h}x{w} not divisible by pixel-shuffle scale {self.scale}")
        x = F.pixel_unshuffle(feature_map, self.scale)  # [B, C*s*s, H/s, W/s]
        x = x.flatten(2).transpose(1, 2)                # [B, (H/s)*(W/s), C*s*s]
        return self.net(x)
