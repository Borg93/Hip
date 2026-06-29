"""Pixel-shuffle + MLP connector (the "bridge" the placeholder did as one Linear).

Two jobs:
  1. **Compress** visual tokens. A 448 tile at patch-14 is a 32x32=1024-token grid;
     pixel-shuffle by 2 folds each 2x2 block into the channel dim -> 16x16=256
     tokens with 4x the channels. This keeps full-page token counts tractable for a
     small decoder.
  2. **Project** from the (shuffled) vision dim into the LLM embedding dim.

The LLM hidden size is passed in from the loaded model config, never hardcoded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pixel_shuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """``[B, H*W, C] -> [B, (H/scale)*(W/scale), C*scale*scale]`` (square grid)."""
    b, n, c = x.shape
    hw = int(round(n ** 0.5))
    if hw * hw != n:
        raise ValueError(f"expected a square patch grid, got {n} tokens")
    if hw % scale != 0:
        raise ValueError(f"grid {hw} not divisible by pixel-shuffle scale {scale}")
    x = x.view(b, hw, hw, c)
    x = x.view(b, hw, hw // scale, c * scale)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(b, hw // scale, hw // scale, c * scale * scale)
    x = x.view(b, (hw // scale) * (hw // scale), c * scale * scale)
    return x


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

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """``[n_tiles, num_patches, vision_dim] -> [n_tiles, num_tokens, llm_dim]``."""
        x = pixel_shuffle(patch_tokens, self.scale)
        return self.net(x)
