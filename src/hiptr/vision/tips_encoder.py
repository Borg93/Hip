"""TIPSv2 vision-encoder wrapper.

Matches the **real** TIPSv2 API. TIPSv2 can be used two ways:

  * **HF AutoModel** (used here): ``AutoModel.from_pretrained("google/tipsv2-l14-dpt",
    trust_remote_code=True)`` -> ``dpt._get_backbone()`` -> ``dpt._backbone.vision_encoder``.
  * **Source / .npz** (the foreground-seg Colab): ``from tips.pytorch import image_encoder;
    vit_large(img_size=..., patch_size=14, ...); load_state_dict(npz)`` with weights from
    ``storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/``. See scripts/fetch_tips.sh.

The encoder is DINOv2-style, so this wrapper returns a **spatial feature map**
``[B, C, H, W]`` (H = input_h // patch, W = input_w // patch), which the connector
pixel-shuffles. Three feature paths:

  * ``intermediate`` (default) — ``get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]``,
    the official dense-task path (used for segmentation; great for HTR).
  * ``standard``     — plain ``vision(x)`` -> ``(cls, _, patch_tokens)``, 3rd element.
  * ``value``        — last-block value-attention surgery (sharper per-patch features).

Inputs are ToTensor-only ([0,1], no mean/std). A ``DummyVisionEncoder`` mirrors the
interface for weight-free smoke tests.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..config import VisionConfig

# TIPS v2 published variants -> (HF DPT id, vision embed dim). No "S" in v2.
TIPS_V2 = {
    "B": ("google/tipsv2-b14-dpt", 768),
    "L": ("google/tipsv2-l14-dpt", 1024),
    "SO": ("google/tipsv2-so400m14-dpt", 1152),
    "g": ("google/tipsv2-g14-dpt", 1536),
}


class TipsVisionEncoder(nn.Module):
    """Wraps the TIPSv2 DPT backbone's vision encoder; returns a patch feature map."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        hf_id, dim = TIPS_V2[cfg.variant]
        if dim != cfg.embed_dim:
            raise ValueError(f"variant {cfg.variant} has embed_dim {dim}, config says {cfg.embed_dim}")
        self.embed_dim = dim
        self.patch_size = cfg.patch_size
        self.feature_mode = cfg.feature_mode  # "intermediate" | "standard" | "value"

        from transformers import AutoModel

        token = os.environ.get("HF_TIPSv2") or os.environ.get("HF_TOKEN")
        dpt = AutoModel.from_pretrained(hf_id, trust_remote_code=True, token=token)
        dpt.eval()
        dpt._get_backbone()  # triggers backbone download
        self.dpt = dpt
        self.backbone = dpt._backbone
        self.vision = self.backbone.vision_encoder

        if cfg.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    @property
    def num_register_tokens(self) -> int:
        return getattr(self.vision, "num_register_tokens", 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H_pix, W_pix] -> [B, embed_dim, H_pix//patch, W_pix//patch]``."""
        _, _, hp, wp = pixel_values.shape
        h, w = hp // self.patch_size, wp // self.patch_size

        if self.feature_mode == "intermediate":
            try:
                feats = self.vision.get_intermediate_layers(
                    pixel_values, n=1, reshape=True, norm=True
                )
                return feats[-1]  # [B, C, h, w]
            except (AttributeError, TypeError):
                pass  # fall back to the plain forward below

        if self.feature_mode == "value":
            tokens = self._value_attention_tokens(pixel_values)  # [B, N, C]
        else:  # "standard" (or intermediate fallback)
            out = self.vision(pixel_values)
            if isinstance(out, (tuple, list)):
                tokens = out[-1]
            elif isinstance(out, dict):
                tokens = out.get("x_norm_patchtokens", out.get("patches"))
            else:
                tokens = out

        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def _value_attention_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Last-block value-attention dense features (mirrors the TIPSv2 demo)."""
        v = self.vision
        x = v.prepare_tokens_with_masks(pixel_values)
        for blk in v.blocks[:-1]:
            x = blk(x)
        blk = v.blocks[-1]

        b, n, c = x.shape
        h = blk.attn.num_heads
        qkv = blk.attn.qkv(blk.norm1(x)).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        val = qkv[2].transpose(1, 2).reshape(b, n, c)
        val = blk.attn.proj(val)
        x = blk.ls1(val) + x
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        x = v.norm(x)
        return x[:, 1 + self.num_register_tokens:, :]  # drop CLS + register prefix


class DummyVisionEncoder(nn.Module):
    """Shape-compatible stand-in for smoke tests (no checkpoint download)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_values)  # [B, C, H_pix//patch, W_pix//patch]


def build_vision_encoder(cfg: VisionConfig, dummy: bool = False) -> nn.Module:
    return DummyVisionEncoder(cfg) if dummy else TipsVisionEncoder(cfg)
