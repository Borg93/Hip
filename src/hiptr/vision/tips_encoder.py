"""TIPSv2 vision-encoder wrapper.

TIPS is **not** a HuggingFace ``AutoModel`` (the placeholder's
``AutoModel.from_pretrained("google/tipsv2-l14-dpt")`` cannot work). The PyTorch
checkpoints are ``.npz`` files loaded via the repo's own ``image_encoder.py``
(``VisionTransformer`` / ``vit_large(patch_size=14)``), whose ``forward`` returns
``(cls1, cls2, patch_features)``. This wrapper:

  * imports the vendored TIPS package (see ``scripts/fetch_tips.sh``),
  * loads the ``.npz`` checkpoint,
  * interpolates positional embeddings to the tile grid if needed,
  * returns **only the dense patch tokens** ``[B, num_patches, embed_dim]``.

A ``DummyVisionEncoder`` with the same interface is provided so the rest of the
pipeline (tiling, splicing, collation, loss) can be smoke-tested without weights.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

from ..config import VisionConfig

_VARIANT_FACTORY = {
    "S": ("vit_small", 384),
    "B": ("vit_base", 768),
    "L": ("vit_large", 1024),
    "SO": ("vit_so400m", 1152),
    "g": ("vit_giant2", 1536),
}


class TipsVisionEncoder(nn.Module):
    """Wraps the vendored TIPS ViT and exposes patch tokens."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.model = self._build_tips_model(cfg)
        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _build_tips_model(self, cfg: VisionConfig) -> nn.Module:
        if cfg.tips_pkg_path and os.path.isdir(cfg.tips_pkg_path):
            if cfg.tips_pkg_path not in sys.path:
                sys.path.insert(0, cfg.tips_pkg_path)
        try:
            import image_encoder as tips_ie  # from vendored third_party/tips/pytorch
        except ImportError as e:  # pragma: no cover - depends on vendored code
            raise ImportError(
                "Could not import the vendored TIPS 'image_encoder'. Run "
                "scripts/fetch_tips.sh to vendor github.com/google-deepmind/tips "
                f"into '{cfg.tips_pkg_path}'. Original error: {e}"
            )
        factory_name, dim = _VARIANT_FACTORY[cfg.variant]
        if dim != cfg.embed_dim:
            raise ValueError(
                f"variant {cfg.variant} has embed_dim {dim}, but config says {cfg.embed_dim}"
            )
        factory = getattr(tips_ie, factory_name)
        model = factory(patch_size=cfg.patch_size)
        if cfg.checkpoint_path:
            self._load_npz(model, cfg.checkpoint_path)
        return model

    @staticmethod
    def _load_npz(model: nn.Module, path: str) -> None:
        """Load a TIPS .npz checkpoint into the torch ViT.

        The exact key mapping comes from the vendored repo's loader (see
        run_image_encoder_inference.py). We defer to it if available, else fall
        back to a direct state_dict load for .pt/.pth files.
        """
        if path.endswith(".npz"):
            try:
                from convert_checkpoint import load_npz_into_model  # type: ignore
                load_npz_into_model(model, path)
                return
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Loading TIPS .npz requires the vendored loader. Mirror the "
                    "key mapping from run_image_encoder_inference.py. Error: " + str(e)
                )
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def _maybe_eval_grad(self):
        return torch.no_grad() if self.cfg.freeze else torch.enable_grad()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T, T] -> [B, num_patches, embed_dim]`` (patch tokens only)."""
        out = self.model(pixel_values)
        # TIPS forward returns (cls1, cls2, patch_features). Be tolerant of dict too.
        if isinstance(out, dict):
            patch = out.get("x_norm_patchtokens", out.get("patch_features"))
        elif isinstance(out, (tuple, list)):
            patch = out[-1]
        else:
            patch = out
        return patch


class DummyVisionEncoder(nn.Module):
    """Shape-compatible stand-in for smoke tests (no checkpoint needed)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        # a trivial learnable projection so .parameters() is non-empty
        self.proj = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values)  # [B, C, H/p, W/p]
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)  # [B, h*w, C]


def build_vision_encoder(cfg: VisionConfig, dummy: bool = False) -> nn.Module:
    return DummyVisionEncoder(cfg) if dummy else TipsVisionEncoder(cfg)
