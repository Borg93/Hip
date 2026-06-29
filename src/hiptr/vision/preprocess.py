"""Image preprocessing for TIPSv2.

Matches the official TIPS recipe: **ToTensor only** (pixels in ``[0,1]``), with
``IMAGE_MEAN=(0,0,0)`` / ``IMAGE_STD=(1,1,1)`` — i.e. **no** normalization. Three
input modes, all returning a **list of image "units"** ``[3, H, W]`` (each side a
multiple of ``patch_size * pixel_shuffle``):

  * ``native`` (default) — one aspect-preserving **rectangular** unit; the longer
    side is resized toward ``native_target`` and both sides snapped to the divisor.
    This is the official dense recipe (see the TIPS foreground-seg Colab) and wastes
    no tokens on padding — ideal for handwriting/line images.
  * ``single`` — one square unit at ``resolution`` (pad-to-square or squish).
  * ``anyres`` — many square tiles (+ thumbnail) for very large / high-DPI pages.

Returning a list (rather than a stacked tensor) lets each unit have its own size;
the model encodes units one at a time and concatenates their tokens.
"""
from __future__ import annotations

from typing import List

import torch

from ..config import HipTRConfig


def to_tensor(image) -> torch.Tensor:
    """PIL RGB -> ``[3, H, W]`` float in ``[0,1]`` (ToTensor; no mean/std)."""
    import numpy as np

    a = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def _snap(value: float, divisor: int, lo: int, hi: int) -> int:
    v = max(float(lo), min(float(hi), value))
    return max(divisor, int(round(v / divisor)) * divisor)


def _native_resize(image, cfg: HipTRConfig):
    """Aspect-preserving rectangular resize, each side snapped to the divisor."""
    from PIL import Image

    vi = cfg.vision_input
    d = cfg.divisor
    w, h = image.size
    scale = vi.native_target / max(w, h)
    W = _snap(w * scale, d, vi.native_min_side, vi.native_max_side)
    H = _snap(h * scale, d, vi.native_min_side, vi.native_max_side)
    return image.resize((W, H), Image.BICUBIC)


def _resize_pad(image, res: int):
    """Aspect-preserving resize into ``res x res``, white-letterboxed."""
    from PIL import Image

    w, h = image.size
    scale = res / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (res, res), (255, 255, 255))
    canvas.paste(resized, ((res - nw) // 2, (res - nh) // 2))
    return canvas


def prepare_image(image, cfg: HipTRConfig) -> List[torch.Tensor]:
    """Return a list of image units ``[3, H, W]`` for one page image."""
    vi = cfg.vision_input

    if vi.mode == "native":
        return [to_tensor(_native_resize(image, cfg))]

    if vi.mode == "single":
        res = vi.resolution
        if vi.aspect == "squish":
            from PIL import Image

            page = image.resize((res, res), Image.BICUBIC)
        elif vi.aspect == "pad":
            page = _resize_pad(image, res)
        else:
            raise ValueError(f"unknown aspect mode {vi.aspect!r}")
        return [to_tensor(page)]

    if vi.mode == "anyres":
        from .tiling import tile_image

        tiles = tile_image(image, vi.tile_size, vi.max_tiles, vi.use_thumbnail)
        return [to_tensor(t) for t in tiles]

    raise ValueError(f"unknown vision input mode {vi.mode!r}")


def count_image_tokens(units: List[torch.Tensor], cfg: HipTRConfig) -> int:
    """Total ``<image>`` placeholders for a list of units (sum of per-unit grids)."""
    return sum(cfg.grid_tokens(u.shape[1], u.shape[2]) for u in units)
