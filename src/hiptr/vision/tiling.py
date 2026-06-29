"""AnyRes dynamic tiling (LLaVA-NeXT / InternVL style).

Why this exists: HTR needs high resolution, but squishing a page to a square
(what the placeholder did with ``resize((1024,1024))``) distorts glyph aspect
ratios — fatal for handwriting. Instead we pick the tile grid whose aspect ratio
best matches the page, resize to that grid, and split into fixed ``tile_size``
tiles. A downsized thumbnail is appended for global layout context.

``tile_size`` must be a multiple of ``patch_size * pixel_shuffle`` so the patch
grid divides cleanly through the connector.
"""
from __future__ import annotations

from typing import List, Tuple


def _candidate_grids(max_tiles: int) -> List[Tuple[int, int]]:
    grids = set()
    for n in range(1, max_tiles + 1):
        for cols in range(1, n + 1):
            if n % cols == 0:
                grids.add((cols, n // cols))
    return sorted(grids)


def choose_grid(width: int, height: int, max_tiles: int, tile_size: int) -> Tuple[int, int]:
    """Pick (cols, rows) minimizing aspect-ratio distortion vs the source image."""
    ar = width / max(1, height)
    best, best_err = (1, 1), float("inf")
    for cols, rows in _candidate_grids(max_tiles):
        grid_ar = (cols * tile_size) / (rows * tile_size)
        err = abs(grid_ar - ar)
        # tie-break toward more tiles (more resolution) for a given aspect match
        if err < best_err - 1e-6 or (abs(err - best_err) <= 1e-6 and cols * rows > best[0] * best[1]):
            best, best_err = (cols, rows), err
    return best


def tile_image(image, tile_size: int, max_tiles: int, use_thumbnail: bool = True):
    """Return a list of ``tile_size x tile_size`` PIL tiles for one page image.

    The thumbnail (if enabled) is the last tile. The number of tiles is
    ``cols*rows (+1)``; the caller multiplies by ``tokens_per_tile`` to know how
    many ``<image>`` placeholders to insert.
    """
    from PIL import Image

    w, h = image.size
    cols, rows = choose_grid(w, h, max_tiles, tile_size)
    resized = image.resize((cols * tile_size, rows * tile_size), Image.BICUBIC)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((tile_size, tile_size), Image.BICUBIC))
    return tiles
