"""Dataset + collator for HipTR.

Each sample becomes:

    [ instruction tokens ][ <image> * (n_tiles*tokens_per_tile) ][ target tokens ][ eos ]
      labels: -100 ........ -100 ............................... target ........... eos

Only the target contributes to the loss. ``n_tiles`` is decided per-image by the
AnyRes tiler, so the number of ``<image>`` placeholders is computed per sample and
must equal the number of visual tokens the model will produce for that image.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..config import HipTRConfig
from ..vision.tiling import tile_image
from .alto import parse_alto

INSTRUCTION = "Transcribe the handwriting with line coordinates."


def _to_pixel_tensor(tiles, mean, std) -> torch.Tensor:
    import numpy as np

    arrs = []
    for t in tiles:
        a = torch.from_numpy(np.asarray(t.convert("RGB"), dtype="float32") / 255.0)
        a = a.permute(2, 0, 1)  # [3, H, W]
        for c in range(3):
            a[c] = (a[c] - mean[c]) / std[c]
        arrs.append(a)
    return torch.stack(arrs, dim=0)  # [n_tiles, 3, T, T]


class ALTOHTRDataset(Dataset):
    def __init__(self, cfg: HipTRConfig, tokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        d = cfg.data
        self.img_paths = sorted(glob.glob(os.path.join(d.img_dir, d.image_glob)))
        self.xml_paths = sorted(glob.glob(os.path.join(d.xml_dir, d.xml_glob)))
        if len(self.img_paths) != len(self.xml_paths):
            raise ValueError(
                f"{len(self.img_paths)} images vs {len(self.xml_paths)} xml files; "
                "expected a 1:1 pairing after sorting."
            )
        self.image_token_id = tokenizer.convert_tokens_to_ids(cfg.tokens.image_token)
        self._instruction_ids = tokenizer(INSTRUCTION, add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image

        image = Image.open(self.img_paths[idx]).convert("RGB")
        tiles = tile_image(
            image,
            tile_size=self.cfg.tiling.tile_size,
            max_tiles=self.cfg.tiling.max_tiles,
            use_thumbnail=self.cfg.tiling.use_thumbnail,
        )
        pixel_values = _to_pixel_tensor(tiles, self.cfg.vision.image_mean, self.cfg.vision.image_std)
        n_image_tokens = len(tiles) * self.cfg.tokens_per_tile

        target = parse_alto(
            self.xml_paths[idx],
            num_bins=self.cfg.tokens.num_loc_bins,
            granularity=self.cfg.data.granularity,
        )
        target_ids = self.tok(target, add_special_tokens=False).input_ids
        target_ids = target_ids[: self.cfg.data.max_target_len] + [self.tok.eos_token_id]

        image_ids = [self.image_token_id] * n_image_tokens
        input_ids = self._instruction_ids + image_ids + target_ids
        labels = ([-100] * (len(self._instruction_ids) + len(image_ids))) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class HTRCollator:
    """Dynamic right-padding; tiles concatenated across the batch in sample order."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = b["input_ids"].size(0)
            pad = max_len - n
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), self.pad_token_id)]))
            labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
            attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            # concat tiles in the SAME order as the <image> placeholders appear
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        }
