"""Dataset + collator for HipTR.

Each sample becomes:

    [ instruction tokens ][ <image> * sum(grid_tokens per unit) ][ target tokens ][ eos ]
      labels: -100 ........ -100 .............................. target ........... eos

Only the target contributes to the loss. ``n_units`` is 1 in single-pass mode or
``n_tiles`` in anyres mode (see vision/preprocess.py), so the number of ``<image>``
placeholders is computed per sample and must equal the number of visual tokens the
model produces for that image.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..config import HipTRConfig
from ..vision.preprocess import count_image_tokens, prepare_image
from .alto import parse_alto

INSTRUCTION = "Transcribe the handwriting: for each line give its polygon and text, in reading order."


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
        units = prepare_image(image, self.cfg)                 # list of [3, H, W]
        n_image_tokens = count_image_tokens(units, self.cfg)

        target = parse_alto(
            self.xml_paths[idx],
            num_bins=self.cfg.tokens.num_loc_bins,
            granularity=self.cfg.data.granularity,
            poly_max_points=self.cfg.data.poly_max_points,
        )
        target_ids = self.tok(target, add_special_tokens=False).input_ids
        target_ids = target_ids[: self.cfg.data.max_target_len] + [self.tok.eos_token_id]

        image_ids = [self.image_token_id] * n_image_tokens
        input_ids = self._instruction_ids + image_ids + target_ids
        labels = ([-100] * (len(self._instruction_ids) + len(image_ids))) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "units": units,  # list of [3, H, W]; sizes may vary across samples
        }


class HTRCollator:
    """Dynamic right-padding; image units concatenated across the batch in order."""

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
        # flat list of units across the batch, in the SAME order as the <image>
        # placeholders appear (sample 0's units first, then sample 1's, ...)
        pixel_values: List[torch.Tensor] = []
        for b in batch:
            pixel_values.extend(b["units"])
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": pixel_values,
        }
