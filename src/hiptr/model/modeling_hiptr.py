"""HipTR VLM: TIPS encoder -> pixel-shuffle MLP -> Qwen decoder.

The key correctness fix over ``placeholder.py``: visual tokens are **spliced into
``<image>`` placeholder positions** via a masked scatter, instead of being
``cat``'d in front with mismatched labels. This means:

  * the dataset inserts exactly ``sum(grid_tokens(h, w))`` ``<image>`` ids (one per
    visual token, summed over the image's units),
  * ``labels`` are ``-100`` at image/prompt/pad positions (loss only on the target),
  * batching is trivial (no per-sample length surgery in the model),
  * units are encoded across the batch in the **same order** the placeholders appear,
    so ``inputs_embeds[image_mask] = visual_tokens`` lines up row-for-row.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import HipTRConfig
from ..vision.tips_encoder import build_vision_encoder
from .projector import PixelShuffleProjector

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HipTRForHTR(nn.Module):
    """Pass ``tokenizer=`` for the production path, or inject ``llm=`` /
    ``image_token_id=`` for tests (avoids downloading weights)."""

    def __init__(self, cfg: HipTRConfig, *, tokenizer=None, llm=None,
                 image_token_id: Optional[int] = None, dummy_vision: bool = False):
        super().__init__()
        self.cfg = cfg
        self.vision = build_vision_encoder(cfg.vision, dummy=dummy_vision)
        self.llm = llm if llm is not None else self._load_llm(cfg)

        if tokenizer is not None:
            # tokenizer already has HipTR special tokens added (see data/tokens.py)
            self.llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
            image_token_id = tokenizer.convert_tokens_to_ids(cfg.tokens.image_token)
        if image_token_id is None:
            raise ValueError("provide tokenizer= (production) or image_token_id= (testing)")
        self.image_token_id = image_token_id

        self.connector = PixelShuffleProjector(
            vision_dim=self.vision.embed_dim,
            llm_dim=self.llm.config.hidden_size,  # read, never hardcode (vs placeholder)
            scale=cfg.connector.pixel_shuffle,
            depth=cfg.connector.mlp_depth,
            activation=cfg.connector.activation,
        )

    @staticmethod
    def _load_llm(cfg: HipTRConfig):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            cfg.llm.model_id,
            torch_dtype=_DTYPE[cfg.llm.dtype],
            attn_implementation=cfg.llm.attn_implementation,
        )

    # --- staged-training freeze helpers ------------------------------------
    def set_trainable(self, stage: str) -> None:
        def freeze(m, flag):
            for p in m.parameters():
                p.requires_grad = flag

        if stage == "align":            # only the connector learns
            freeze(self.vision, False)
            freeze(self.llm, False)
            freeze(self.connector, True)
        elif stage == "sft":            # connector + LLM (LoRA applied separately)
            freeze(self.vision, False)
            freeze(self.llm, True)
            freeze(self.connector, True)
        elif stage == "encoder":        # everything (low LR on vision)
            freeze(self.vision, True)
            freeze(self.llm, True)
            freeze(self.connector, True)
        else:
            raise ValueError(f"unknown stage {stage!r}")

    # --- forward -----------------------------------------------------------
    def encode_images(self, units: List[torch.Tensor]) -> torch.Tensor:
        """Encode a list of image units ``[3, H, W]`` -> ``[total_tokens, llm_hidden]``.

        Units may have different sizes (native rectangular mode), so each is encoded
        on its own and the resulting token rows are concatenated in list order — the
        same order the ``<image>`` placeholders appear across the batch.
        """
        device = next(self.llm.parameters()).device
        rows = []
        for u in units:
            x = u.unsqueeze(0).to(device)                # [1, 3, H, W]
            fmap = self.vision(x)                         # [1, C, h, w]
            tokens = self.connector(fmap)                # [1, n_tok, llm_hidden]
            rows.append(tokens[0])
        return torch.cat(rows, dim=0)                    # [total_tokens, llm_hidden]

    def _embed(self, input_ids, units):
        embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]
        image_mask = input_ids == self.image_token_id        # [B, L]
        if units and image_mask.any():
            visual = self.encode_images(units).to(embeds.dtype)  # [N_img, H]
            n_slots = int(image_mask.sum().item())
            if visual.shape[0] != n_slots:
                raise ValueError(
                    f"{visual.shape[0]} visual tokens != {n_slots} <image> placeholders. "
                    "The dataset inserts grid_tokens(h,w) placeholders per unit and the "
                    "collator must keep units in the same order."
                )
            embeds = embeds.clone()
            embeds[image_mask] = visual  # row-major scatter, batch-then-sequence order
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,                 # [B, L] with <image> placeholders
        attention_mask: torch.Tensor,            # [B, L]
        pixel_values: List[torch.Tensor],        # flat list of [3, H, W] units (whole batch)
        labels: Optional[torch.Tensor] = None,   # [B, L], -100 except target tokens
    ):
        embeds = self._embed(input_ids, pixel_values)
        return self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **gen_kwargs):
        embeds = self._embed(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=embeds, attention_mask=attention_mask, **gen_kwargs
        )
