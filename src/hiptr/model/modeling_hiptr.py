"""HipTR VLM: TIPS encoder -> pixel-shuffle MLP -> Qwen decoder.

The key correctness fix over ``placeholder.py``: visual tokens are **spliced into
``<image>`` placeholder positions** via a masked scatter, instead of being
``cat``'d in front with mismatched labels. This means:

  * the dataset inserts exactly ``n_tiles * tokens_per_tile`` ``<image>`` ids,
  * ``labels`` are ``-100`` at image/prompt/pad positions (loss only on the target),
  * batching is trivial (no per-sample length surgery in the model),
  * tiles are concatenated across the batch in the **same order** the placeholders
    appear, so ``inputs_embeds[image_mask] = visual_tokens`` lines up row-for-row.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import HipTRConfig
from ..vision.tips_encoder import build_vision_encoder
from .projector import PixelShuffleProjector

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HipTRForHTR(nn.Module):
    def __init__(self, cfg: HipTRConfig, tokenizer, dummy_vision: bool = False):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.vision = build_vision_encoder(cfg.vision, dummy=dummy_vision)
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm.model_id,
            torch_dtype=_DTYPE[cfg.llm.dtype],
            attn_implementation=cfg.llm.attn_implementation,
        )
        # tokenizer already has HipTR special tokens added (see data/tokens.py)
        self.llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        llm_hidden = self.llm.config.hidden_size  # read, never hardcode (vs placeholder)
        self.connector = PixelShuffleProjector(
            vision_dim=self.vision.embed_dim,
            llm_dim=llm_hidden,
            scale=cfg.connector.pixel_shuffle,
            depth=cfg.connector.mlp_depth,
            activation=cfg.connector.activation,
        )
        self.image_token_id = tokenizer.convert_tokens_to_ids(cfg.tokens.image_token)
        self.tokens_per_tile = cfg.tokens_per_tile

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
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[n_tiles, 3, T, T] -> [n_tiles*tokens_per_tile, llm_hidden]`` (flat)."""
        patch = self.vision(pixel_values)            # [n_tiles, num_patches, vis_dim]
        tokens = self.connector(patch)               # [n_tiles, tokens_per_tile, llm_hidden]
        return tokens.reshape(-1, tokens.shape[-1])

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, L] with <image> placeholders
        attention_mask: torch.Tensor,    # [B, L]
        pixel_values: torch.Tensor,      # [sum_tiles, 3, T, T] across the whole batch
        labels: Optional[torch.Tensor] = None,  # [B, L], -100 except target tokens
    ):
        embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]
        image_mask = input_ids == self.image_token_id        # [B, L]

        if pixel_values is not None and image_mask.any():
            visual = self.encode_images(pixel_values).to(embeds.dtype)  # [N_img, H]
            n_slots = int(image_mask.sum().item())
            if visual.shape[0] != n_slots:
                raise ValueError(
                    f"{visual.shape[0]} visual tokens != {n_slots} <image> placeholders. "
                    "The dataset must insert n_tiles*tokens_per_tile placeholders per sample "
                    "and the collator must concat tiles in the same order."
                )
            embeds = embeds.clone()
            embeds[image_mask] = visual  # row-major scatter, batch-then-sequence order

        return self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **gen_kwargs):
        embeds = self.llm.get_input_embeddings()(input_ids)
        image_mask = input_ids == self.image_token_id
        if pixel_values is not None and image_mask.any():
            visual = self.encode_images(pixel_values).to(embeds.dtype)
            embeds = embeds.clone()
            embeds[image_mask] = visual
        return self.llm.generate(
            inputs_embeds=embeds, attention_mask=attention_mask, **gen_kwargs
        )
