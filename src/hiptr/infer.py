"""Inference: image -> structured transcription.

Loads a trained HipTR checkpoint, tiles the page, and greedily decodes the §4
output (line bbox + text). For very long full pages, prefer the detector ->
recognizer mode (transcribe detected line crops) or an R-SWA decoder; see
DESIGN.md §6.
"""
from __future__ import annotations

import argparse

import torch

from .config import HipTRConfig
from .data.dataset import INSTRUCTION
from .data.tokens import add_htr_tokens
from .model.modeling_hiptr import HipTRForHTR
from .vision.preprocess import count_image_tokens, prepare_image


def load_model(cfg: HipTRConfig, ckpt: str | None, dummy_vision: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = HipTRForHTR(cfg, tokenizer=tokenizer, dummy_vision=dummy_vision)
    add_htr_tokens(tokenizer, model.llm, cfg.tokens)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def transcribe(model, tokenizer, cfg: HipTRConfig, image_path: str, max_new_tokens: int = 1024) -> str:
    from PIL import Image

    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    units = prepare_image(image, cfg)                     # list of [3, H, W]

    image_token_id = tokenizer.convert_tokens_to_ids(cfg.tokens.image_token)
    instr = tokenizer(INSTRUCTION, add_special_tokens=False).input_ids
    n_img = count_image_tokens(units, cfg)
    input_ids = torch.tensor([instr + [image_token_id] * n_img], device=device)
    attention_mask = torch.ones_like(input_ids)

    out = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=units,
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dummy-vision", action="store_true")
    args = ap.parse_args()

    cfg = HipTRConfig()
    cfg.llm.model_id = args.llm
    model, tokenizer = load_model(cfg, args.ckpt, dummy_vision=args.dummy_vision)
    print(transcribe(model, tokenizer, cfg, args.image))


if __name__ == "__main__":
    main()
