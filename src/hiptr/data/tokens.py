"""Special-token management for HipTR.

The placeholder emitted strings like ``<loc_512>`` but never added them to the
tokenizer, so BPE would shatter each coordinate into several pieces. Here we build
the coordinate + structural tokens as *atomic* special tokens and (at runtime)
register them on the tokenizer and resize the model embeddings.

The token-list construction is pure-Python so it can be imported and tested
without ``transformers`` installed.
"""
from __future__ import annotations

from typing import List

from ..config import TokenConfig


def location_tokens(num_bins: int = 1000) -> List[str]:
    """``['<loc_0>', ..., '<loc_999>']`` — quantized coordinate atoms."""
    return [f"<loc_{i}>" for i in range(num_bins)]


def structural_tokens(cfg: TokenConfig) -> List[str]:
    return [cfg.image_token, cfg.line_open, cfg.line_close, cfg.poly_open, cfg.poly_close]


def all_special_tokens(cfg: TokenConfig) -> List[str]:
    """Every token HipTR adds to the base Qwen vocabulary."""
    return structural_tokens(cfg) + location_tokens(cfg.num_loc_bins)


def add_htr_tokens(tokenizer, model=None, cfg: TokenConfig | None = None):
    """Register HipTR special tokens and resize embeddings to match.

    Returns the dict of added-token ids keyed by token string. Call this once,
    before training, after loading the base tokenizer/model.
    """
    cfg = cfg or TokenConfig()
    new_tokens = all_special_tokens(cfg)
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if model is not None and added > 0:
        # pad to a multiple of 8 for tensor-core friendliness
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in new_tokens}
    return ids
