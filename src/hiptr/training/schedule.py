"""Optimizer + LR schedule: one learning rate per component, cosine + warmup."""
from __future__ import annotations

from typing import List

import torch

from ..config import HipTRConfig


def param_groups(model, cfg: HipTRConfig) -> List[dict]:
    """One param group (and LR) per trainable component, so each stage trains the
    right thing at the right rate."""
    groups = []
    for module, lr in (
        (model.connector, cfg.train.lr_projector),
        (model.llm, cfg.train.lr_llm),
        (model.vision, cfg.train.lr_vision),
    ):
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr})
    return groups


def build_optimizer(model, cfg: HipTRConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.train.weight_decay)


def build_scheduler(optimizer, num_training_steps: int, cfg: HipTRConfig):
    from transformers import get_cosine_schedule_with_warmup

    warmup = int(cfg.train.warmup_ratio * num_training_steps)
    return get_cosine_schedule_with_warmup(optimizer, warmup, max(1, num_training_steps))
