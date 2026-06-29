"""Training/eval loop orchestration — small free functions, no god class."""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..config import HipTRConfig
from ..data.alto import parse_page
from ..data.dataset import ALTOHTRDataset, HTRCollator
from ..eval import corpus_cer, transcription
from .schedule import build_optimizer, build_scheduler
from .setup import build_model_and_tokenizer


def _to_device(batch: dict, device: torch.device) -> dict:
    # pixel_values is a list of variable-size units; only move the tensors
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _trainable(model) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch(model, loader, optimizer, scheduler, device, cfg: HipTRConfig) -> float:
    model.train()
    use_amp = device.type == "cuda"
    running = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
        (out.loss / cfg.train.grad_accum).backward()
        if (step + 1) % cfg.train.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(_trainable(model), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        running += out.loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, tokenizer, cfg: HipTRConfig, pairs: List[Tuple[str, str]]) -> float:
    """Page-level CER over a small validation set (generate -> strip markup -> compare)."""
    from ..infer import transcribe

    model.eval()
    d = cfg.data
    refs, hyps = [], []
    for img_path, xml_path in pairs[: cfg.train.eval_max_samples]:
        pred = transcribe(model, tokenizer, cfg, img_path)
        hyps.append(transcription(pred))
        gt = parse_page(
            xml_path, num_bins=cfg.tokens.num_loc_bins, output=d.output,
            region_geometry=d.region_geometry, line_geometry=d.line_geometry,
            include_region_type=d.include_region_type, poly_max_points=d.poly_max_points,
        )
        refs.append(transcription(gt))
    return corpus_cer(refs, hyps)


def save_checkpoint(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def _val_pairs(cfg: HipTRConfig) -> List[Tuple[str, str]]:
    if not cfg.train.val_img_dir or not cfg.train.val_xml_dir:
        return []
    imgs = sorted(glob.glob(os.path.join(cfg.train.val_img_dir, cfg.data.image_glob)))
    xmls = sorted(glob.glob(os.path.join(cfg.train.val_xml_dir, cfg.data.xml_glob)))
    return list(zip(imgs, xmls))


def run_training(cfg: HipTRConfig, dummy_vision: bool = False, resume: Optional[str] = None):
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = build_model_and_tokenizer(cfg, dummy_vision=dummy_vision)
    model.to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location="cpu"), strict=False)

    dataset = ALTOHTRDataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=HTRCollator(tokenizer.pad_token_id),
    )
    steps_per_epoch = max(1, len(loader) // cfg.train.grad_accum)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, steps_per_epoch * cfg.train.epochs, cfg)

    val_pairs = _val_pairs(cfg)
    best = float("inf")
    for epoch in range(cfg.train.epochs):
        loss = train_one_epoch(model, loader, optimizer, scheduler, device, cfg)
        metric, name = loss, "loss"
        if val_pairs:
            metric, name = evaluate(model, tokenizer, cfg, val_pairs), "val_cer"
        print(f"[{cfg.train.stage}] epoch {epoch} | loss {loss:.4f} | {name} {metric:.4f}")

        save_checkpoint(model, os.path.join(cfg.train.output_dir, f"hiptr_{cfg.train.stage}_ep{epoch}.pt"))
        if metric < best:
            best = metric
            save_checkpoint(model, os.path.join(cfg.train.output_dir, f"hiptr_{cfg.train.stage}_best.pt"))
    return model
