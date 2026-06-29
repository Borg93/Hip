"""Staged training entrypoint for HipTR.

Implements the §5 recipe: ``align`` (projector only) -> ``sft`` (LLM+projector,
LoRA by default) -> ``encoder`` (optional, low-LR vision unfreeze). Loss is causal
cross-entropy on target tokens only (handled by the dataset's ``-100`` labels).

This is a reference loop, not a launcher — wire it to your accelerator/Trainer of
choice. Run with ``--dummy-vision`` to exercise the plumbing without TIPS weights.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .config import HipTRConfig
from .data.dataset import ALTOHTRDataset, HTRCollator
from .data.tokens import add_htr_tokens
from .model.modeling_hiptr import HipTRForHTR


def _param_groups(model: HipTRForHTR, cfg: HipTRConfig):
    """One LR per component so each stage trains the right thing at the right rate."""
    groups = []
    proj = [p for p in model.connector.parameters() if p.requires_grad]
    if proj:
        groups.append({"params": proj, "lr": cfg.train.lr_projector})
    llm = [p for p in model.llm.parameters() if p.requires_grad]
    if llm:
        groups.append({"params": llm, "lr": cfg.train.lr_llm})
    vis = [p for p in model.vision.parameters() if p.requires_grad]
    if vis:
        groups.append({"params": vis, "lr": cfg.train.lr_vision})
    return groups


def build(cfg: HipTRConfig, dummy_vision: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = HipTRForHTR(cfg, tokenizer, dummy_vision=dummy_vision)
    add_htr_tokens(tokenizer, model.llm, cfg.tokens)

    model.set_trainable(cfg.train.stage)
    if cfg.train.stage == "sft" and cfg.llm.use_lora:
        _apply_lora(model, cfg)
    return model, tokenizer


def _apply_lora(model: HipTRForHTR, cfg: HipTRConfig):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.llm.lora_r, lora_alpha=cfg.llm.lora_alpha, lora_dropout=cfg.llm.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora)
    for p in model.connector.parameters():  # keep the connector fully trainable
        p.requires_grad = True


def train(cfg: HipTRConfig, dummy_vision: bool = False):
    torch.manual_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build(cfg, dummy_vision=dummy_vision)
    model.to(device)

    ds = ALTOHTRDataset(cfg, tokenizer)
    loader = DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=HTRCollator(tokenizer.pad_token_id),
    )
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.train.weight_decay)

    model.train()
    for epoch in range(cfg.train.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            # pixel_values is a list of variable-size units; the model moves them itself
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = out.loss / cfg.train.grad_accum
            loss.backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            running += out.loss.item()
        print(f"[{cfg.train.stage}] epoch {epoch} | loss {running / max(1, len(loader)):.4f}")
        torch.save(model.state_dict(), f"{cfg.train.output_dir}/hiptr_{cfg.train.stage}_ep{epoch}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="align", choices=["align", "sft", "encoder"])
    ap.add_argument("--img-dir", default="./data/images")
    ap.add_argument("--xml-dir", default="./data/alto_xml")
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dummy-vision", action="store_true", help="run without TIPS weights")
    args = ap.parse_args()

    cfg = HipTRConfig()
    cfg.train.stage = args.stage
    cfg.data.img_dir = args.img_dir
    cfg.data.xml_dir = args.xml_dir
    cfg.llm.model_id = args.llm
    train(cfg, dummy_vision=args.dummy_vision)


if __name__ == "__main__":
    main()
