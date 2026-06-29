"""CLI entrypoint for staged HipTR training.

Stages (see DESIGN.md §5): ``align`` (connector only) -> ``sft`` (connector + LLM,
LoRA) -> ``encoder`` (optional vision unfreeze). Use ``--dummy-vision`` to exercise
the pipeline without downloading TIPS weights.
"""
from __future__ import annotations

import argparse

from .config import HipTRConfig
from .training import run_training


def _parse_args():
    ap = argparse.ArgumentParser(description="Train HipTR (TIPSv2 + Qwen3.5-0.8B for HTR).")
    ap.add_argument("--stage", default="align", choices=["align", "sft", "encoder"])
    ap.add_argument("--img-dir", default="./data/images")
    ap.add_argument("--xml-dir", default="./data/alto_xml")
    ap.add_argument("--val-img-dir", default="")
    ap.add_argument("--val-xml-dir", default="")
    ap.add_argument("--llm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--output-dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dummy-vision", action="store_true", help="run without TIPS weights")
    return ap.parse_args()


def main():
    args = _parse_args()
    cfg = HipTRConfig()
    cfg.train.stage = args.stage
    cfg.train.epochs = args.epochs
    cfg.train.output_dir = args.output_dir
    cfg.train.val_img_dir = args.val_img_dir
    cfg.train.val_xml_dir = args.val_xml_dir
    cfg.data.img_dir = args.img_dir
    cfg.data.xml_dir = args.xml_dir
    cfg.llm.model_id = args.llm
    run_training(cfg, dummy_vision=args.dummy_vision, resume=args.resume)


if __name__ == "__main__":
    main()
