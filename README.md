# HipTR

**A compact, open Vision–Language Model for Handwritten Text Recognition (HTR)** — built by
connecting the **TIPSv2** vision encoder to a small **Qwen3.5 / Qwen3** decoder.

The idea: TIPS gives you *dense, spatially-aware* patch features (great for the small strokes
and layout of handwriting); a 0.6–0.8B Qwen decoder turns those features into a structured
transcription — **text + reading order + coordinates** — in one model, exportable straight to
**ALTO / PAGE-XML**.

> Read **[DESIGN.md](./DESIGN.md)** for the full architecture, training recipe, and the
> rationale (including a point-by-point critique of the original `placeholder.py` sketch).

## Why these pieces

| Part | Choice | Reason |
|---|---|---|
| Vision | **TIPSv2 L/14** (1024-dim, patch-14) | dense patch features + strong spatial awareness — a better fit for HTR than a global CLIP/SigLIP encoder |
| Connector | pixel-shuffle ×2 + 2-layer MLP | compresses visual tokens (full pages explode patch counts) and bridges 1024 → LLM hidden |
| Decoder | **Qwen3.5-0.8B** (or **Qwen3-0.6B**) | small, multilingual, 262K context, Apache-2.0 |

> ⚠️ There is **no Qwen3.5-0.6B**. The smallest Qwen3.5 is **0.8B**; the literal 0.6B is
> **Qwen3-0.6B**. Both are wired in — pick via config.

Prior art that shaped the design: **Surya-OCR-2** (≈650M Qwen3.5-style OCR VLM — closest
analog), **Eagle/Embodied LocateAnything** (ViT+MLP+Qwen, parallel box decoding), and
**Unlimited-OCR** (R-SWA for long one-shot page parsing). See DESIGN.md §1.

## Layout

```
DESIGN.md                  full design doc
placeholder.py             original napkin sketch (kept for reference; superseded by src/)
configs/base.yaml          all the knobs
scripts/fetch_tips.sh      vendor the TIPS encoder into third_party/
src/hiptr/                 the package (config, vision, model, data, train, infer)
tests/test_alto.py         runnable without torch
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. smoke-test the data/token logic (no torch needed)
python tests/test_alto.py

# 3. vendor the TIPS vision encoder (clone may be blocked in restricted sandboxes)
bash scripts/fetch_tips.sh

# 4. exercise the full pipeline WITHOUT TIPS weights (random vision encoder)
python -m hiptr.train --stage align --dummy-vision \
  --img-dir ./data/images --xml-dir ./data/alto_xml

# 5. real training, staged (see DESIGN.md §5)
python -m hiptr.train --stage align --llm Qwen/Qwen3.5-0.8B   # projector only
python -m hiptr.train --stage sft                              # + LLM (LoRA)
python -m hiptr.train --stage encoder                          # optional: unfreeze vision

# 6. inference
python -m hiptr.infer --image page.jpg --ckpt checkpoints/hiptr_sft_ep4.pt
```

Run module commands from the `src/` directory (or `pip install -e .`) so `hiptr` is importable.

## Data

Pairs of page images + ALTO-XML transcriptions (IAM, READ/ICDAR-HTR, Bentham, Norhand,
Transkribus exports, …). `src/hiptr/data/alto.py` serializes ALTO into the training target:

```
<line><loc_100><loc_143><loc_699><loc_171>der Briefträger kam</line>
<line><loc_100><loc_186><loc_599><loc_214>am Morgen</line>
```

Coordinates are normalized + quantized to 1000 bins and added to the tokenizer as atomic
`<loc_*>` tokens. Switch to word-level boxes with `data.granularity: word`.

## Status

Research scaffold. The model code is correct and importable, the data/token layer is tested,
but a real run needs (a) the vendored TIPS weights and (b) a GPU. Open items are tracked in
**DESIGN.md §9**.
