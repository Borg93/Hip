# HipTR

**A compact, open Vision–Language Model for Handwritten Text Recognition (HTR)** — built by
connecting the **TIPSv2** vision encoder to a small **Qwen3.5 / Qwen3** decoder.

The idea: TIPS gives you *dense, spatially-aware* patch features (great for the small strokes
and layout of handwriting); a 0.6–0.8B Qwen decoder turns those features into a structured
transcription — **text + reading order + coordinates** — in one model, exportable straight to
**ALTO / PAGE-XML**.

> Read **[DESIGN.md](./DESIGN.md)** for the full architecture, training recipe, and the
> rationale (including a point-by-point critique of the original `placeholder.py` sketch).
> See **[DATA.md](./DATA.md)** for how much data this actually needs (and how Surya-OCR-2
> reached strong results without hyperscaler-scale human labels).

## Why these pieces

| Part | Choice | Reason |
|---|---|---|
| Vision | **TIPSv2 L/14** (1024-dim, patch-14) | dense patch features + strong spatial awareness — a better fit for HTR than a global CLIP/SigLIP encoder. Loads from HF: `google/tipsv2-l14-dpt` (`trust_remote_code=True`) → `._backbone.vision_encoder` |
| Connector | pixel-shuffle ×2 + 2-layer MLP | compresses visual tokens (full pages explode patch counts) and bridges 1024 → LLM hidden |
| Decoder | **Qwen3.5-0.8B** | small, multilingual, 262K context, natively multimodal; `hidden_size` 1024 matches TIPS L/14 |

> ⚠️ There is **no Qwen3.5-0.6B** — the smallest Qwen3.5 is **0.8B**, which is what HipTR uses.

Prior art that shaped the design: **Surya-OCR-2** (≈650M Qwen3.5-style OCR VLM — closest
analog), **Eagle/Embodied LocateAnything** (ViT+MLP+Qwen, parallel box decoding), and
**Unlimited-OCR** (R-SWA for long one-shot page parsing). See DESIGN.md §1.

## Layout

```
DESIGN.md                  full design doc
DATA.md                    how much data is needed + bootstrapping without big labels
placeholder.py             original napkin sketch (kept for reference; superseded by src/)
configs/base.yaml          all the knobs
scripts/fetch_tips.sh      optional: vendor TIPS source for the offline/.npz path
src/hiptr/                 the package (config, vision, model, data, train, infer)
tests/test_alto.py         runnable without torch
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. smoke-test the data/token logic (no torch needed)
python tests/test_alto.py

# 3. authenticate with HuggingFace for the gated TIPSv2 repo (loaded automatically)
export HF_TIPSv2=hf_...     # or HF_TOKEN
# No clone needed: the encoder pulls google/tipsv2-l14-dpt via trust_remote_code.
# scripts/fetch_tips.sh is only for the offline source/.npz path.

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

Pairs of page images + ALTO/PAGE-XML transcriptions (IAM, READ/ICDAR-HTR, Bentham, Norhand,
Transkribus / eScriptorium exports, …). `src/hiptr/data/alto.py` serializes them into the
training target — per line, **polygon + text in reading order** (Surya-style):

```
<line><poly><loc_100><loc_143><loc_699><loc_143><loc_699><loc_171><loc_100><loc_171></poly>der Briefträger kam</line>
<line><poly>...</poly>am Morgen</line>
```

Coordinates are normalized + quantized to 1000 bins and added to the tokenizer as atomic
`<loc_*>` tokens (with `<line>`/`<poly>`). Switch geometry via `data.granularity`:
`polygon` (default) | `line` (bbox) | `word`; cap polygon vertices with `data.poly_max_points`.

## Status

Research scaffold. The model code is correct and importable, the data/token layer is tested,
but a real run needs (a) HF access to the gated `google/tipsv2-*-dpt` repo (`HF_TIPSv2`/`HF_TOKEN`)
and (b) a GPU. Open items are tracked in **DESIGN.md §9**.
