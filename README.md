# HipTR

**A compact, open Vision–Language Model for end-to-end Handwritten Text Recognition (HTR)** —
the **TIPSv2** vision encoder connected to a small **Qwen3.5-0.8B** decoder.

Feed a **whole page**; the model outputs its **layout (regions + types), reading order, and
transcription** in a single pass — no line segmentation, no separate detector — exportable
straight to **ALTO / PAGE-XML**. TIPS supplies dense, spatially-aware patch features (good for
small strokes and page layout); the Qwen decoder turns them into the structured page.

> Read **[DESIGN.md](./DESIGN.md)** for the full architecture, training recipe, and the
> rationale (including a point-by-point critique of the original `placeholder.py` sketch).
> See **[DATA.md](./DATA.md)** for the data + training recipe — HipTR targets the data-rich
> case: a large labeled page corpus, trained directly.

## Why these pieces

| Part | Choice | Reason |
|---|---|---|
| Vision | **TIPSv2 L/14** (1024-dim, patch-14) | dense patch features + strong spatial awareness — a better fit for HTR than a global CLIP/SigLIP encoder. Loads from HF: `google/tipsv2-l14-dpt` (`trust_remote_code=True`) → `._backbone.vision_encoder` |
| Connector | pixel-shuffle ×2 + 2-layer MLP | compresses visual tokens (full pages explode patch counts) and bridges 1024 → LLM hidden |
| Decoder | **Qwen3.5-0.8B** | small, multilingual, 262K context, natively multimodal; `hidden_size` 1024 matches TIPS L/14 |

> ⚠️ There is **no Qwen3.5-0.6B** — the smallest Qwen3.5 is **0.8B**, which is what HipTR uses.

Design lineage: a frozen high-res ViT + MLP connector + a small autoregressive LLM that emits
geometry as coordinate tokens — the standard recipe for grounded document VLMs. See DESIGN.md §1.

## Layout

```
DESIGN.md                  full design doc
DATA.md                    data + training recipe (data-rich, direct training)
pyproject.toml             packaging (hatchling, src layout), ruff + pytest config
placeholder.py             original napkin sketch (kept for reference; superseded by src/)
configs/base.yaml          all the knobs
scripts/fetch_tips.sh      optional: vendor TIPS source for the offline/.npz path
src/hiptr/                 vision/ · model/ · data/ · training/ · eval.py · infer.py
tests/                     test_alto, test_eval (pure-Python) + test_smoke (torch)
```

## Quickstart

```bash
# 1. install (editable; add the train extra for LoRA/accelerate)
pip install -e ".[train,dev]"

# 2. fast checks: data/token/eval logic (no torch) + the CPU forward/backward smoke test
python tests/test_alto.py && python tests/test_eval.py
pytest -m "not gpu"        # includes the tiny-LLM forward/backward smoke test

# 3. authenticate with HuggingFace for the gated TIPSv2 repo (loaded automatically)
export HF_TIPSv2=hf_...     # or HF_TOKEN
# No clone needed: the encoder pulls google/tipsv2-l14-dpt via trust_remote_code.
# scripts/fetch_tips.sh is only for the offline source/.npz path.

# 4. exercise the full pipeline WITHOUT TIPS weights (random vision encoder)
python -m hiptr.train --stage align --dummy-vision \
  --img-dir ./data/images --xml-dir ./data/page_xml

# 5. real training, staged (see DESIGN.md §5)
python -m hiptr.train --stage align --llm Qwen/Qwen3.5-0.8B   # projector only
python -m hiptr.train --stage sft                              # + LLM (LoRA)
python -m hiptr.train --stage encoder                          # optional: unfreeze vision

# 6. inference
python -m hiptr.infer --image page.jpg --ckpt checkpoints/hiptr_sft_ep4.pt
```

Run module commands from the `src/` directory (or `pip install -e .`) so `hiptr` is importable.

## Data

Whole **page** images + ALTO/PAGE-XML transcriptions (Transkribus / eScriptorium exports, READ,
IAM, Bentham, Norhand, …). `src/hiptr/data/alto.py` serializes each page — in reading order —
into the end-to-end target: regions (type + polygon) containing their text.

```
<page><region><type>paragraph</type><poly><loc_100><loc_143>...</poly>der Briefträger kam
am Morgen</region><region><type>heading</type><poly>...</poly>Kapitel Ett</region></page>
```

Coordinates are normalized + quantized to 1000 bins and added to the tokenizer as atomic
`<loc_*>` tokens (with `<page>`/`<region>`/`<type>`/`<poly>`). Tune via `data.output`
(`page` | `lines` | `text`), `data.region_geometry` / `line_geometry` (`poly`|`bbox`|`none`),
and `data.poly_max_points`.

## Status

Research scaffold. The model code is correct and importable, the data/token layer is tested,
but a real run needs (a) HF access to the gated `google/tipsv2-*-dpt` repo (`HF_TIPSv2`/`HF_TOKEN`)
and (b) a GPU. Open items are tracked in **DESIGN.md §9**.
