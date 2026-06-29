# HipTR — A TIPSv2 + Qwen VLM Optimized for Handwritten Text Recognition

> Status: design / research scaffold. This document fleshes out the idea of bolting the
> **TIPSv2** vision encoder onto a small **Qwen3.5 / Qwen3** decoder to build a compact,
> open vision–language model (VLM) specialized for **Handwritten Text Recognition (HTR)**
> and historical-document transcription.

---

## 1. Goal & positioning

Build a **small (≈1B-parameter class) end-to-end VLM** that takes a page (or line) image of
**handwritten** text and emits a structured transcription — text **plus** reading order and
**coordinates** — directly, without a separate OCR/segmentation cascade where possible.

### Why this combination

| Component | Choice | Why |
|---|---|---|
| Vision encoder | **TIPSv2 L/14** (1024-dim, 303M, patch-14) | TIPS is explicitly optimized for **dense, spatially-aware patch features** (its headline result is spatial awareness + patch–text alignment). HTR is a *dense, spatial* task — every patch matters and small strokes carry signal. This is a better fit than a CLIP/SigLIP encoder tuned for global semantics. |
| Connector | **Pixel-shuffle + 2-layer MLP** | Reduces visual token count (full pages explode patch counts) and bridges 1024 → LLM hidden. Mirrors InternVL / LLaVA-NeXT / Eagle's MLP connector. |
| Language decoder | **Qwen3.5-0.8B** (or **Qwen3-0.6B**) | Small, multilingual (119 langs), 262K context, Apache-2.0. The smallest *Qwen3.5* is **0.8B** — there is **no Qwen3.5-0.6B**; if you want the literal 0.6B use **Qwen3-0.6B**. Surya-OCR-2 uses a "Qwen3.5-style ~650M" decoder for exactly this job, which validates the size class. |

### How it differs from the reference projects

- **Surya-OCR-2** (`datalab-to/surya-ocr-2`) is already a ~650M Qwen3.5-style VLM that emits
  layout-JSON / full-page HTML, with a *separate* EfficientViT line detector. It is the closest
  prior art and proves the size class works. **Our differentiators:** (a) the **TIPSv2** encoder
  (stronger dense spatial features than a generic encoder), and (b) a deliberate focus on
  **handwriting + historical documents** with an **ALTO/PAGE-XML-native** coordinate output
  format, rather than print-first general OCR.
- **Eagle / Embodied → LocateAnything** (`NVlabs/Eagle/Embodied`) contributes the
  **ViT + MLP-projector + Qwen** recipe and, importantly, **Parallel Box Decoding (PBD)** and
  continual-SFT with configurable freezing/LoRA — directly relevant to predicting line/word
  **bounding boxes** efficiently.
- **Unlimited-OCR** (arXiv 2606.23050, Baidu) contributes **Reference Sliding-Window Attention
  (R-SWA)** for *constant* KV-cache during very long one-shot page parsing — the right answer
  to "a full handwritten page is a very long output sequence."

So the lineage is: **TIPS** (encoder) × **Eagle/Surya** (connector + coordinate decoding) ×
**Unlimited-OCR** (long-output efficiency), specialized for **HTR**.

---

## 2. Architecture

```
                 page image (variable size)
                          │
        ┌─────────────────┴──────────────────┐
        │  AnyRes tiling (aspect-preserving,  │   tiles are multiples of patch_size
        │  multiple-of-14 tiles + thumbnail)  │
        └─────────────────┬──────────────────┘
                          │  [n_tiles, 3, T, T]
                 ┌────────▼─────────┐
                 │  TIPSv2 L/14     │  frozen (stage A) → optionally unfrozen (stage C)
                 │  vision encoder  │  returns patch tokens, drops CLS/register tokens
                 └────────┬─────────┘
                          │  [n_tiles, (T/14)^2, 1024]
                 ┌────────▼─────────┐
                 │ Pixel-shuffle ×2 │  (T/14)^2 → (T/28)^2 tokens, 1024 → 4096
                 │   + MLP (GELU)   │  4096 → llm_hidden
                 └────────┬─────────┘
                          │  visual tokens in LLM embedding space
        ┌─────────────────▼──────────────────┐
        │  splice into <image> placeholder    │  masked scatter; labels = -100 here
        │  positions of the prompt embeddings │
        └─────────────────┬──────────────────┘
                          │
                 ┌────────▼─────────┐
                 │  Qwen3.5-0.8B    │  causal LM; cross-entropy on target tokens only
                 │  decoder         │
                 └────────┬─────────┘
                          │
        structured transcription (see §4 output format)
```

### Token budget (the central HTR tension)

HTR needs **high resolution** (small handwriting, diacritics, faded ink) but high resolution
explodes the visual-token count, which a 0.6–0.8B decoder cannot absorb cheaply.

- A single 448×448 tile at patch-14 → 32×32 = **1024 patches**. Pixel-shuffle ×2 → **256 tokens**.
- A full A4 page tiled at 448 to roughly preserve detail might be ~3×4 = 12 tiles + 1 thumbnail
  → 13 × 256 ≈ **3,328 visual tokens** before any text. That is fine for Qwen's 262K context but
  expensive for the decoder. Hence: **pixel-shuffle compression is not optional**, and the tile
  grid is a tunable quality/cost knob (`configs/base.yaml: vision.max_tiles`).

This is exactly why `placeholder.py`'s "resize the whole page to 1024×1024 and feed it raw" is
the wrong default (see §7).

---

## 3. Resolution, tiling & preprocessing

- **Aspect-ratio preservation.** Squishing a page to 1024×1024 distorts glyph shapes — fatal for
  handwriting. We use **AnyRes / LLaVA-NeXT-style dynamic tiling**: pick the tile grid whose
  aspect ratio best matches the page, resize to that grid, split into fixed `T×T` tiles, and add
  one downsized **thumbnail** tile for global layout.
- **Patch divisibility.** Tile size `T` must be a multiple of `patch_size` (14) **and** of
  `patch_size × pixel_shuffle` (28) so shuffle is clean. `448 = 14×32 = 28×16` ✓.
  (`1024` used in the placeholder is **not** divisible by 14 — the placeholder silently relies on
  the encoder to crop/pad, losing edge text.)
- **Position embeddings.** TIPS is trained at a fixed grid; feeding `T=448` requires
  **bicubic interpolation of the positional embeddings** to the tile grid. The wrapper does this
  once at load and caches it. (Confirm TIPSv2's native train resolution from the cloned repo.)
- **Normalization.** Use TIPS's expected mean/std (confirm from the repo; default placeholder of
  `pixel/255` only is almost certainly wrong). Configurable in `vision.image_mean/std`.

---

## 4. Output format

Two production-relevant targets, both supported by the dataset layer:

**(a) Line-level, layout-aware (recommended canonical target)** — one segment per text line,
each prefixed by its quantized bounding box, in reading order:

```
<line><loc_023><loc_041><loc_512><loc_058>der Briefträger kam am Morgen</line>
<line><loc_024><loc_060><loc_498><loc_077>und brachte die Nachricht ...</line>
```

**(b) Word-level** (what `placeholder.py` sketched) — `<loc_x><loc_y>word` per token. Higher
coordinate density but more brittle reading order; kept as an option.

**Coordinate tokens.** Coordinates are **quantized to 1000 bins** (0–999), normalized to page
size, and added to the tokenizer as **special tokens** `<loc_0>…<loc_999>` (+ `<line>`,`</line>`,
`<image>`). This is the Florence-2 / Pix2Struct / Surya approach. Without adding them as atoms,
`<loc_512>` tokenizes into several BPE pieces and the model wastes capacity learning to spell
numbers. **`placeholder.py` never adds these tokens** — a correctness bug.

This format makes the model emit **transcription + reading order + geometry** in one pass,
exportable straight to **ALTO** or **PAGE-XML**, the standard HTR interchange formats.

---

## 5. Training recipe

Staged training (LLaVA / Eagle "continual SFT" style) — never train a randomly-initialized
projector while also updating pretrained weights:

| Stage | Trainable | Data | LR | Purpose |
|---|---|---|---|---|
| **A. Alignment** | projector only (vision ❄, LLM ❄) | large, cheap **printed** OCR pairs (image → text) | ~1e-3 | teach the connector to map TIPS features into Qwen's space |
| **B. HTR SFT** | projector + LLM (LoRA or full); vision ❄ | **handwritten** pages w/ ALTO/PAGE-XML (IAM, READ/ICDAR, Bentham, Norhand, transkribus exports) | ~1e-5 LLM / 2e-5 proj | learn handwriting + the coordinate/line output format |
| **C. (optional) Encoder unfreeze** | + last *k* TIPS blocks | same as B, smaller | ~5e-6 | squeeze domain gains; risk of catastrophic forgetting — do last, low LR |

- **Loss:** causal-LM cross-entropy on **target tokens only** — prompt, image placeholders, and
  padding are masked with `-100`. (`placeholder.py` sets `labels = input_ids` over the *whole*
  concatenated sequence including image tokens → it trains the model to "predict" image
  embeddings and the prompt. Bug.)
- **Sequence-length mismatch:** when visual tokens are spliced in, labels must be expanded with
  `-100` for every visual position so logits and labels align. The placeholder's `cat` makes
  logits length `N_vis + N_text` while labels stay `N_text` → shape error / silent misalignment.
- **PEFT:** default to **LoRA** on the LLM (and projector full) for cheap iteration; full-FT as a
  later, well-resourced run. (Eagle supports exactly this freeze/LoRA matrix.)
- **Curriculum:** start with single lines / clean scans, progress to full pages / degraded
  historical documents. Augment: elastic distortion, ink bleed, rotation ±3°, contrast jitter.

---

## 6. Inference & long outputs

- **Two deployment modes:**
  1. **Detector → recognizer (pragmatic, Surya-style):** a small line detector (reuse Surya's
     EfficientViT-SegFormer or train one) crops lines; the VLM transcribes each crop. Robust,
     parallel, bounded output length. Best near-term accuracy/throughput.
  2. **End-to-end full page (research track):** one forward pass emits the whole §4 structure.
     Elegant, but the output is long → KV cache grows.
- **Long-output efficiency:** adopt **R-SWA** (Unlimited-OCR) or a sliding-window decoder to keep
  KV cache ~constant for full-page decoding; alternatively chunk by detected region. Qwen's 262K
  context is a safety net, not a license to ignore decode cost.
- **Decoding:** greedy/beam for fidelity; coordinate tokens benefit from constrained decoding
  (only `<loc_*>` valid in coordinate slots) — a cheap accuracy win.

---

## 7. What's wrong with `placeholder.py` (and how this design fixes it)

`placeholder.py` is a useful napkin sketch but won't train correctly as written:

| # | Issue in `placeholder.py` | Fix in this design |
|---|---|---|
| 1 | `AutoModel.from_pretrained("google/tipsv2-l14-dpt", trust_remote_code=True)` — TIPS is **not** a HF AutoModel; checkpoints are `.npz` loaded via the repo's own `image_encoder.py`. | `vision/tips_encoder.py` wraps the vendored TIPS `VisionTransformer` (`vit_large(patch_size=14)`), loads the `.npz`, returns patch tokens. |
| 2 | `self.vision_model(pixel_values).last_hidden_state` — TIPS forward returns `(cls1, cls2, patch_features)`, no `.last_hidden_state`. | Wrapper returns `patch_features` explicitly (CLS/register tokens dropped). |
| 3 | `Qwen/Qwen3.5-0.8B` is named but text refers to "Qwen3.5-0.6B" — **0.6B doesn't exist in 3.5**. | Config: default `Qwen/Qwen3.5-0.8B`, alt `Qwen/Qwen3-0.6B`; hidden size read from `config.hidden_size`, never hardcoded. |
| 4 | `projector = nn.Linear(1024, 1024)` — single linear, hardcoded dims, no token compression. | `PixelShuffleProjector`: shuffle ×2 then 2-layer MLP `(1024·4 → llm_hidden → llm_hidden)`, dims from configs. |
| 5 | `torch.cat([vis_tokens, text_embeds])` with `labels=input_ids` — labels don't cover visual positions → length mismatch; trains on prompt + image. | `<image>` placeholder splicing via masked scatter; labels `-100` on image/prompt/pad. |
| 6 | `image.resize((1024,1024))` — distorts aspect ratio; 1024 not divisible by 14. | AnyRes aspect-preserving tiling into 448 tiles (`14×32`) + thumbnail. |
| 7 | `pixel/255` normalization only. | TIPS mean/std normalization (configurable). |
| 8 | `<loc_x><loc_y>` strings never added to vocab. | `data/tokens.py` adds `<loc_0…999>`,`<line>`,`</line>`,`<image>` and resizes embeddings. |
| 9 | Full-FT everything at 5e-6 from a random projector. | Staged: align projector → SFT LLM+proj → optional encoder unfreeze. |
| 10 | `padding='max_length'` to 1024 every sample, no attention mask / no collator. | Dynamic padding collator with attention mask + image-token mask. |

---

## 8. Evaluation

- **Metrics:** CER / WER (primary), plus layout metrics (line IoU, reading-order/BLEU-on-order)
  for the end-to-end target. Report on held-out **handwritten** sets, not just print.
- **Benchmarks:** IAM, READ2016/ICDAR-HTR, Bentham, Norhand/handwritten-Norwegian, plus
  olmOCR-bench and a multilingual slice for parity with Surya-2 (83.3% olmOCR / 87.2% multilingual
  are the public targets to beat or match at this size).
- **Ablations:** TIPSv2 vs SigLIP/CLIP encoder; tile count; pixel-shuffle factor; LoRA vs full-FT;
  line-mode vs end-to-end.

---

## 9. Risks & open questions

1. **TIPS variable-resolution behavior.** Confirm TIPSv2's native train resolution and that
   bicubic pos-embed interpolation to 448 tiles holds dense quality (it's the linchpin).
2. **TIPS license & checkpoint access.** Verify the TIPSv2 weights' license permits derived
   model release; checkpoints come from `storage.googleapis.com/tips_data/` / HF `google/tips…`.
3. **Decoder capacity.** 0.6–0.8B may bottleneck multilingual + long-page. Have Qwen3.5-2B ready
   as the next rung.
4. **Visual-token count vs decoder cost.** The tile/shuffle budget is the main quality/latency
   dial; measure before committing.
5. **Coordinate supervision quality.** Historical ALTO/PAGE-XML coordinates are noisy; quantize
   robustly and consider line-only (drop word boxes) if word-level supervision is too noisy.
6. **Encoder is a JAX/Scenic-origin port.** The PyTorch path must load the `.npz` faithfully;
   validate feature parity against the reference notebook before training.

---

## 10. Repo layout

```
DESIGN.md                 ← this document
README.md                 ← overview + quickstart
requirements.txt
placeholder.py            ← original napkin sketch (kept for reference; superseded by src/)
configs/base.yaml         ← model + training configuration
scripts/fetch_tips.sh     ← vendor the TIPS pytorch code into third_party/ (clone is blocked here)
src/hiptr/
  config.py               ← dataclass configs
  vision/tips_encoder.py  ← TIPS wrapper (+ DummyVisionEncoder for smoke tests)
  vision/tiling.py        ← AnyRes aspect-preserving tiling
  model/projector.py      ← pixel-shuffle + MLP connector
  model/modeling_hiptr.py ← the VLM: image-token splicing, label masking
  data/tokens.py          ← location / structural special tokens
  data/alto.py            ← ALTO-XML parser → target string (pure stdlib, no torch)
  data/dataset.py         ← torch Dataset + dynamic-padding collator
  train.py                ← staged training entrypoint
  infer.py                ← inference / generate
tests/test_alto.py        ← runnable without torch
```
