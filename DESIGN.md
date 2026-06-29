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
| Language decoder | **Qwen3.5-0.8B** | Small, multilingual, 262K context, natively multimodal; `hidden_size` 1024 matches TIPS L/14. It is the smallest Qwen3.5 (there is no 0.6B in the 3.5 line); ~0.8B is enough because both backbones are pretrained — the connector and the HTR head are what we actually train. |

### What's distinctive

- **End-to-end full page.** One forward pass over the whole page emits **layout (regions +
  types), reading order, and transcription** — no line segmentation and no separate detector.
- **TIPSv2 dense features.** A vision encoder built for spatial / patch–text alignment, which
  suits the small strokes and dense layout of historical handwriting better than a globally-pooled
  CLIP/SigLIP encoder.
- **ALTO/PAGE-XML-native output.** Geometry is emitted as coordinate tokens and maps directly back
  to the archival interchange formats, so predictions round-trip to Transkribus / eScriptorium.

Technique notes: the MLP connector + coordinate-token decoding is the standard grounded-VLM
recipe; for very long pages a sliding-window / constant-KV-cache decoder keeps full-page
generation affordable (see §6).

---

## 2. Architecture

```
                 page image (variable size)
                          │
        ┌─────────────────┴──────────────────┐
        │  preprocess (ToTensor, [0,1]):      │   native = aspect-preserving rect
        │  native (default) | single | anyres │   every side a multiple of 28
        └─────────────────┬──────────────────┘
                          │  list of units [3, H, W]  (sizes may vary)
                 ┌────────▼─────────┐
                 │  TIPSv2 L/14     │  HF AutoModel (-dpt, trust_remote_code);
                 │  vision encoder  │  get_intermediate_layers → [B,1024,h,w]
                 └────────┬─────────┘
                          │  [B, 1024, H/14, W/14]  patch feature map
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

- **Native** (default): a portrait page snapped to e.g. 1372×980 → 98×70 patches → ×2 shuffle →
  49×35 = **1,715 tokens** (and *no* padding tokens — the win over a square pass).
- **Single pass**: @896 → 64×64 = 4096 patches → ×2 shuffle → **1024 tokens**; @1372 (square) →
  98×98 → **2401 tokens**; @1792 → 128×128 → **4096 tokens**.
- **AnyRes**: a 448 tile → 32×32 = 1024 patches → ×2 shuffle → **256 tokens**; a full page at
  ~12 tiles + thumbnail → ~3,328 tokens.
- All of these fit Qwen's 262K context but are expensive for a 0.8B decoder. Hence
  **pixel-shuffle compression is not optional**, and the resolution / tile grid is the tunable
  quality–cost knob (`configs/base.yaml: vision_input.native_target` / `resolution` / `max_tiles`).

This is exactly why `placeholder.py`'s "resize the whole page to 1024×1024 and feed it raw" is
the wrong default (see §7).

---

## 3. Resolution, input modes & preprocessing

TIPSv2 natively accepts a **resolution ladder of 224→1792** (all multiples of patch-14:
224, 336, 448, 672, 896, 1120, 1372, 1792). That changes the calculus from the original sketch:
for most pages a single high-res pass is enough, and explicit tiling is only needed for very
large / high-DPI scans.

- **Three input modes** (`vision_input.mode`):
  - **`native` (default, recommended).** One **aspect-preserving rectangular** unit: resize the
    longer side toward `native_target` (default **1372**) and snap **both** sides to the divisor
    `patch_size × pixel_shuffle` (28). No padding waste, glyph aspect ratio preserved. This is the
    official TIPS dense recipe (`resize_transform` in the foreground-seg Colab: height fixed,
    width scaled, both rounded to ×14) — the encoder handles rectangular inputs via pos-embed
    interpolation.
  - **`single`.** One square pass at `vision_input.resolution` (default 896). `aspect="pad"`
    letterboxes on white; `aspect="squish"` reproduces the demo's `Resize((res,res))`.
  - **`anyres`.** LLaVA-NeXT-style dynamic tiling into `tile_size` tiles (+ thumbnail) for pages
    whose detail exceeds 1792px or whose aspect ratio is extreme.
- **Normalization = ToTensor only.** TIPS uses `IMAGE_MEAN=(0,0,0)`, `IMAGE_STD=(1,1,1)` — pixels
  in `[0,1]` with **no** mean/std (confirmed in both the HF demo and the Colab). The placeholder's
  `pixel/255` was actually right; the trap is adding a mean/std the encoder never saw.
- **Patch divisibility.** Every side must be a multiple of `patch_size` (14) **and** of
  `patch_size × pixel_shuffle` (28) so the shuffle is clean. `896 = 28×32` ✓, `1372 = 28×49` ✓,
  `448 = 28×16` ✓; native mode snaps to this automatically. (The placeholder's `1024` is **not**
  divisible by 14.)
- **Position embeddings.** TIPSv2 is trained mostly at 224/448; higher/rectangular sizes rely on
  the encoder's **pos-embed interpolation** (`interpolate_antialias=True`). It works (the demo
  exposes the whole ladder and the Colab feeds rectangular inputs), but treat very high
  resolutions as interpolation-dependent and validate dense quality — small handwriting is exactly
  where this matters.
- **Feature path.** `vision.feature_mode="intermediate"` (default) uses
  `get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]` — the official dense-task API
  (used for the seg probe), returning a `[B, C, H, W]` map. `"value"` uses the value-attention
  last-block surgery (sharper per-patch features); `"standard"` is the plain forward.

---

## 4. Output format

The model reads a **whole page** and emits its **layout + reading order + transcription** as one
sequence. A page is a list of **regions in reading order**; each region carries a type and
geometry, then its text:

```
<page><region><type>paragraph</type><poly><loc_100><loc_143>…</poly>der Briefträger kam
am Morgen</region><region><type>heading</type><poly>…</poly>Kapitel Ett</region></page>
```

- **Layout** = the region `<type>` (paragraph, heading, marginalia, table, …) plus its polygon.
- **Reading order** = the order regions are emitted; no separate order head, the sequence *is*
  the order. Uses PAGE `<ReadingOrder>` when present, else document order; sort regions at
  data-prep time when geometry and reading order diverge (multi-column).
- **Transcription** = each region's text (its lines joined by newlines).

Geometry sources: PAGE `<Coords points>` or ALTO `<Shape><Polygon POINTS>` / `HPOS…`; a missing
polygon falls back to the bbox rectangle. `data.poly_max_points` (>0) subsamples long polygons.
Configurable via `data.*`: `output` (`page` | `lines` | `text`), `region_geometry` /
`line_geometry` (`poly`|`bbox`|`none`), `include_region_type`.

**Coordinate & structural tokens.** Coordinates are **quantized to 1000 bins** (0–999), normalized
to page size, and added to the tokenizer as **atomic special tokens** `<loc_0>…<loc_999>` plus
`<image>`, `<page>`, `<region>`, `<type>`, `<line>`, `<poly>` (and closers) — 1011 new tokens. The
Florence-2 / Pix2Struct coordinate-token approach; without atomic tokens `<loc_512>` shatters into
BPE pieces and the model wastes capacity spelling numbers. **`placeholder.py` never adds these.**

The output maps straight back to **ALTO / PAGE-XML** (see `eval.to_page_xml`).

---

## 5. Training recipe

Staged training — never train a randomly-initialized projector while also updating pretrained
weights. With a large labeled page corpus you train directly on your own data: Stage A is a short
projector warm-up, Stage B does the work, and synthetic data is not required.

| Stage | Trainable | Data | LR | Purpose |
|---|---|---|---|---|
| **A. Alignment** | projector only (vision ❄, LLM ❄) | a slice of your pages (image → page text) | ~1e-3 | teach the connector to map TIPS features into Qwen's space |
| **B. HTR SFT** | projector + LLM (LoRA or full); vision ❄ | your labeled **pages** (ALTO/PAGE-XML): layout + reading order + transcription | ~1e-5 LLM / 2e-5 proj | learn handwriting + the full-page output format |
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
- **Curriculum:** start with cleaner pages, progress to degraded historical ones. Augment:
  elastic distortion, ink bleed, rotation ±3°, contrast jitter.

---

## 6. Inference & long outputs

- **End-to-end, single pass.** One forward over the page emits the whole §4 structure — layout,
  reading order, and transcription together. No line detector, no per-line recognition pass.
- **Long-output efficiency.** A full page is a long target sequence, so the decoder's KV cache
  grows. Keep it affordable with a sliding-window / constant-KV-cache decoder; Qwen's 262K context
  is a safety net, not a license to ignore decode cost. For very large pages, scale the **input**
  with `anyres` tiling rather than splitting the output.
- **Decoding.** Greedy/beam for fidelity; constrained decoding (only `<loc_*>` valid in coordinate
  slots, structural tokens only where the grammar allows) is a cheap accuracy win.
- **Export.** `eval.transcription` strips markup for CER; `eval.parse_regions` + `eval.to_page_xml`
  round-trip predictions back to PAGE-XML.

---

## 7. What's wrong with `placeholder.py` (and how this design fixes it)

`placeholder.py` is a useful napkin sketch but won't train correctly as written:

| # | Issue in `placeholder.py` | Fix in this design |
|---|---|---|
| 1 | `self.vision_model(pixel_values).last_hidden_state` — the **load id is correct** (TIPSv2 *is* a HF AutoModel via the `-dpt` repos), but you must call `dpt._get_backbone()` and use `dpt._backbone.vision_encoder`, whose forward returns a **3-tuple** `(cls, _, patch_tokens)` — there is no `.last_hidden_state`. | `vision/tips_encoder.py` loads the DPT AutoModel, grabs `._backbone.vision_encoder`, and returns the 3rd element (patch tokens). |
| 2 | Plain forward only; no use of TIPSv2's dense-feature path. | Optional **value-attention** extraction (`feature_mode="value"`) gives sharper per-patch features for HTR; the `1 + num_register_tokens` prefix is dropped via the model's own attribute. |
| 3 | `Qwen/Qwen3.5-0.8B` is named but the comment refers to "Qwen3.5-0.6B" — **0.6B doesn't exist in 3.5**. | Config uses `Qwen/Qwen3.5-0.8B`; hidden size read from `config.hidden_size`, never hardcoded. |
| 4 | `projector = nn.Linear(1024, 1024)` — single linear, hardcoded dims, no token compression. | `PixelShuffleProjector`: shuffle ×2 then 2-layer MLP `(1024·4 → llm_hidden → llm_hidden)`, dims from configs. |
| 5 | `torch.cat([vis_tokens, text_embeds])` with `labels=input_ids` — labels don't cover visual positions → length mismatch; trains on prompt + image. | `<image>` placeholder splicing via masked scatter; labels `-100` on image/prompt/pad. |
| 6 | `image.resize((1024,1024))` — distorts aspect ratio; 1024 is not a TIPSv2 resolution and not divisible by 14. | Single high-res pass at a supported resolution (default 896, `aspect="pad"`) or AnyRes tiling — see §3. |
| 7 | `pixel/255` normalization only. | **Correct as-is** — TIPSv2 is ToTensor/`[0,1]` with no mean/std. (An earlier draft of this doc wrongly proposed mean/std; reverted.) |
| 8 | `<loc_x><loc_y>` strings never added to vocab. | `data/tokens.py` adds `<loc_0…999>` + `<image>/<page>/<region>/<type>/<line>/<poly>` (and closers) and resizes embeddings. |
| 9 | Full-FT everything at 5e-6 from a random projector. | Staged: align projector → SFT LLM+proj → optional encoder unfreeze. |
| 10 | `padding='max_length'` to 1024 every sample, no attention mask / no collator. | Dynamic padding collator with attention mask + image-token mask. |

---

## 8. Evaluation

- **Metrics:** CER / WER on the transcription (`eval.transcription` strips layout markup), plus
  layout/order metrics — region IoU, region-type accuracy, and a reading-order score — for the
  end-to-end target.
- **Benchmarks:** your held-out pages (primary), plus public HTR sets (READ2016/ICDAR, IAM,
  Bentham, Norhand) for comparability.
- **Ablations:** TIPSv2 vs SigLIP/CLIP encoder; native vs single vs anyres input; pixel-shuffle
  factor; `feature_mode` (intermediate/value/standard); LoRA vs full-FT; with/without region types.

---

## 9. Risks & open questions

1. **TIPS resolution interpolation.** TIPSv2 is trained mostly at 224/448; rungs up to 1792 rely
   on positional-embedding interpolation. Validate dense quality at the resolution you pick — this
   is the linchpin for small handwriting.
2. **TIPS access & license.** The encoder loads from gated HF repos (`google/tipsv2-*-dpt`,
   `trust_remote_code=True`); set `HF_TIPSv2`/`HF_TOKEN` and verify the weights' license permits a
   derived release.
3. **Decoder capacity.** 0.6–0.8B may bottleneck multilingual + long-page. Have Qwen3.5-2B ready
   as the next rung.
4. **Visual-token count vs decoder cost.** The tile/shuffle budget is the main quality/latency
   dial; measure before committing.
5. **Coordinate supervision quality.** Historical ALTO/PAGE-XML coordinates are noisy; quantize
   robustly and consider line-only (drop word boxes) if word-level supervision is too noisy.
6. **Register-token count.** `num_register_tokens` isn't documented per variant; the wrapper reads
   it from the model (`getattr(..., 1)`). Confirm it once real weights are loaded so the
   value-attention prefix slice (`1 + num_register_tokens`) is correct.

---

## 10. Repo layout

```
DESIGN.md                 ← this document
README.md                 ← overview + quickstart
DATA.md                   ← data budget + label-free bootstrapping
pyproject.toml            ← packaging (hatchling, src layout), ruff + pytest config
placeholder.py            ← original napkin sketch (kept for reference; superseded by src/)
configs/base.yaml         ← model + training configuration
scripts/fetch_tips.sh     ← optional: vendor the TIPS source for offline/reference use
src/hiptr/
  config.py               ← dataclass configs (vision, vision_input, connector, llm, …)
  eval.py                 ← CER/WER, output parsing, PAGE-XML export (pure stdlib)
  vision/tips_encoder.py  ← TIPSv2 HF loader + value-attention path (+ DummyVisionEncoder)
  vision/preprocess.py    ← ToTensor-only preprocessing; native / single / anyres
  vision/tiling.py        ← AnyRes aspect-preserving tiling (anyres mode)
  model/projector.py      ← pixel-shuffle + MLP connector
  model/modeling_hiptr.py ← the VLM: image-token splicing, label masking
  data/tokens.py          ← location / structural special tokens
  data/alto.py            ← ALTO/PAGE-XML parser → polygon target (pure stdlib, no torch)
  data/dataset.py         ← torch Dataset + dynamic-padding collator
  training/               ← setup · schedule · loop (staged trainer, no god class)
  train.py                ← thin CLI entrypoint
  infer.py                ← inference / generate
tests/                    ← test_alto, test_eval (no torch) + test_smoke (torch)
```
