# Data & training recipe

HipTR is built for the **data-rich** case: you already have a large labeled corpus of
**full pages** (images + ALTO/PAGE-XML with layout, reading order, and transcription). That
changes everything — **you train directly on your own data.** No synthetic generation, no
pseudo-labeling, no distillation.

---

## The only hard requirement

The connector (the MLP between TIPS and Qwen) starts **randomly initialized**. Until it learns to
map TIPS's visual features into Qwen's embedding space, the decoder sees noise. So the one thing
training must do is fit the connector (and adapt the decoder) on **image → structured-page-text**
examples. With a large labeled page set, that's exactly what you have.

Everything else people reach for — synthetic line rendering, teacher-VLM pseudo-labels, self-
training — exists only to *manufacture* labels when you don't have them. **You do, so skip them.**

---

## Recipe (maps to `train.py --stage …`)

| Stage | Trains | Data | Notes |
|---|---|---|---|
| **A. Warm-up** *(optional)* | connector only (vision ❄, LLM ❄) | a slice of your pages (e.g. 5–20k) | LR ~1e-3, 1 epoch. Stabilises the random connector before touching the LLM. With lots of data you can also skip this and go straight to B. |
| **B. SFT** | connector + LLM (LoRA or full); vision ❄ | **all** your labeled pages | LR ~1e-5 (LLM) / 2e-5 (connector). This is where the model learns handwriting + the full-page output format. |
| **C. Encoder unfreeze** *(optional)* | + last *k* TIPS blocks | same, smaller LR (~5e-6) | Only if B plateaus and you want domain gains; do it last. |

- **Full fine-tune vs LoRA:** with >100k labeled pages you have enough signal to **full-fine-tune**
  the decoder if you have the GPUs; LoRA is the cheaper default for fast iteration.
- **Loss** is computed only on the target tokens (prompt + image positions are masked), so the
  model is scored on producing the page structure + text, nothing else.

### Rough sizing (engineering estimate, not a guarantee)

Because both backbones are pretrained, the labeled set mostly buys *handwriting style + your output
format*, not language modeling. With a large in-domain corpus you should reach strong in-domain CER;
the practical levers on quality are **input resolution** (`vision_input`), **how much you unfreeze**,
and **label/layout consistency** — not raw data volume, which you already have.

---

## What you can skip (and when you'd revisit)

| Technique | Skip because… | Revisit only if… |
|---|---|---|
| Synthetic line/page rendering | you have real labels | you add a new script/language with little data |
| Teacher-VLM pseudo-labeling | you have real labels | you want to label a large *untranscribed* backlog |
| Self-training | you have real labels | adapting to an unlabeled new collection |

---

## Practical notes

- **Split by document/writer/collection**, not by random page — pages from one volume are highly
  correlated, and a random split inflates your numbers.
- **Augment** to bridge scan variation: elastic distortion, ink bleed, slight rotation, contrast
  jitter. This matters more than more data once the corpus is large.
- **Layout + reading-order quality is its own axis.** CER on text says nothing about whether
  regions and reading order are right — track region IoU, region-type accuracy, and a reading-order
  score too (see DESIGN §8). Garbage reading order in the labels teaches garbage reading order.
- **Long pages = long targets.** A dense page is a long output sequence; watch decode cost and, for
  very large pages, scale the *input* with `anyres` tiling rather than splitting the output
  (DESIGN §6).
- **Resolution is the main quality/cost dial** (`vision_input.native_target`). Small or faint hands
  need more pixels; bump it before reaching for anything exotic.
