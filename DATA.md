# Data budget — how much is actually needed to align TIPSv2 + Qwen3.5-0.8B for HTR

Short answer: **far less *human-labeled* data than people assume**, because the heavy lifting is
done by (1) **frozen pretrained backbones**, (2) **synthetic rendered text**, and (3)
**pseudo-labeling / distillation**. Human transcription is the *smallest* line item.

> Findings below were gathered by a multi-source research pass and adversarially verified
> (20 quantitative claims checked, 18 confirmed). Numbers are cited; uncertain ones are flagged.

---

## TL;DR

| Stage | What trains | Data **order of magnitude** | Nature of the data |
|---|---|---|---|
| **A. Connector alignment** | projector only (TIPSv2 ❄, Qwen ❄) | **~0.5–0.6M** image–text pairs, 1 epoch | generic/cheap (LLaVA-1.5 used 558K; MobileVLM/LLaVA-v1.0 ~595K) |
| **B. HTR SFT** | connector + LLM (LoRA/full), vision ❄ | **~5–20M lines** (single script) → **10s–100s M** (multilingual) | mostly **synthetic + pseudo-labeled**, not human |
| **C. Real fine-tune** (folded into B) | same | **~0.3K–50K human-labeled lines per script** | the only human-labeled part |

The reason the *labeled* set can be tiny: TrOCR hits **2.89% CER on IAM** fine-tuning on only
**~6,161 real lines**, and transfers to a new hand (Washington) at **~3.3% CER with ~325–350
labeled lines** (vs ~18.2% from scratch) — *because* it was pretrained on **684M synthetic printed
lines + 17.9M synthetic handwritten lines** first. The "data" is overwhelmingly free/rendered.

---

## Why labeled data is cheap in *this* architecture

1. **Both backbones are already pretrained.** Qwen3.5-0.8B supplies the language prior; TIPSv2
   supplies dense visual features. Stage-A alignment only has to learn the *vision→text mapping*,
   not language modeling — which is why ~0.5M pairs suffice instead of tens of millions.
2. **Frozen encoder + projector-only Stage A** is the validated LLaVA-1.5 recipe: 558K pairs,
   1 epoch, LR 1e-3, batch 256, both backbones frozen. MobileVLM (a *small*-LLM VLM) used the same
   ~595K order. So the alignment cost does **not** scale with model fame — it's ~half a million pairs.
3. **HTR specialization is sample-efficient** once a synthetic/pseudo-labeled prior exists.

---

## How Surya-OCR-2 did it without hyperscaler labels

Surya-OCR-2 is a single ~650M Qwen3.5-style VLM (shared across layout/OCR/table, with a separate
small EfficientViT line detector) scoring **83.3% olmOCR-bench** and **87.2% across 91 languages**.

- **Their exact training-data volume/composition is undisclosed** — the README only says the VLM is
  "trained on diverse document images to emit layout JSON or full-page HTML." Treat any specific
  number as unknown.
- The widely-repeated *"Common-Crawl PDFs filtered for bad OCR + synthetic PDFs"* line describes
  Surya's **evaluation/benchmark construction (and old v1 README), not its training set** — a common
  misattribution. Don't plan against it as a training recipe.
- What **is** transferable is the pattern Surya shares with TrOCR/olmOCR:
  1. start from strong pretrained backbones (don't train from scratch);
  2. lean on **synthetic rendering** for free pixel-perfect labels;
  3. **pseudo-label / distill** instead of hand-transcribing — e.g. olmOCR fine-tuned Qwen2-VL-7B on
     **~250–266K pages auto-labeled by GPT-4o** (document anchoring, ~$190/M pages, ~1/32 GPT-4o
     cost, **zero human transcription**) and beat larger systems;
  4. **self-train** on unlabeled in-domain data (AT-ST cut handwriting CER ~55% relative,
     6.43%→2.88%, from *unlabeled* target lines only).

> ⚠️ **Handwriting caveat:** the printed-OCR bootstrap (mining a PDF's embedded text layer for
> weak labels) **does not exist for handwriting** — there is no text layer. Handwritten HTR must
> rely on **synthetic fonts + teacher-VLM distillation + self-training** instead.

---

## Three concrete tiers for HipTR

Mapped to the repo's stages (`train.py --stage align|sft|encoder`).

### Tier 1 — Minimal proof-of-concept (one script, a few hands; prove the pipeline aligns)
- **Stage A:** ~100K–250K image–text pairs (can be document/line crops + transcriptions, not generic
  captions), projector-only, 1 epoch, LR ~1e-3.
- **Stage B:** ~0.5–2M synthetic rendered lines (TRDG-style, tens–hundreds of fonts) + connector+LLM
  via LoRA, vision frozen, LR ~2e-5.
- **Human-labeled:** ~300–1,000 lines of the target hand.
- **Expected:** line polygons + transcription in reading order; **~3–10% CER** on a clean/narrow
  hand; brittle to new hands. Proves alignment works.

### Tier 2 — Solid single-language HTR (production-quality one script)
- **Stage A:** full ~0.5–0.6M pairs (LLaVA-1.5-558K / MobileVLM-595K order), projector-only.
- **Stage B:** ~10–20M lines — synthetic handwritten (thousands of fonts; cf. TrOCR's 5,427 fonts →
  17.9M lines) + some printed/pseudo-labeled; connector+LLM, vision frozen; + self-training on
  unlabeled in-domain pages.
- **Human-labeled:** ~6K–15K lines (cf. IAM train 6,161 lines / 747 forms; full IAM 13,353 lines).
- **Expected:** **~2.9–4% CER** for that script (TrOCR-on-IAM class). Robust in-domain.

### Tier 3 — Strong multilingual (Surya-OCR-2-class)
- **Stage A:** ~0.5–1M+ pairs spanning scripts, projector-only.
- **Stage B:** tens–hundreds of millions of lines: large multilingual synthetic rendering + heavy
  pseudo-labeling/distillation (PDF text-layer for printed; teacher-VLM à la olmOCR for layout/HTML);
  connector+LLM, optional Stage C vision unfreeze.
- **Human-labeled:** ~5K–50K lines **per target script** for handwriting (printed can lean almost
  entirely on synthetic+pseudo-labels); prioritize low-resource scripts.
- **Expected:** ~83% olmOCR-bench / ~87% across ~91 languages **for printed**; handwritten
  multilingual trails this and needs more real labels per script.

---

## Bootstrapping playbook (highest leverage first)

1. **Synthetic rendering at scale** — free pixel-perfect labels. (TrOCR: 684M printed + 17.9M HW
   synthetic lines → only ~6K real IAM lines needed for 2.89% CER.) Render with handwriting fonts,
   degrade (ink bleed, paper texture, slant, blur, scanner noise).
2. **Distillation / pseudo-labeling from a teacher** — no human transcription. (olmOCR: ~250–266K
   GPT-4o-labeled pages.) For **printed** docs mine the PDF text layer; for **handwriting** use a
   strong teacher VLM to transcribe, then train the small model on its outputs.
3. **Self-training on unlabeled in-domain pages** — AT-ST: ~55% relative CER drop on handwriting from
   unlabeled target data, filtering pseudo-labels by confidence. (Multipliers are CTC-specific; treat
   as an upper bound for an autoregressive VLM.)
4. **Transfer from a related labeled hand** — IAM→Washington: ~3.3% CER with ~325–350 lines.
5. **Keep TIPSv2 + Qwen frozen in Stage A** — alignment buys vision→text mapping, not language
   modeling, so ~0.5M pairs suffice.

---

## Pitfalls (especially for handwriting)

- **Synthetic→real domain gap:** rendered fonts miss real degradation and cursive ligatures.
  Synthetic-only HW models look great on synthetic and collapse on real hands without a real
  fine-tune set + augmentation.
- **Don't assume the printed pseudo-label trick for handwriting** — there's no text layer to mine.
- **Don't treat Surya-OCR-2's volume as a known template** — it's undisclosed; the Common-Crawl
  pipeline is its *eval*, not training.
- **Don't skimp below the ~0.5M-pair alignment order or unfreeze the LLM/vision in Stage A** — it
  destabilizes the clean connector-only alignment.
- **Per-script scarcity hides in aggregate budgets** — low-resource scripts underperform even with
  synthetic data (cf. Surya Arabic 72.7% vs English 92.3%). Budget real labels *per script*.
- **Self-training can amplify errors** if the seed model is weak or confidence is miscalibrated.
- **Layout/reading-order is a separate problem** — line-recognition CER (TrOCR/IAM) says nothing
  about detection/ordering quality. The per-line **polygon + reading-order** output needs layout
  supervision (Surya uses a separate detector); pseudo-label or annotate it explicitly.
- **CER targets drift** — 2.89% (TrOCR, 2021) is no longer the ceiling (DTrOCR ~2.38%; frontier VLMs
  ~1.2–1.5% on IAM). Don't over-claim a tiny-data plan as SOTA.

---

## Suggested HipTR starting point

For a first **single-script** working model (Tier 1→2):

1. **Stage A:** ~0.3–0.5M printed/handwritten line crops + transcriptions (mix public OCR + TRDG
   synthetic), projector-only, 1 epoch, LR 1e-3.
2. **Stage B:** ~5–10M synthetic handwritten lines (a few thousand fonts) + your ALTO/PAGE-XML pages
   for the polygon+reading-order target; connector + LLM (LoRA), vision frozen; then self-train on
   your unlabeled scans.
3. **Real labels:** start with whatever you have (even ~1–6K lines), prioritize the polygon +
   reading-order supervision the synthetic set can't teach perfectly.

Datasets that export ALTO/PAGE-XML with line polygons: IAM, READ2016/ICDAR, Bentham, Norhand,
Transkribus / eScriptorium exports.

---

## Sources

- LLaVA-1.5 pretrain script & README — https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh ,
  https://github.com/haotian-liu/LLaVA ; paper https://arxiv.org/abs/2310.03744
- LLaVA v1.0 — https://arxiv.org/abs/2304.08485 ;
  https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K ,
  https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- MobileVLM — https://arxiv.org/abs/2312.16886 , https://github.com/Meituan-AutoML/MobileVLM
- Surya — https://github.com/datalab-to/surya ,
  https://raw.githubusercontent.com/datalab-to/surya/master/README.md ,
  https://datalab.to/blog/surya-2 , https://huggingface.co/datalab-to/surya-ocr-2
- TrOCR — https://arxiv.org/abs/2109.10282 , https://github.com/microsoft/unilm/blob/master/trocr/README.md
- Self-training (AT-ST) — https://arxiv.org/abs/2104.13037
- olmOCR — https://arxiv.org/abs/2502.18443 , https://huggingface.co/datasets/allenai/olmOCR-mix-0225
- DTrOCR — https://openaccess.thecvf.com/content/WACV2024/html/Fujitake_DTrOCR_Decoder-Only_Transformer_for_Optical_Character_Recognition_WACV_2024_paper.html
