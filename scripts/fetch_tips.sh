#!/usr/bin/env bash
# OPTIONAL: vendor the TIPS source + checkpoints for the offline / .npz path.
#
# HipTR loads TIPSv2 from HuggingFace by default (AutoModel google/tipsv2-l14-dpt,
# trust_remote_code=True) — no clone needed. This script is only for the *source*
# path (the foreground-seg Colab style): `from tips.pytorch import image_encoder`,
# build `vit_large(img_size=..., patch_size=14, ...)`, and load a .npz checkpoint.
#
# Note: some sandboxes restrict outbound git/storage to an allow-list. If a step
# is blocked, run it on a networked machine and copy third_party/tips over.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/tips"
CKPT_DIR="${ROOT}/third_party/checkpoints"
V2_BASE="https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch"
TOKENIZER_URL="https://storage.googleapis.com/tips_data/v1_0/checkpoints/tokenizer.model"

mkdir -p "${ROOT}/third_party" "${CKPT_DIR}"

if [ ! -d "${DEST}" ]; then
  echo "Cloning google-deepmind/tips into ${DEST} ..."
  git clone --depth 1 https://github.com/google-deepmind/tips.git "${DEST}"
else
  echo "TIPS already present at ${DEST}"
fi

echo
echo "Downloading TIPSv2 L/14 vision checkpoint (.npz) ..."
curl -fL "${V2_BASE}/tips_v2_oss_l14_vision.npz" -o "${CKPT_DIR}/tips_v2_oss_l14_vision.npz"
curl -fL "${TOKENIZER_URL}" -o "${CKPT_DIR}/tokenizer.model"

cat <<EOF

Vendored TIPS source: ${DEST}/pytorch   (add the repo root to PYTHONPATH for 'tips.*')
V2 checkpoints (.npz):
  L/14  -> ${V2_BASE}/tips_v2_oss_l14_vision.npz
  B/14  -> ${V2_BASE}/tips_v2_oss_b14_vision.npz
  So/14 -> ${V2_BASE}/tips_v2_oss_so14_vision.npz
  g/14  -> ${V2_BASE}/tips_v2_oss_g14_vision.npz

Source load (mirrors the TIPS Colab):
  from tips.pytorch import image_encoder
  m = image_encoder.vit_large(img_size=IMG, patch_size=14, ffn_layer='mlp',
        block_chunks=0, init_values=1.0, interpolate_antialias=True, interpolate_offset=0.0)
  import numpy as np, torch
  m.load_state_dict({k: torch.tensor(v) for k, v in np.load(CKPT).items()})
  feats = m.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]  # [B,C,H,W]

The default HF path needs none of this — just set HF_TIPSv2 / HF_TOKEN.
EOF
