#!/usr/bin/env bash
# Vendor the TIPS vision encoder into third_party/.
#
# HipTR uses the PyTorch TIPS implementation directly (it is NOT a HuggingFace
# AutoModel). This script clones github.com/google-deepmind/tips and, optionally,
# downloads the TIPSv2 vision checkpoint(s).
#
# Note: some sandboxes restrict outbound git to an allow-list. If the clone is
# blocked, run this on a machine with network access and copy third_party/tips over.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/tips"

mkdir -p "${ROOT}/third_party"

if [ ! -d "${DEST}" ]; then
  echo "Cloning google-deepmind/tips into ${DEST} ..."
  git clone --depth 1 https://github.com/google-deepmind/tips.git "${DEST}"
else
  echo "TIPS already present at ${DEST}"
fi

echo
echo "Vendored TIPS pytorch package: ${DEST}/pytorch"
echo "Set vision.tips_pkg_path = third_party/tips/pytorch in your config."
echo
echo "Download a vision checkpoint (TIPSv2 L/14) with the repo's helper, e.g.:"
echo "  cd ${DEST}/pytorch/checkpoints && chmod +x download_checkpoints.sh && ./download_checkpoints.sh"
echo "or fetch directly from storage.googleapis.com/tips_data/ (see the TIPS README),"
echo "then set vision.checkpoint_path to the .npz path."
