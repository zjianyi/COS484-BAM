#!/usr/bin/env bash
# Pull few-shot candidate-logit baseline eval from Neuronic into this repo (metrics/ is gitignored).
#
# Neuronic repo path (see docs/NEURONIC_RUNBOOK.md): /u/bw7520/COS484-BAM — not ~/COS484-BAM unless that symlink exists.
#
# Usage on your laptop:
#   bash scripts/rsync_fewshot_next_token_baseline_from_neuronic.sh
#
# Override if needed:
#   REMOTE_HOST=bw7520@neuronic.cs.princeton.edu REMOTE_REPO=/u/bw7520/COS484-BAM bash scripts/rsync_fewshot_next_token_baseline_from_neuronic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/metrics/fewshot_babilong"

REMOTE_HOST="${REMOTE_HOST:-bw7520@neuronic.cs.princeton.edu}"
REMOTE_REPO="${REMOTE_REPO:-/u/bw7520/COS484-BAM}"

mkdir -p "${DEST}"

REMOTE_BASE="${REMOTE_HOST}:${REMOTE_REPO}/metrics/fewshot_babilong"

echo "rsync from ${REMOTE_BASE}"
rsync -avz \
  "${REMOTE_BASE}/no_cache_baseline_next_token_qa1_qa3.json" \
  "${REMOTE_BASE}/no_cache_baseline_next_token_qa1_qa3.cells.jsonl" \
  "${DEST}/"

echo "Done → ${DEST}/"
