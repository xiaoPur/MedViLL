#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/MedViLL}"
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/IU x-ray}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOADDIR="${OPENI_LOADDIR:-$REPO_ROOT}"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_openi_server.py" \
  --repo-root "$REPO_ROOT" \
  --raw-root "$RAW_ROOT" \
  --overwrite

cd "$REPO_ROOT/downstream_task/classification"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
"$PYTHON_BIN" cls.py \
  --openi True \
  --data_path "$REPO_ROOT/data/openi" \
  --loaddir "$LOADDIR" \
  --save_name openi_server_single_gpu \
  --task_type multilabel \
  --batch_sz 4 \
  --max_epochs 5 \
  --n_workers 4 \
  --lr 0.0001
