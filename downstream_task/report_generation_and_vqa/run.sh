#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${1:-"${REPO_ROOT}/outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/report_generation/openi"}
MASTER_PORT=${MASTER_PORT:-34221}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}

cd "${REPO_ROOT}"

if [ -f "${MODEL_ROOT}" ]; then
    MODEL_FILES=("${MODEL_ROOT}")
else
    mapfile -t MODEL_FILES < <(find "${MODEL_ROOT}" -name pytorch_model.bin)
fi

for itr in "${MODEL_FILES[@]}";
do
    echo ""
    echo "${itr}"
    torchrun --standalone --nproc_per_node=1 --master_port "${MASTER_PORT}" \
        "${SCRIPT_DIR}/finetune.py" \
        --repo_root "${REPO_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 50 \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --tasks report_generation \
        --generation_dataset openi \
        --mask_prob 0.15 \
        --s2s_prob 1 \
        --bi_prob 0 \
        --model_recover_path "${itr}"
done
