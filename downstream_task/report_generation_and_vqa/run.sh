#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${1:-"${REPO_ROOT}/outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/report_generation/openi"}
MASTER_PORT=${MASTER_PORT:-34221}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
BERT_MODEL=${BERT_MODEL:-"/root/autodl-tmp/models/bert-base-uncased"}

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
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "${MASTER_PORT}" --use_env \
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
        --bert_model "${BERT_MODEL}" \
        --model_recover_path "${itr}"
done
