#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_ROOT=${1:-"${REPO_ROOT}/outputs/report_generation/openi"}
MASTER_PORT=${MASTER_PORT:-34222}
BERT_MODEL=${BERT_MODEL:-"/root/autodl-tmp/models/bert-base-uncased"}

cd "${REPO_ROOT}"

if [ -f "${MODEL_ROOT}" ]; then
    MODEL_FILES=("${MODEL_ROOT}")
else
    mapfile -t MODEL_FILES < <(find "${MODEL_ROOT}" -name "model.50.bin")
fi

for itr in "${MODEL_FILES[@]}";
do
    echo ""
    echo "${itr}"
    python "${SCRIPT_DIR}/generation_decode.py" \
        --repo_root "${REPO_ROOT}" \
        --generation_dataset openi \
        --bert_model "${BERT_MODEL}" \
        --model_recover_path "${itr}" \
        --beam_size 1
done
