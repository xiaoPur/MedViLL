# OpenI Server Workflow

This repo now has a small OpenI-only server workflow that works with raw IU X-ray data placed at:

`/root/autodl-tmp/IU x-ray`

and the repo checked out at:

`/root/autodl-tmp/MedViLL`

## What the new files do

`scripts/prepare_openi_server.py` reads `data/openi/Train.jsonl`, `Valid.jsonl`, and `Test.jsonl`, finds a matching frontal IU X-ray image for each study id, and creates:

`data/preprocessed/openi/{train,valid,test}/{id}.jpg`

It prefers PA/AP/frontal projections and falls back to other non-lateral images if needed. By default it symlinks; use `--copy` if you want physical files instead.

`scripts/run_openi_classification.sh` runs the prep step and then launches OpenI classification on a single GPU with conservative defaults.

## How to run

From the repo root:

```bash
bash scripts/run_openi_classification.sh
```

If you want to point at a different raw data location:

```bash
RAW_ROOT="/your/path/IU x-ray" bash scripts/run_openi_classification.sh
```

If you already have a checkpoint directory for classification, set:

```bash
OPENI_LOADDIR="/path/to/checkpoint_dir" bash scripts/run_openi_classification.sh
```

## What is still needed for other tasks

The new workflow only covers OpenI classification. Other MedViLL tasks still need extra files:

`Retrieval`

You still need the OpenI label-conditioned retrieval JSONL files mentioned in the original README:
`T2I_Label_Valid.jsonl`, `T2I_Label_Test.jsonl`, `I2T_Label_Valid.jsonl`, `I2T_Label_Test.jsonl`.

`Report generation`

You still need the report-generation training/eval assets expected by `downstream_task/report_generation_and_vqa`, plus the model checkpoint you want to fine-tune or decode from.

`VQA`

You still need the VQA-RAD dataset bundle and its cached answer files.

`Pretraining`

Pretraining is still a separate path and needs the larger pretraining dataset setup, which is not covered by these new files.
