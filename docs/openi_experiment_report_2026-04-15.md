# OpenI Report Generation Experiment Report

## 1. Overview

This report summarizes an end-to-end reproduction of MedViLL for OpenI/IU X-ray report generation on a single-GPU server, including data preparation, training, decoding, observed results, and comparison with related papers.

The practical experiment flow was:

`IU X-ray raw data -> prepare_openi_server.py -> finetune.py -> generation_decode.py`

## 2. Environment

- Repository: `MedViLL`
- Python: `3.8`
- PyTorch: `1.7.0`
- CUDA: `11.0`
- GPU: `1 x 48 GB`
- CPU: `12 vCPU`
- RAM: `90 GB`

The environment is broadly aligned with the repo environment file, which pins `python=3.8.5`, `pytorch=1.7.0`, and `torchvision=0.8.1`. See [medvill.yaml](../medvill.yaml:36).

## 3. Dataset and Preparation

### 3.1 Raw data

The raw OpenI/IU X-ray data root contained:

- `images_normalized/`
- `indiana_projections.csv`
- `indiana_reports.csv`

This is also the expected minimum raw dataset structure in the project workflow. See [OPENI_SERVER_WORKFLOW.md](../OPENI_SERVER_WORKFLOW.md:19).

### 3.2 Split files used by the project

The repo already provides split files:

- `data/openi/Train.jsonl`
- `data/openi/Valid.jsonl`
- `data/openi/Test.jsonl`

Observed sample counts:

- Train: `2483`
- Valid: `710`
- Test: `354`
- Total: `3547`

This total matches the repo README description of the OpenI dataset as `3,547 AP and PA image-report pairs`. See [README.md](../README.md:34).

### 3.3 Image preparation

The raw IU X-ray images were converted into the repo-expected structure with:

```bash
python scripts/prepare_openi_server.py \
  --repo-root /root/autodl-tmp/MedViLL \
  --raw-root "/root/autodl-tmp/IU x-ray" \
  --overwrite
```

This script prepares:

- `data/preprocessed/openi/train/*.jpg`
- `data/preprocessed/openi/valid/*.jpg`
- `data/preprocessed/openi/test/*.jpg`

and writes a manifest file:

- `scripts/openi_prep_manifest.json`

See [prepare_openi_server.py](../scripts/prepare_openi_server.py:445) and [OPENI_SERVER_WORKFLOW.md](../OPENI_SERVER_WORKFLOW.md:51).

## 4. Model and Configuration

### 4.1 Base model

The experiment fine-tuned MedViLL from the official pretrained checkpoint:

- `/root/autodl-tmp/checkpoints/medvill/pytorch_model.bin`

The repo README describes MedViLL as a BERT-base vision-language model. See [README.md](../README.md:16).

### 4.2 Text backbone

The tokenizer/model directory used during training and decoding was:

- `/root/autodl-tmp/models/bert-base-uncased`

This local path was used explicitly to avoid slow online download of old Hugging Face S3 assets.

### 4.3 Visual backbone

The image encoder used by the MedViLL report-generation stack is based on pretrained `ResNet-50`. See [models/image.py](../models/image.py:11) and [downstream_task/report_generation_and_vqa/pytorch_pretrained_bert/model.py](../downstream_task/report_generation_and_vqa/pytorch_pretrained_bert/model.py:40).

### 4.4 Training configuration

The OpenI report-generation setup used the following core parameters:

- `--tasks report_generation`
- `--generation_dataset openi`
- `--num_train_epochs 50`
- `--train_batch_size 8`
- `--mask_prob 0.15`
- `--s2s_prob 1`
- `--bi_prob 0`
- `--bert_model /root/autodl-tmp/models/bert-base-uncased`
- `--model_recover_path /root/autodl-tmp/checkpoints/medvill/pytorch_model.bin`

See [OPENI_SERVER_WORKFLOW.md](../OPENI_SERVER_WORKFLOW.md:87).

### 4.5 Image resize compatibility fix

During reproduction, the original OpenI training pipeline exposed a batch collation issue because different raw X-ray images had different shapes. A compatibility fix was applied so that visual inputs are resized consistently:

- `224 x 224` when `len_vis_input < 100`
- `512 x 512` otherwise

See [image_preprocess.py](../downstream_task/report_generation_and_vqa/image_preprocess.py:1) and [data_loader.py](../downstream_task/report_generation_and_vqa/data_loader.py:438).

## 5. Training and Decoding

### 5.1 Training

Training was run with the PyTorch 1.7-compatible launcher:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 34221 --use_env \
downstream_task/report_generation_and_vqa/finetune.py \
  --repo_root /root/autodl-tmp/MedViLL \
  --output_dir /root/autodl-tmp/MedViLL/outputs/report_generation/openi \
  --num_train_epochs 50 \
  --train_batch_size 8 \
  --tasks report_generation \
  --generation_dataset openi \
  --mask_prob 0.15 \
  --s2s_prob 1 \
  --bi_prob 0 \
  --bert_model /root/autodl-tmp/models/bert-base-uncased \
  --model_recover_path /root/autodl-tmp/checkpoints/medvill/pytorch_model.bin
```

Training produced checkpoints up to `model.50.bin`.

### 5.2 Decoding

Decoding was run on the OpenI test split with:

```bash
python downstream_task/report_generation_and_vqa/generation_decode.py \
  --repo_root /root/autodl-tmp/MedViLL \
  --generation_dataset openi \
  --bert_model /root/autodl-tmp/models/bert-base-uncased \
  --model_recover_path /root/autodl-tmp/MedViLL/outputs/report_generation/openi/model.50.bin \
  --beam_size 1
```

Note that `generation_decode.py` defaults to `--random_bootstrap_testnum 2`, so the evaluation ran twice. See [generation_decode.py](../downstream_task/report_generation_and_vqa/generation_decode.py:144).

## 6. Experimental Results

### 6.1 Two observed decoding runs

Checkpoint evaluated:

- `model.50.bin`

Run 1:

- Decoding batches: `354`
- Time: `1335.08 s` (`22 min 15 s`)
- PPL: `4.7309771119538`
- BLEU-1: `0.2302211824755423`
- BLEU-2: `0.1575555166398683`
- BLEU-3: `0.11258218855061926`
- BLEU-4: `0.07878960773161373`

Run 2:

- Decoding batches: `354`
- Time: `1318.02 s` (`21 min 58 s`)
- PPL: `4.786202274473373`
- BLEU-1: `0.22754619681993984`
- BLEU-2: `0.15690903803771944`
- BLEU-3: `0.11077421107550464`
- BLEU-4: `0.07582904839067035`

### 6.2 Mean across the two runs

- PPL: `4.7586`
- BLEU-1: `0.2289`
- BLEU-2: `0.1572`
- BLEU-3: `0.1117`
- BLEU-4: `0.0773`

### 6.3 Missing CE metrics

The decoding log showed:

`Falling back to local BLEU evaluation because bleu.py could not be imported: No module named 'chexpert_labeler.loader'`

Therefore:

- `accuracy = NaN`
- `precision = NaN`
- `recall = NaN`
- `f1 = NaN`

This means the current reproduction successfully produced `BLEU` and `PPL`, but did not yet reproduce the clinical efficacy evaluation stack.

## 7. Comparison with MedViLL and the Context-Enhanced Paper

### 7.1 MedViLL official Open-I level

The local repo documents do not contain a direct official OpenI result table for MedViLL. However, a recent review article summarizing the MedViLL paper reports the following Open-I metrics for MedViLL:

- BLEU-4: `0.049`
- PPL: `5.637`
- Accuracy: `0.734`
- Precision: `0.512`
- Recall: `0.594`
- F1: `0.550`

Source:

- [Vision-language models for medical report generation and visual question answering: a review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11611889/)
- [Frontiers review mirror](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1430984/full)

Under that comparison, the current reproduction result:

- BLEU-4 `0.0773`
- PPL `4.7586`

is not weaker than the MedViLL Open-I number reported by that review, at least on these two metrics.

### 7.2 Confirmation for the user-mentioned paper

The paper [Context-enhanced framework for medical image reportgeneration using multimodal .pdf](</d:/Project/MedViLL/MedViLL/Context-enhanced framework for medical image reportgeneration using multimodal .pdf>) indeed reports much higher IU-Xray BLEU-4 than MedViLL.

From the extracted Table 1 in that PDF:

- Dataset: `IU-Xray`
- Method: `Ours`
- BLEU-1: `0.491`
- BLEU-2: `0.359`
- BLEU-3: `0.263`
- BLEU-4: `0.209`
- METEOR: `0.212`
- ROUGE-L: `0.408`
- CIDEr: `0.396`

The same PDF also contains Table 4, which reports a specific quantitative comparison for the IU-Xray dataset under different cross-modal context fusion settings:

- `None`: BLEU-4 `0.175`
- `Add`: BLEU-4 `0.181`
- `Concat`: BLEU-4 `0.179`
- `Attention`: BLEU-4 `0.197`
- `Ours`: BLEU-4 `0.223`

So the user observation is correct: that paper reports IU-Xray BLEU-4 around `0.209` in the main comparison table and up to `0.223` in a focused quantitative table.

### 7.3 Why that paper is not directly comparable to this MedViLL reproduction

The higher BLEU-4 in the context-enhanced paper does not mean the current MedViLL reproduction is wrong. The two systems are not the same setup.

Key differences visible in that paper:

- It uses additional multimodal context beyond a plain image-report model.
- The ablation tables explicitly reference:
  - clinical text context `T`
  - medical knowledge embedding `R`
  - diagnostic prompts `D`
- The model description includes cross-modal context enhancement, knowledge embedding enhancement, and later-stage prompt-based optimization.

In other words, that paper is a later and stronger report-generation method with extra contextual signals, so its IU-Xray BLEU-4 being much higher than MedViLL is expected.

## 8. Evaluation

### 8.1 What went well

- The entire OpenI/IU X-ray reproduction pipeline was run end-to-end.
- Data preparation, training, checkpoint saving, and decoding all worked.
- The experiment produced stable BLEU and PPL values over two evaluation runs.
- The observed BLEU-4 and PPL are competitive with, and likely better than, the Open-I MedViLL level cited by the review article.

### 8.2 Current limitations

- The current evaluation still lacks the CheXpert-based clinical efficacy metrics.
- The experiment used the MedViLL architecture and workflow, not the stronger context-enhanced framework from the newer paper.
- Therefore, lower BLEU-4 than `0.209~0.223` is expected and does not indicate a failed reproduction.

## 9. Final Conclusion

This reproduction can be considered successful for the main MedViLL OpenI report-generation pipeline:

- dataset preparation succeeded
- fine-tuning succeeded
- final checkpoint decoding succeeded
- BLEU and PPL were produced successfully

The most important interpretation point is:

- If the benchmark is the MedViLL Open-I level, the result is reasonable and likely competitive.
- If the benchmark is the newer context-enhanced framework paper, then the current BLEU-4 is lower, but this is an architecture gap rather than a reproduction failure.

For presentation, the safest conclusion is:

`The MedViLL OpenI report-generation pipeline was successfully reproduced. The current result is meaningful and stable, but it should be compared with MedViLL-style baselines rather than directly judged against newer context-enhanced multimodal report-generation methods.`
