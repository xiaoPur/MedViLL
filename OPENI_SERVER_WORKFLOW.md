# OpenI 报告生成服务器使用指南

本文只聚焦 `OpenI` 报告生成任务，目标环境固定为：

- 项目目录：`/root/autodl-tmp/MedViLL`
- 数据目录：`/root/autodl-tmp/IU x-ray`
- 计算资源：单卡 `RTX 5090 32GB`

如果你只想先跑通 `BLEU-1/2/3/4` 和 `ppl`，按本文准备即可。`ROUGE` / `CIDEr` 属于可选扩展，不是当前主流程默认输出。

## 你现在要准备什么

仓库里已经有 `data/openi/Train.jsonl`、`Valid.jsonl`、`Test.jsonl`，但它们引用的是仓库内部的预处理图片路径：

`data/preprocessed/openi/{train,valid,test}/*.jpg`

所以你在云服务器上还需要把原始 `IU X-ray` 数据整理成这个目录结构。

原始数据目录应至少包含：

- `images_normalized`
- `indiana_projections.csv`
- `indiana_reports.csv`

## Environment Check

Before training, activate the same conda env used to install `medvill.yaml`.
That environment should already contain runtime packages such as `boto3`.
If you start training from the base env instead, `pytorch_pretrained_bert/file_utils.py`
will usually fail with `ModuleNotFoundError: No module named 'boto3'`.

Recommended checks:

```bash
conda activate medvill
which python
python -c "import sys, boto3, torch; print(sys.executable); print('boto3', boto3.__version__); print('torch', torch.__version__)"
```

Expected result:

- `which python` points to `.../envs/medvill/bin/python`
- the Python check prints both `boto3` and `torch` versions without error

If `boto3` is still missing inside that env, install the pinned pair once:

```bash
pip install boto3==1.16.31 botocore==1.19.31
```

## 从原始 IU X-ray 到仓库可用数据

在服务器上执行：

```bash
cd /root/autodl-tmp/MedViLL
python scripts/prepare_openi_server.py \
  --repo-root /root/autodl-tmp/MedViLL \
  --raw-root "/root/autodl-tmp/IU x-ray" \
  --overwrite
```

这一步会生成：

- `data/preprocessed/openi/train/*.jpg`
- `data/preprocessed/openi/valid/*.jpg`
- `data/preprocessed/openi/test/*.jpg`

默认优先创建软链接；如果软链接失败，会自动回退为复制。脚本还会生成一个清单文件：

- `scripts/openi_prep_manifest.json`

如果你想先检查原始目录是否完整，只看这三项就够了：

```text
/root/autodl-tmp/IU x-ray/images_normalized
/root/autodl-tmp/IU x-ray/indiana_projections.csv
/root/autodl-tmp/IU x-ray/indiana_reports.csv
```

## 报告生成怎么跑

### 训练

推荐从仓库根目录启动，单卡先用较保守的 batch size：

```bash
cd /root/autodl-tmp/MedViLL
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

说明：

- `--model_recover_path` 需要指向一个可用的预训练权重文件。
- 训练输出会写到脚本默认的输出目录体系里；如果你想统一管理，建议额外指定输出目录参数。
- `medvill.yaml` 默认安装的是 `PyTorch 1.7.0`，这个环境通常没有 `torchrun`，先用 `python -m torch.distributed.launch --use_env` 最稳妥。
- 如果你的环境里 `torchrun --help` 可以正常执行，也可以把上一条命令替换成 `torchrun --standalone --nproc_per_node=1 --master_port 34221`。

### 解码与评测

训练完以后，用生成结果做评测：

注意：这里不要手动输入 shell 自动显示的续行提示符 `>`，只输入下面这些命令内容即可。

```bash
cd /root/autodl-tmp/MedViLL
python downstream_task/report_generation_and_vqa/generation_decode.py \
  --repo_root /root/autodl-tmp/MedViLL \
  --generation_dataset openi \
  --bert_model /root/autodl-tmp/models/bert-base-uncased \
  --model_recover_path /path/to/output/model.50.bin \
  --beam_size 1
```

如果你想直接单行复制，推荐用这一条：

```bash
cd /root/autodl-tmp/MedViLL && python downstream_task/report_generation_and_vqa/generation_decode.py --repo_root /root/autodl-tmp/MedViLL --generation_dataset openi --bert_model /root/autodl-tmp/models/bert-base-uncased --model_recover_path /path/to/output/model.50.bin --beam_size 1
```

当前主流程实际接上的指标是：

- `BLEU-1`
- `BLEU-2`
- `BLEU-3`
- `BLEU-4`
- `ppl`

`ROUGE` / `CIDEr` 目前不是这条主流程的默认输出。如果你一定要这两类指标，需要额外补评测栈和对应数据文件，见下面的下载清单。

## 上云前下载清单

### 必须下载

- `IU X-ray` 原始数据，至少包含 `images_normalized`、`indiana_projections.csv`、`indiana_reports.csv`
- 一个可用的 `MedViLL` 预训练权重目录，至少包含：
  - `pytorch_model.bin`
  - 同目录下的 `config.json`
  
  MedViLL 预训练权重在项目 README 的 Downloads -> Pre-trained weights 里，作者给的是 Google Drive 链接：README.md (line 14)。你现在要跑报告生成，先用主版本 MedViLL 就行：
  
  - MedViLL: https://drive.google.com/file/d/1shOQrOWbkIeUUsQN48fEP6wj0e266jOb/view?usp=sharing
  - 其他几个变体也在同一段里：[GitHub README](https://github.com/SuperSupermoon/MedViLL#1-downloads)
  
  建议你下载后解压到服务器这个位置：
  
  ```
  /root/autodl-tmp/checkpoints/medvill/ 
  ```
  
  理想状态下这个目录里至少有：
  
  ```
  /root/autodl-tmp/checkpoints/medvill/pytorch_model.bin /root/autodl-tmp/checkpoints/medvill/config.json 
  ```
  
  然后训练时用这个：
  
  ```
  --model_recover_path /root/autodl-tmp/checkpoints/medvill/pytorch_model.bin 
  ```
  
  如果下载包里没有 config.json，你可以先用仓库自带这个兜底：
  
  ```
  /root/autodl-tmp/MedViLL/downstream_task/report_generation_and_vqa/config.json 
  ```
  
  当前我改过的代码也会优先找 checkpoint 同目录下的 config.json，找不到才回退到仓库里的 config.json (line 1)。
  
  结合你现在这台服务器，我建议你这样放：
  
  - MedViLL 权重：/root/autodl-tmp/checkpoints/medvill/
  - bert-base-uncased：如果能联网就让它自动进 /root/.pytorch_pretrained_bert；如果你想完全可控，就手动放 /root/autodl-tmp/models/bert-base-uncased/
- `bert-base-uncased` 的 tokenizer / 权重缓存
  
  - 如果服务器不能联网，这一步必须提前准备好
  
  bert-base-uncased 建议从 Hugging Face 官方页拿：
  
  - 模型页: https://huggingface.co/google-bert/bert-base-uncased
  
  这个仓库的报告生成代码用的是老版 pytorch_pretrained_bert，它默认缓存目录不是 Hugging Face 常见的 ~/.cache/huggingface，而是：
  
  - /root/.pytorch_pretrained_bert
  
  这是代码里写死的默认值：file_utils.py (line 26)。
  
  所以你有两个选法：
  
  1. 最省事
     让服务器联网，第一次跑时自动下载到：
  
  ```
  /root/.pytorch_pretrained_bert 
  ```
  
  1. 更稳妥
     手动提前下载到一个固定目录，比如：
  
  ```
  /root/autodl-tmp/models/bert-base-uncased/ 
  ```
  
  里面至少放：
  
  ```
  vocab.txt config.json tokenizer_config.json tokenizer.json 
  ```
  
  如果你走这条路，最好后面运行时显式传：
  
  ```
  --bert_model /root/autodl-tmp/models/bert-base-uncased 
  ```
  
- 如果你打算训练后立即评测，确保训练产生的输出目录和 checkpoint 能保留在服务器磁盘上

### 可选下载

#### CheXpert / NegBio

仅当你还想看报告生成里的标签类指标时才需要：

- `chexpert_labeler` 相关代码和资源
- `NegBio` 及其依赖文件
- `phrases/mention`
- `phrases/unmention`
- `patterns/*.txt`

这部分当前仓库没有完整带齐，不影响你先跑 `BLEU` 和 `ppl`。

#### ROUGE / CIDEr

仅当你坚持要算 `ROUGE` / `CIDEr` 时才需要：

- `coco-caption` 或兼容的 `pycocoevalcap` 评测包
- 对应的胸片报告 annotation 文件

这部分不是当前主流程默认接线，属于额外扩展。

## 推荐先后顺序

1. 先确认服务器上有 `/root/autodl-tmp/MedViLL` 和 `/root/autodl-tmp/IU x-ray`。
2. 先跑 `scripts/prepare_openi_server.py` 生成 `data/preprocessed/openi/...`。
3. 先只准备 `MedViLL` 预训练权重和 `bert-base-uncased`。
4. 先跑一次训练，看 `BLEU` 和 `ppl` 是否正常输出。
5. 如果主链路稳定，再决定要不要补 `CheXpert` 或 `ROUGE/CIDEr`。

更完整的下载清单见：[OpenI 报告生成下载清单](docs/openi_report_generation_checklist.md)
