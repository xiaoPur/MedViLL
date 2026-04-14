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
torchrun --standalone --nproc_per_node=1 --master_port 34221 \
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
  --model_recover_path /path/to/medvill_pretrain/pytorch_model.bin
```

说明：

- `--model_recover_path` 需要指向一个可用的预训练权重文件。
- 训练输出会写到脚本默认的输出目录体系里；如果你想统一管理，建议额外指定输出目录参数。
- 当前仓库已经兼容单卡 `torchrun`，推荐优先用这条命令。

### 解码与评测

训练完以后，用生成结果做评测：

```bash
cd /root/autodl-tmp/MedViLL
python downstream_task/report_generation_and_vqa/generation_decode.py \
  --model_recover_path /path/to/output/model.50.bin \
  --beam_size 1
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
- `bert-base-uncased` 的 tokenizer / 权重缓存
  - 如果服务器不能联网，这一步必须提前准备好
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
