# OpenI 报告生成下载清单

这个清单只面向 `OpenI` 报告生成任务。默认服务器路径是：

- 项目：`/root/autodl-tmp/MedViLL`
- 数据：`/root/autodl-tmp/IU x-ray`

## 必须下载

### 1. 原始 IU X-ray 数据

至少要有：

- `images_normalized`
- `indiana_projections.csv`
- `indiana_reports.csv`

如果你已经有压缩包，上传到服务器后先解压到：

```text
/root/autodl-tmp/IU x-ray
```

### 2. 预训练权重

报告生成不建议从零开始跑。你需要一个可用的 `MedViLL` 预训练 checkpoint，通常至少包括：

- `pytorch_model.bin`
- `config.json`

如果你已经有训练输出目录，也可以直接用里面的 `model.*.bin` 作为后续微调起点。

### 3. `bert-base-uncased`

训练和解码都会用到 `bert-base-uncased`。

推荐固定放到这个本地目录：

```text
/root/autodl-tmp/models/bert-base-uncased
```

里面至少准备：

```text
vocab.txt
config.json
tokenizer_config.json
tokenizer.json
```

如果服务器联网不稳定，建议提前把这些文件下载好再拷到服务器，并在训练和解码命令里显式传：

```text
--bert_model /root/autodl-tmp/models/bert-base-uncased
```

### 4. 训练输出空间

这不是单独的文件下载，但你需要在服务器上预留足够磁盘，至少放得下：

- 原始 `IU X-ray`
- 预处理后的 `data/preprocessed/openi`
- 训练 checkpoint
- 解码结果和日志

## 可选下载

### A. 只在你要看 CheXpert 类指标时需要

下载以下内容：

- `chexpert_labeler`
- `NegBio`
- `phrases/mention`
- `phrases/unmention`
- `patterns/*.txt`

这些资源用于标签类评估，不影响 `BLEU` / `ppl` 的主流程。

### B. 只在你要看 `ROUGE` / `CIDEr` 时需要

下载以下内容：

- `coco-caption` / `pycocoevalcap`
- 与胸片报告评测对应的 annotation 文件

当前仓库主流程没有默认打开这些指标，所以它们属于后补项。

## 服务器上最小可跑组合

如果你只想先验证“报告生成能不能跑”，最小组合是：

- `/root/autodl-tmp/MedViLL`
- `/root/autodl-tmp/IU x-ray`
- `MedViLL` 预训练权重
- `bert-base-uncased`

## 先后顺序建议

1. 上传 `IU X-ray` 原始数据到 `/root/autodl-tmp/IU x-ray`
2. 上传 `MedViLL` 预训练权重到服务器某个固定目录
3. 确保 `bert-base-uncased` 可用
4. 在 `/root/autodl-tmp/MedViLL` 执行数据准备脚本
5. 再启动报告生成训练
## Local BERT Recommendation

For the current OpenI workflow, prefer downloading `bert-base-uncased`
into this fixed local directory:

```text
/root/autodl-tmp/models/bert-base-uncased
```

Run both training and decoding with:

```text
--bert_model /root/autodl-tmp/models/bert-base-uncased
```
