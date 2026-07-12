# Meta-analysis Information Extraction with Qwen2.5-7B-Instruct

## 项目简介

本项目使用 **Qwen2.5-7B-Instruct** 从医疗可及性相关文献中抽取结构化信息，用于辅助荟萃分析。整体框架为：

```text
原始文献
→ 长文本分块
→ Baseline 推理
→ QLoRA 微调
→ 微调模型推理
→ 文献级结果聚合
→ 自动评测与消融实验
```

模型最终输出 10 个目标字段，包括文献标题、研究区域、数据年份、可及性测量方法、设施类型、需求人群、距离或时间计算方法、交通方式、出行时间范围和城市化率。
1. `Literature Title`
2. `Study Area & Country`
3. `Data Year`
4. `Accessibility Method`
5. `Facility Type`
6. `Demand Population`
7. `Dist/Time Calc Method`
8. `Transport Mode`
9. `Travel Time Period`
10. `Urbanization Rate`

## 1. 运行环境

原实验环境如下：

| 项目 | 配置 |
|---|---|
| Python | 3.10 |
| 基座模型 | `Qwen2.5-7B-Instruct` |
| 微调方法 | QLoRA |
| 量化方式 | 4-bit |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 显存 | 24 GB |

安装主要依赖：

```bash
pip install torch transformers datasets peft bitsandbytes accelerate sentencepiece safetensors numpy
```

## 2. 实验耗时

仓库日志中可以确认的耗时为：

| 实验 | 配置 | 耗时 |
|---|---|---:|
| Fold 1 QLoRA 微调 | 50 steps，batch size=1，gradient accumulation=8 | 约 5 分 16 秒 |

对应主要训练参数：

```text
LoRA rank: 2
LoRA alpha: 4
LoRA dropout: 0.2
learning rate: 5e-5
max steps: 50
max sequence length: 4096
```

## 3. 实验复现步骤

### 3.1 克隆仓库

```bash
https://github.com/RachelChang42/meta-analysis-finetune

```

### 3.2 创建环境

```bash
conda create -n llm-finetune python=3.10 -y
conda activate llm-finetune

pip install torch transformers datasets peft bitsandbytes accelerate sentencepiece safetensors numpy
```

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3.3 准备基座模型

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct   --local-dir model/Qwen2.5-7B-Instruct
```

### 3.4 准备数据

原数据为：

```text
data/
├── folds/
│   ├── train_fold_1.jsonl
│   ├── val_fold_1.jsonl
│   ├── ...
│   ├── train_fold_5.jsonl
│   └── val_fold_5.jsonl
└── chunk_train/
    ├── train_chunk_fold_1.jsonl
    ├── ...
    └── train_chunk_fold_5.jsonl
```

### 3.5 修改路径

部分脚本仍保留原服务器绝对路径。运行前检查：

```bash
grep -R "/root/autodl-tmp" -n scripts eval
```

需要修改的主要内容包括模型路径、数据路径、Prompt 路径、LoRA adapter 路径、输出路径和评测路径。

### 3.6 数据检查与分块

```bash
python scripts/check_data.py
python scripts/check_folds.py
python scripts/data_stats.py
python scripts/build_chunk_train.py
```

### 3.7 运行 Baseline

```bash
python scripts/extract_and_aggregate_llm.py
```

首次运行建议只处理一条样本，确认模型、数据和输出格式正确后，再运行完整验证集。

### 3.8 进行 QLoRA 微调

```bash
python scripts/train_qlora_v2.py 2>&1 | tee logs/train_qlora.log
```

运行前确认脚本中的：

```text
MODEL_PATH
TRAIN_FILE
OUTPUT_DIR
```

五折实验需要分别使用五个训练集进行训练，并将 adapter 保存到不同目录。

### 3.9 使用微调模型推理

```bash
python scripts/extract_and_aggregate_lora_v3.py
```

运行前将脚本中的 LoRA 路径改为当前 fold 对应的 adapter 路径。

### 3.10 自动评测

```bash
python eval/eval_script_finaluse.py
```

评测前确认真实标签文件和预测结果文件路径，并保证预测结果包含统一的 10 个字段。

### 3.11 消融实验

消融实验主要比较：

```text
聚合方法：
1. LLM-only
2. Consensus-only
3. Hybrid

参与聚合的 chunk 数量：
1. top-10
2. top-15
3. top-25
```

对应代码位于：

```text
scripts/ablation/
```
