<div align="center">

# COBOL-Coder: A COBOL-Specialized LLM for Code Generation and Translation

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2408.04660-red?style=flat&label=arXiv)](https://arxiv.org/abs/2408.04660)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Data Augmentation Pipeline](#data-augmentation-pipeline)
- [Model Download](#model-download)
- [Evaluation Results](#evaluation-results)
  - [COBOL Code Generation](#cobol-code-generation)
  - [COBOL-Java Translation](#cobol-java-translation)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [COBOL-JavaTrans Benchmark](#cobol-javatrans-benchmark)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Introduction

**COBOL-Coder** is a family of domain-adapted LLMs specialized for COBOL code generation and bidirectional COBOL-Java code translation. Built on top of [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct), COBOL-Coder addresses the critical gap in LLM capabilities for legacy programming languages.

Key highlights:

- **79.29% compilation success rate** and **47.50% Pass@1** on COBOLEval, compared to 41.8% and 16.4% for GPT-4o
- **34.93% Pass@1** on Java-to-COBOL translation, where general-purpose models achieve near-zero scores
- An **automated data augmentation pipeline** leveraging compiler feedback and LLM-based refinement
- **COBOL-JavaTrans**, the first benchmark for bidirectional COBOL-Java translation

## Data Augmentation Pipeline

We construct a COBOL-specific training corpus through three data sources:

| Data Source | Instruction Format | Token Count | Instances |
|---|---|---|---|
| GitHub Repositories | Description - COBOL Code | 38.4M | 31,492 |
| Synthetic COBOL Code | COBOL-Java pairs | 92M | 101,735 |
| Synthetic COBOL Code | Description - COBOL Code | 81M | 101,735 |
| Technical References | Question-Answer | 241M | 153,415 |

The pipeline includes:
1. **COBOL code from GitHub** - Cleaned, deduplicated, and self-debugged with compiler feedback (GnuCOBOL) over K=3 iterations
2. **Synthetic COBOL via code translation** - 300K Java programs from The Stack v2 translated to COBOL using GPT-4o, then validated through compilation
3. **COBOL and mainframe knowledge** - Licensed textbooks and documentation converted into 153K instruction-style QA pairs

## Model Download

| Model | Base Model | Parameters | Download |
|---|---|---|---|
| COBOL-Coder-7B | Qwen2.5-Coder-7B-Instruct | 7B | Coming soon |
| COBOL-Coder-14B | Qwen2.5-Coder-14B-Instruct | 14B | Coming soon |

## Evaluation Results

### COBOL Code Generation

Performance on COBOLEval and COBOLCodeBench benchmarks. **Bold** = best in block; **Bold + Underline** = overall best.

| Model | COBOLEval CSR | COBOLEval Pass@1 | COBOLCodeBench CSR | COBOLCodeBench Pass@1 |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 14.98 | 1.37 | 0 | 0 |
| CodeGemma 7B | 0 | 0 | 0 | 0 |
| CodeLlama 7B | 0 | 0 | 0 | 0 |
| Mainframer 7B | **69.17** | 6.16 | 0 | 0 |
| Qwen2.5-Coder 7B | 10.27 | 0.68 | 0 | 0 |
| **COBOL-Coder-7B (Ours)** | 65.53 | **31.42** | **13.04** | 0 |
| CodeLlama 13B | 3.40 | 0.68 | 0 | 0 |
| Mainframer 13B | 62.24 | 11.64 | 0 | 0 |
| Qwen2.5-Coder 14B | 12.32 | 2.74 | 0 | 0 |
| **COBOL-Coder-14B (Ours)** | **_79.29_** | **_47.50_** | **_26.09_** | **_4.35_** |
| GPT-oss-120B | 19.17 | 4.11 | 17.39 | 2.17 |
| GPT-4 | 24.12 | 15.75 | 13.04 | 0 |
| GPT-4o | 41.80 | 16.40 | 13.04 | 0 |

### COBOL-Java Translation

Performance on the COBOL-JavaTrans benchmark. C2J = COBOL-to-Java, J2C = Java-to-COBOL.

| Model | C2J CSR | C2J Pass@1 | J2C CSR | J2C Pass@1 |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 88.11 | 63.64 | 0 | 0 |
| CodeGemma 7B | 76.22 | 48.25 | 0 | 0 |
| Qwen2.5-Coder 7B | 14.68 | 10.47 | 0 | 0 |
| **COBOL-Coder-7B (Ours)** | **95.80** | **81.11** | **36.18** | **24.72** |
| Qwen2.5-Coder 14B | 8.39 | 3.50 | 0 | 0 |
| DeepSeekCoder-V2 16B | 95.10 | 75.52 | 0 | 0 |
| **COBOL-Coder-14B (Ours)** | **95.80** | **83.22** | **_45.97_** | **_34.93_** |
| GPT-oss-120B | **_98.60_** | **_89.51_** | 5.38 | 3.93 |
| GPT-4 | 94.40 | 72.73 | 5.45 | 1.73 |
| GPT-4o | 97.20 | 85.31 | 4.36 | 2.18 |

For full results including developer survey findings, please refer to our paper.

## Getting Started

### Installation

```bash
conda create -n cobol-coder python=3.10 && conda activate cobol-coder

git clone https://github.com/FSoft-AI4Code/COBOL-Coder.git
cd COBOL-Coder
pip install -r requirements.txt
```

For evaluation, you also need:
- [GnuCOBOL](https://gnucobol.sourceforge.io/) compiler (`cobc`) for COBOL compilation
- Java JDK (`javac`, `java`) for Java compilation (C2J evaluation)

### Training

Fine-tune Qwen2.5-Coder on COBOL-specific data using DeepSpeed ZeRO-3:

```bash
bash sft.sh
```

The training script uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the fine-tuning framework. Key hyperparameters:
- **Learning rate**: 5e-6 (7B) / 1e-5 (14B)
- **Batch size**: 2,048 (global)
- **Max sequence length**: 4,096
- **Optimizer**: AdamW with cosine schedule
- **Precision**: BF16 with Flash Attention 2

### Inference

Generate COBOL code completions for the COBOLEval benchmark:

```bash
python evaluation/generate_coboleval.py \
    --model_path path/to/COBOL-Coder-14B \
    --output_dir evaluation/output \
    --data_path evaluation/data/CobolEval.jsonl
```

Generate translations for COBOL-JavaTrans:

```bash
# COBOL-to-Java
python evaluation/generate_cobol_javatrans.py \
    --model_path path/to/COBOL-Coder-14B \
    --direction c2j \
    --output_dir evaluation/output

# Java-to-COBOL
python evaluation/generate_cobol_javatrans.py \
    --model_path path/to/COBOL-Coder-14B \
    --direction j2c \
    --output_dir evaluation/output
```

### Evaluation

Evaluate COBOL code generation (COBOLEval) and Java-to-COBOL translation (J2C):

```bash
python evaluation/evaluate_coboleval.py \
    --samples_file evaluation/output/<model>_coboleval_samples.jsonl \
    --data_path evaluation/data/CobolEval.jsonl
```

Evaluate COBOL-to-Java translation (C2J):

```bash
python evaluation/evaluate_translation_c2j.py \
    --samples_file evaluation/output/<model>_javatrans_c2j.jsonl \
    --data_path evaluation/data/COBOL-JavaTrans.jsonl
```

Example output files from COBOL-Coder are provided in `evaluation/output/`.

## COBOL-JavaTrans Benchmark

COBOL-JavaTrans is the first benchmark for bidirectional COBOL-Java code translation, derived from HumanEval. It contains 143 task pairs with both COBOL and Java implementations, along with test cases for both languages.

| Benchmark | Source Language | Task | Size |
|---|---|---|---|
| COBOLEval | Python | Code Generation | 146 problems |
| COBOLCodeBench | Python | Code Generation | 46 problems |
| COBOL-JavaTrans (Ours) | COBOL, Java | C2J, J2C Translation | 143 pairs |

The benchmark data is included in `evaluation/data/`.

## Repository Structure

```
COBOL-Coder/
├── src/                          # LLaMA-Factory source (training framework)
├── evaluation/
│   ├── data/                     # Benchmark datasets
│   │   ├── CobolEval.jsonl
│   │   └── COBOL-JavaTrans.jsonl
│   ├── output/                   # Example outputs from COBOL-Coder
│   ├── generate_coboleval.py     # Inference for COBOLEval
│   ├── generate_cobol_javatrans.py  # Inference for COBOL-JavaTrans
│   ├── evaluate_coboleval.py     # Evaluation for COBOL code (CSR + Pass@1)
│   ├── evaluate_translation_c2j.py  # Evaluation for C2J translation
│   └── utils.py                  # Shared utilities
├── examples/                     # DeepSpeed configs and training examples
├── scripts/                      # Utility scripts
├── sft.sh                        # Training launch script
├── requirements.txt              # Python dependencies
└── cobol_reserved_words.txt      # COBOL vocabulary for tokenizer
```

## Acknowledgements

- This codebase is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning
- COBOLEval benchmark by [BloopAI](https://bloop.ai/blog/evaluating-llms-on-cobol)
- Base model: [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) by the Qwen team

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{dau2025cobolcoder,
  title={Does Domain Specialization Matter? A Study on LLMs for COBOL Code Generation and Translation},
  author={Dau, Anh T. V. and Tan, Shin Hwei and Yang, Jinqiu and Bui, Nghi D. Q. and Nguyen, Anh Tuan},
  year={2025}
}
```

For our earlier work on mainframe modernization:

```bibtex
@article{dau2024xmainframe,
  title={XMainframe: A Large Language Model for Mainframe Modernization},
  author={Dau, Anh TV and Dao, Hieu Trung and Nguyen, Anh Tuan and Tran, Hieu Trung and Nguyen, Phong X and Bui, Nghi DQ},
  journal={arXiv preprint arXiv:2408.04660},
  year={2024}
}
```
