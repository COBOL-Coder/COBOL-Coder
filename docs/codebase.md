# COBOL-Coder Codebase Documentation

## Overview

COBOL-Coder is a COBOL-specialized LLM built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (v0.9.2) for fine-tuning and [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) as the base model. The project supports COBOL code generation and bidirectional COBOL-Java code translation.

## Repository Structure

```
COBOL-Coder/
├── src/                          # Training framework (LLaMA-Factory)
│   └── llamafactory/
│       ├── train/                # Training algorithms (SFT, DPO, etc.)
│       ├── model/                # Model loading and patching
│       ├── data/                 # Data processing, templates, tokenization
│       ├── hparams/              # Hyperparameter definitions
│       ├── api/                  # OpenAI-compatible API server
│       ├── chat/                 # Chat interface
│       ├── eval/                 # Built-in evaluation (MMLU, etc.)
│       ├── extras/               # Constants, utilities, logging
│       └── webui/                # Gradio web interface
│
├── evaluation/                   # COBOL-Coder evaluation pipeline
│   ├── data/
│   │   ├── CobolEval.jsonl       # COBOLEval benchmark (146 tasks)
│   │   └── COBOL-JavaTrans.jsonl # COBOL-JavaTrans benchmark (143 pairs)
│   ├── output/                   # Example model outputs
│   ├── generate_coboleval.py     # Inference: COBOL code generation
│   ├── generate_cobol_javatrans.py  # Inference: C2J and J2C translation
│   ├── evaluate_coboleval.py     # Evaluation: COBOL compile + test (CSR, Pass@1)
│   ├── evaluate_translation_c2j.py  # Evaluation: Java compile + test
│   └── utils.py                  # Shared utilities
│
├── examples/                     # Training config examples
│   ├── deepspeed/                # DeepSpeed ZeRO configs
│   ├── train_full/               # Full fine-tuning YAML configs
│   ├── train_lora/               # LoRA fine-tuning configs
│   └── ...
│
├── scripts/                      # Utility scripts
│   ├── vllm_infer.py             # vLLM batch inference
│   ├── api_example/              # API usage examples
│   ├── convert_ckpt/             # Checkpoint conversion
│   └── stat_utils/               # Training statistics (FLOPs, LR, PPL)
│
├── sft.sh                        # Training launch script
├── requirements.txt              # Python dependencies
├── cobol_reserved_words.txt      # COBOL vocabulary for tokenizer augmentation
├── setup.py                      # Package installation
├── pyproject.toml                # Build configuration
└── LICENSE
```

## Training

### Entry Point

Training is launched via `sft.sh`, which calls `src/train.py` through DeepSpeed:

```
sft.sh -> deepspeed src/train.py -> llamafactory.train.tuner.run_exp()
```

### Key Training Configuration

| Parameter | COBOL-Coder-7B | COBOL-Coder-14B |
|---|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct | Qwen2.5-Coder-14B-Instruct |
| Learning rate | 2e-5 | 1e-5 |
| Batch size (global) | 2,048 | 2,048 |
| Max sequence length | 4,096 | 4,096 |
| Optimizer | AdamW (cosine schedule) | AdamW (cosine schedule) |
| Precision | BF16 + Flash Attention 2 | BF16 + Flash Attention 2 |
| Parallelism | DeepSpeed ZeRO-3 | DeepSpeed ZeRO-3 |

### COBOL Tokenizer Augmentation

`src/llamafactory/model/loader.py` loads COBOL reserved words from `cobol_reserved_words.txt` and identifies tokens not in the base tokenizer vocabulary. This supports better tokenization of COBOL-specific keywords.

## Evaluation Pipeline

### Benchmarks

| Benchmark | File | Task | Evaluation |
|---|---|---|---|
| COBOLEval | `CobolEval.jsonl` | COBOL code generation | GnuCOBOL compile + test |
| COBOL-JavaTrans (C2J) | `COBOL-JavaTrans.jsonl` | COBOL-to-Java translation | javac compile + test |
| COBOL-JavaTrans (J2C) | `COBOL-JavaTrans.jsonl` | Java-to-COBOL translation | GnuCOBOL compile + test |

### Inference Scripts

**`generate_coboleval.py`** - Generates COBOL completions given function signatures and docstrings from COBOLEval. Uses vLLM for efficient batch inference.

**`generate_cobol_javatrans.py`** - Generates translations in either direction (`--direction c2j` or `--direction j2c`). Constructs prompts with source code and a target-language template.

### Evaluation Scripts

**`evaluate_coboleval.py`** - Compiles generated COBOL with `cobc -w -fformat=variable -x`, runs against test cases, parses output, and computes CSR and Pass@1. Also used for J2C evaluation (same COBOL compilation pipeline).

**`evaluate_translation_c2j.py`** - Compiles generated Java with `javac`, runs test classes, and computes CSR and Pass@1.

### Metrics

- **CSR (Compilation Success Rate)**: Proportion of generated programs that compile without errors
- **Pass@1**: Proportion of tasks where the first generated solution passes all test cases

## Dependencies

Core stack: PyTorch, Transformers, DeepSpeed, PEFT, TRL, vLLM, Flash Attention 2.

See `requirements.txt` for the full list.
