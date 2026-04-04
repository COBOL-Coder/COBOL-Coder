"""
Generate translations for the COBOL-JavaTrans benchmark.

Supports two translation directions:
  - COBOL-to-Java (c2j): Translate COBOL source code into Java
  - Java-to-COBOL (j2c): Translate Java source code into COBOL

Usage:
    # COBOL-to-Java translation
    python evaluation/generate_cobol_javatrans.py \
        --model_path <path_to_model> \
        --direction c2j \
        --output_dir <output_directory> \
        --data_path <path_to_COBOL-JavaTrans.jsonl>

    # Java-to-COBOL translation
    python evaluation/generate_cobol_javatrans.py \
        --model_path <path_to_model> \
        --direction j2c \
        --output_dir <output_directory> \
        --data_path <path_to_COBOL-JavaTrans.jsonl>
"""

import argparse
import json
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    extract_code_block,
    re_structure_output,
    remove_block_comments,
    remove_cobol_comments,
    stream_jsonl,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

C2J_PROMPT = """You are a Java developer. Your task is to translate the following COBOL program into Java.
**COBOL Program:**
{cobol_code}

**Instructions:**
Generate only valid Java code following the below template
{template}

The translated Java code:
"""

J2C_PROMPT = """You are a COBOL developer. Your task is to translate the following Java program into COBOL.
**Java Program:**
{java_code}


**Instructions:**
Generate only valid COBOL code following the below template
{template}

The translated COBOL code:
"""


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def read_problems_c2j(evalset_file: str):
    """Read problems for COBOL-to-Java translation."""
    data = []
    for task in stream_jsonl(evalset_file):
        data.append({
            "task_id": task["task_id"],
            "java_prompt": task["Java_prompt"],
            "cobol_code": task["COBOL_canonical_solution"],
        })
    return data


def read_problems_j2c(evalset_file: str):
    """Read problems for Java-to-COBOL translation."""
    data = []
    for task in stream_jsonl(evalset_file):
        data.append({
            "task_id": task["task_id"],
            "cobol_prompt": task["COBOL_prompt"],
            "java_code": task["Java_canonical_solution"],
        })
    return data


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(args):
    """Run inference on COBOL-JavaTrans benchmark."""
    # Load model
    llm = LLM(
        model=args.model_path,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    direction = args.direction
    model_name = os.path.basename(args.model_path.rstrip("/"))

    if direction == "c2j":
        problems = read_problems_c2j(args.data_path)
        raw_outputs, outputs = _generate_c2j(llm, sampling_params, problems)
    elif direction == "j2c":
        problems = read_problems_j2c(args.data_path)
        raw_outputs, outputs = _generate_j2c(llm, sampling_params, problems)
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'c2j' or 'j2c'.")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, f"{model_name}_javatrans_{direction}_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_outputs:
            f.write(json.dumps({"completion": entry}) + "\n")

    out_path = os.path.join(args.output_dir, f"{model_name}_javatrans_{direction}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in outputs:
            f.write(json.dumps({"completion": entry}) + "\n")

    samples_path = os.path.join(args.output_dir, f"{model_name}_javatrans_{direction}_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for i, (problem, completion) in enumerate(zip(problems, outputs)):
            sample = {
                "sample_id": 0,
                "task_id": problem["task_id"],
                "completion": completion,
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Raw outputs saved to: {raw_path}")
    print(f"Processed outputs saved to: {out_path}")
    print(f"Samples for evaluation saved to: {samples_path}")

    return out_path


def _generate_c2j(llm, sampling_params, problems):
    """Generate COBOL-to-Java translations."""
    raw_outputs = []
    outputs = []

    for problem in tqdm(problems, desc="Translating COBOL -> Java"):
        prompt = C2J_PROMPT.format(
            cobol_code=problem["cobol_code"],
            template=remove_block_comments(problem["java_prompt"]),
        )

        output = llm.generate([prompt], sampling_params)
        response = output[0].outputs[0].text

        raw_outputs.append(response)
        outputs.append(extract_code_block(response))

    return raw_outputs, outputs


def _generate_j2c(llm, sampling_params, problems):
    """Generate Java-to-COBOL translations."""
    raw_outputs = []
    outputs = []

    for problem in tqdm(problems, desc="Translating Java -> COBOL"):
        prompt = J2C_PROMPT.format(
            java_code=problem["java_code"],
            template=remove_cobol_comments(problem["cobol_prompt"]),
        )

        output = llm.generate([prompt], sampling_params)
        response = output[0].outputs[0].text

        raw_outputs.append(response)
        outputs.append(re_structure_output("", response))

    return raw_outputs, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate translations for COBOL-JavaTrans benchmark"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (local or HuggingFace)")
    parser.add_argument("--direction", type=str, required=True, choices=["c2j", "j2c"],
                        help="Translation direction: 'c2j' (COBOL->Java) or 'j2c' (Java->COBOL)")
    parser.add_argument("--output_dir", type=str, default="./evaluation/output",
                        help="Directory to save generated outputs")
    parser.add_argument("--data_path", type=str,
                        default="./evaluation/data/COBOL-JavaTrans.jsonl",
                        help="Path to COBOL-JavaTrans benchmark file")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    generate(args)


if __name__ == "__main__":
    main()
