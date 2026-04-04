"""
Generate completions for the COBOLEval benchmark (COBOL code generation).

This script loads a model via vLLM and generates COBOL code completions
for each task in the COBOLEval dataset. Outputs are saved in JSONL format
compatible with the COBOLEval evaluation pipeline.

Usage:
    python evaluation/generate_coboleval.py \
        --model_path <path_to_model> \
        --output_dir <output_directory> \
        --data_path <path_to_CobolEval.jsonl> \
        [--tensor_parallel_size 1] \
        [--gpu_memory_utilization 0.9] \
        [--max_tokens 4096]
"""

import argparse
import json
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import extract_code_block, re_structure_output, stream_jsonl


SYSTEM_PROMPT = (
    'Complete the given program. It should consist of a single markdown '
    'code block following on from the lines above until the end of the '
    'program. It should terminate with `GOBACK`.\n'
)


def read_problems(evalset_file: str):
    """Read COBOLEval problems from JSONL file."""
    data = []
    for task in stream_jsonl(evalset_file):
        data.append({
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "entry_point": task["entry_point"],
        })
    return data


def generate(args):
    """Run inference on COBOLEval benchmark."""
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

    # Read problems
    problems = read_problems(args.data_path)

    # Generate completions
    raw_outputs = []
    outputs = []

    for problem in tqdm(problems, desc="Generating COBOLEval completions"):
        prompt = SYSTEM_PROMPT + problem["prompt"]

        output = llm.generate([prompt], sampling_params)
        response = output[0].outputs[0].text

        raw_outputs.append(response)
        outputs.append(re_structure_output(problem["prompt"], response))

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))

    # Raw outputs (before post-processing)
    raw_path = os.path.join(args.output_dir, f"{model_name}_coboleval_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_outputs:
            f.write(json.dumps({"completion": entry}) + "\n")

    # Processed outputs (ready for evaluation)
    out_path = os.path.join(args.output_dir, f"{model_name}_coboleval.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in outputs:
            f.write(json.dumps({"completion": entry}) + "\n")

    # Also save as samples file for the evaluation script
    samples_path = os.path.join(args.output_dir, f"{model_name}_coboleval_samples.jsonl")
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


def main():
    parser = argparse.ArgumentParser(description="Generate COBOL completions for COBOLEval")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (local or HuggingFace)")
    parser.add_argument("--output_dir", type=str, default="./evaluation/output",
                        help="Directory to save generated outputs")
    parser.add_argument("--data_path", type=str, default="./evaluation/data/CobolEval.jsonl",
                        help="Path to CobolEval.jsonl benchmark file")
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
