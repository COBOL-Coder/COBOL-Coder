"""
Evaluate functional correctness of generated COBOL programs on COBOLEval.

This script compiles each generated COBOL program using GnuCOBOL (cobc),
runs it against the test cases from the benchmark, and computes:
  - Compilation Success Rate (CSR)
  - Pass@1 (functional correctness)

Adapted from: https://github.com/openai/human-eval

Prerequisites:
  - GnuCOBOL compiler (cobc) must be installed and available in PATH
  - Generated samples file in JSONL format with fields: task_id, completion

Usage:
    python evaluation/evaluate_coboleval.py \
        --samples_file <path_to_samples.jsonl> \
        --data_path <path_to_CobolEval.jsonl> \
        [--k 1]
"""

import argparse
import itertools
import math
import os
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from utils import cleanup_file, clean_response_for_eval, cmd, stream_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Parsing COBOL output values
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def parse(result, result_type, true):
    """Parse COBOL output value according to the expected Python type."""
    try:
        match result_type:
            case "Bool":
                return _parse_bool(result[0])
            case "Int":
                return _parse_int(result[0])
            case "Float":
                return _parse_float(result[0])
            case "String":
                return _parse_string(result[0])
            case {"List": "Int"}:
                return [_parse_int(x) for x in result][: len(true)]
            case {"List": "Float"}:
                return [_parse_float(x) for x in result][: len(true)]
            case {"List": "String"}:
                return [_parse_string(x) for x in result][: len(true)]
            case _:
                raise ParseError(f"Invalid result type: {result_type}")
    except Exception as e:
        raise ParseError(f"Result {result} of type {result_type} failed: {e}")


def _parse_bool(res: str) -> bool:
    return res.strip() == "1"


def _parse_int(res: str) -> int:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        return -int(res[1:])
    return int(res)


def _parse_float(res: str) -> float:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        return -float(res[1:])
    return float(res)


def _parse_string(res: str) -> str:
    return res.strip()


def is_equal(result_type, result, true):
    """Compare parsed result with expected value."""
    match result_type:
        case "Float":
            return math.isclose(result, true, abs_tol=0.001)
        case {"List": "Float"}:
            return all(math.isclose(r, t, abs_tol=0.001) for r, t in zip(result, true))
        case _:
            return result == true


# ---------------------------------------------------------------------------
# Compile and execute
# ---------------------------------------------------------------------------

def compile_and_run(name: str, path: str, call_path: str) -> bool:
    """Compile a COBOL program with its test caller and execute it."""
    success, _ = cmd(f"cobc -w -fformat=variable -x {call_path} {path}")
    if not success:
        return False

    success, _ = cmd(f"./call_{name}")
    if not success:
        return False

    return True


def check_correctness(problem: Dict, completion: str, work_dir: str) -> Dict:
    """Check the correctness of a single completion against all test cases."""
    name = problem["entry_point"]
    tests = problem["tests"]

    solutions_dir = os.path.join(work_dir, "solutions")
    callers_dir = os.path.join(work_dir, "callers")
    os.makedirs(solutions_dir, exist_ok=True)
    os.makedirs(callers_dir, exist_ok=True)

    path = os.path.join(solutions_dir, f"{name}.cbl")
    call_path = os.path.join(callers_dir, f"call_{name}.cbl")
    result_path = f"{name.upper().replace('_', '-')}.TXT"

    try:
        with open(path, "w") as f:
            f.write(completion)
    except Exception:
        with open(path, "w") as f:
            f.write("")

    passed, trues, results, compiled = [], [], [], []

    for test in tests:
        true = eval(test["result"]["value"])
        if isinstance(true, tuple):
            true = list(true)

        trues.append(true)
        passed.append(False)
        results.append(None)
        compiled.append(False)

        with open(call_path, "w") as f:
            f.write(test["test"])

        try:
            if compile_and_run(name, path, call_path):
                compiled[-1] = True
                with open(result_path) as f:
                    result = f.readlines()

                if result:
                    type_ = test["result"]["type_"]
                    parsed_result = parse(result, type_, true)
                    passed[-1] = is_equal(type_, parsed_result, true)
                    results[-1] = parsed_result
        except Exception:
            pass
        finally:
            cleanup_file(f"call_{name}")
            cleanup_file(result_path)

    return {
        "all_passed": all(passed),
        "passed": passed,
        "results": results,
        "trues": trues,
        "compiled": compiled,
    }


# ---------------------------------------------------------------------------
# Pass@k estimation
# ---------------------------------------------------------------------------

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def read_problems(evalset_file: str) -> Dict[str, Dict]:
    """Read benchmark problems as a dict keyed by task_id."""
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def evaluate(args):
    """Evaluate functional correctness of generated COBOL samples."""
    problems = read_problems(args.data_path)
    task_ids = [task["task_id"] for task in stream_jsonl(args.data_path)]

    # Create working directory for compilation artifacts
    work_dir = os.path.join(os.path.dirname(args.samples_file), "eval_workdir")
    os.makedirs(work_dir, exist_ok=True)

    n_samples = 0
    results = defaultdict(list)

    # Store samples alongside results for later saving
    samples_with_results = []

    print(f"Reading samples from: {args.samples_file}")
    for idx, sample in enumerate(tqdm(list(stream_jsonl(args.samples_file)), desc="Evaluating")):
        task_id = sample.get("task_id", task_ids[idx])
        completion = clean_response_for_eval(sample["completion"])

        correct = check_correctness(problems[task_id], completion, work_dir)

        n_samples += 1
        results[task_id].append((0, correct))
        samples_with_results.append({**sample, **correct})

    # Compute metrics
    total_arr, correct_arr = [], []
    for result in results.values():
        result.sort()
        passed_list = [r[1]["all_passed"] for r in result]
        total_arr.append(len(passed_list))
        correct_arr.append(sum(passed_list))

    total_arr = np.array(total_arr)
    correct_arr = np.array(correct_arr)

    ks = [int(k) for k in args.k.split(",")]
    pass_at_k = {
        f"pass@{k}": float(estimate_pass_at_k(total_arr, correct_arr, k).mean())
        for k in ks
        if (total_arr >= k).all()
    }

    # Compute CSR and detailed stats
    total_tests = 0
    total_passed = 0
    total_compiled = 0
    total_tasks_passed = 0
    total_tasks_compiled = 0

    for result in results.values():
        for r in result:
            total_tests += len(r[1]["passed"])
            total_passed += sum(r[1]["passed"])
            total_compiled += sum(r[1]["compiled"])
            total_tasks_compiled += int(sum(r[1]["compiled"]) == len(r[1]["compiled"]))
            total_tasks_passed += int(len(r[1]["passed"]) == sum(r[1]["passed"]))

    n_tasks = len(results)
    csr = total_tasks_compiled / n_tasks * 100 if n_tasks > 0 else 0.0

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total tasks:           {n_tasks}")
    print(f"Total tests:           {total_tests}")
    print(f"Compiled (tests):      {total_compiled}/{total_tests}")
    print(f"CSR (tasks):           {total_tasks_compiled}/{n_tasks} ({csr:.2f}%)")
    print(f"Pass@1 (tasks):        {total_tasks_passed}/{n_tasks} ({total_tasks_passed / n_tasks * 100:.2f}%)")
    for k_name, v in pass_at_k.items():
        print(f"{k_name} (estimated):   {v * 100:.2f}%")
    print("=" * 60)

    # Save detailed results (reuse already-computed results)
    out_file = args.samples_file.replace(".jsonl", "_results.jsonl")
    write_jsonl(out_file, tqdm(iter(samples_with_results), total=n_samples, desc="Writing results"))
    print(f"\nDetailed results saved to: {out_file}")

    return pass_at_k


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COBOL code generation on COBOLEval benchmark"
    )
    parser.add_argument("--samples_file", type=str, required=True,
                        help="Path to generated samples JSONL file")
    parser.add_argument("--data_path", type=str, default="./evaluation/data/CobolEval.jsonl",
                        help="Path to CobolEval.jsonl benchmark file")
    parser.add_argument("--k", type=str, default="1",
                        help="Comma-separated list of k values for pass@k (default: 1)")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
