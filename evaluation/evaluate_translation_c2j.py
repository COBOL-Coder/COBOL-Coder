"""
Evaluate COBOL-to-Java translations on COBOL-JavaTrans benchmark.

This script compiles each generated Java program using javac,
runs it against the Java test cases from the benchmark, and computes:
  - Compilation Success Rate (CSR)
  - Pass@1 (functional correctness)

Prerequisites:
  - Java JDK (javac, java) must be installed and available in PATH

Usage:
    python evaluation/evaluate_translation_c2j.py \
        --samples_file <path_to_c2j_samples.jsonl> \
        --data_path <path_to_COBOL-JavaTrans.jsonl>
"""

import argparse
import json
import signal
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from utils import clean_java_response, stream_jsonl


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, _timeout_handler)


def run_with_timeout(func, *args, timeout=30, **kwargs):
    """Run a function with a timeout (Unix only)."""
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutException:
        return {"test_returncode": 1, "compile_returncode": 1}


# ---------------------------------------------------------------------------
# Java compilation and test execution
# ---------------------------------------------------------------------------

def run_java_test(solution_code: str, solution_class: str,
                  test_code: str, test_class: str):
    """
    Compile and run a Java solution against a test.

    Returns:
        dict with compile/run output, errors, exit codes
    """
    temp_dir = Path(tempfile.mkdtemp())

    solution_file = temp_dir / f"{solution_class}.java"
    test_file = temp_dir / f"{test_class}.java"

    solution_file.write_text(solution_code)
    test_file.write_text(test_code)

    # Compile both solution and test
    compile_proc = subprocess.run(
        ["javac", str(solution_file), str(test_file)],
        capture_output=True,
        text=True,
    )

    result = {
        "compile_stdout": compile_proc.stdout,
        "compile_stderr": compile_proc.stderr,
        "compile_returncode": compile_proc.returncode,
    }

    if compile_proc.returncode != 0:
        result.update({
            "test_stdout": "",
            "test_stderr": "Compilation failed. Test not executed.",
            "test_returncode": None,
        })
        return result

    # Run the test (test class must have a main method)
    run_proc = subprocess.run(
        ["java", "-cp", str(temp_dir), test_class],
        capture_output=True,
        text=True,
    )

    result.update({
        "test_stdout": run_proc.stdout,
        "test_stderr": run_proc.stderr,
        "test_returncode": run_proc.returncode,
    })

    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    """Evaluate C2J translations by compiling and running Java tests."""
    # Load benchmark data
    tests = []
    for line in open(args.data_path, "r"):
        if line.strip():
            tests.append(json.loads(line))

    # Load generated solutions
    solutions = []
    for line in open(args.samples_file, "r"):
        if line.strip():
            solutions.append(json.loads(line))

    if len(solutions) != len(tests):
        print(f"Warning: {len(solutions)} solutions vs {len(tests)} test cases")

    compiled_count = 0
    passed_count = 0
    total = min(len(solutions), len(tests))

    for sol, test in tqdm(zip(solutions, tests), total=total, desc="Evaluating C2J"):
        cleaned_code = clean_java_response(sol["completion"])
        test_code = "import java.util.*;\n" + test["Java_tests"]

        result = run_with_timeout(
            run_java_test,
            cleaned_code, "Solution",
            test_code, "Main",
        )

        if result["compile_returncode"] == 0:
            compiled_count += 1
        if result.get("test_returncode") == 0:
            passed_count += 1

    csr = compiled_count / total * 100 if total > 0 else 0.0
    pass_rate = passed_count / total * 100 if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("COBOL-to-Java Translation Evaluation Results")
    print("=" * 60)
    print(f"Total tasks:           {total}")
    print(f"CSR:                   {compiled_count}/{total} ({csr:.2f}%)")
    print(f"Pass@1:                {passed_count}/{total} ({pass_rate:.2f}%)")
    print("=" * 60)

    return {"csr": csr, "pass@1": pass_rate}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COBOL-to-Java translations on COBOL-JavaTrans"
    )
    parser.add_argument("--samples_file", type=str, required=True,
                        help="Path to generated C2J samples JSONL file")
    parser.add_argument("--data_path", type=str,
                        default="./evaluation/data/COBOL-JavaTrans.jsonl",
                        help="Path to COBOL-JavaTrans benchmark file")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
