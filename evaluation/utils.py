"""
Shared utilities for COBOL-Coder evaluation.
"""

import gzip
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List

import marko
from marko.block import FencedCode


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """Parses each jsonl line and yields it as a dictionary."""
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """Writes an iterable of dictionaries to jsonl."""
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# Code extraction & post-processing
# ---------------------------------------------------------------------------

def extract_code_block(src: str) -> str:
    """Extract the first fenced code block from markdown source."""
    try:
        markdown = marko.parse(src)

        def search_for_code(element, code_blocks):
            if isinstance(element, FencedCode):
                code_blocks.append(element.children[0].children)
            elif hasattr(element, "children"):
                for child in element.children:
                    search_for_code(child, code_blocks)

        code_blocks: List[str] = []
        search_for_code(markdown, code_blocks)
        return code_blocks[0]
    except Exception:
        return src


def re_structure_output(prefix: str, output: str) -> str:
    """
    Post-process generated COBOL output to ensure correct section ordering
    (WORKING-STORAGE before LINKAGE before PROCEDURE DIVISION).
    """
    output = extract_code_block(output)

    if "IDENTIFICATION DIVISION" in output:
        try:
            index = output.index("IDENTIFICATION DIVISION")
            output = output[index:]
            working_index = output.index("WORKING-STORAGE SECTION.")
            procedure_index = output.index("\n         PROCEDURE DIVISION")
            linkage_index = output.index("LINKAGE SECTION.")
            working_division = output[working_index:procedure_index] + "\n         "
            prefix = output[:linkage_index]
            procedure = output[procedure_index:]
            linkage_section = output[linkage_index:working_index]
            output = prefix + working_division + linkage_section + procedure
        except Exception:
            pass
    else:
        try:
            procedure_index = output.index("PROCEDURE DIVISION")
            working_division = "WORKING-STORAGE SECTION.\n" + output[:procedure_index]
            procedure = output[procedure_index:]
            prefix = prefix[: prefix.rindex("WORKING-STORAGE SECTION.")]
            linkage_index = prefix.index("LINKAGE SECTION.")
            linkage_section = prefix[linkage_index:]
            prefix = prefix[: prefix.index("LINKAGE SECTION.")]
            output = prefix + working_division + linkage_section + procedure
        except Exception:
            pass

    return output


def swap_sections(src: str) -> str:
    """
    Swap the Working Storage and Linkage Sections to canonical order.
    Used for COBOLEval evaluation format.
    """
    working_storage, linkage, procedure, begin = [], [], [], []
    current_section = begin

    for line in src.split("\n"):
        stripped_line = line.strip().upper()
        if stripped_line.startswith("WORKING-STORAGE SECTION."):
            current_section = working_storage
        elif stripped_line.startswith("LINKAGE SECTION."):
            current_section = linkage
        elif stripped_line.startswith("PROCEDURE DIVISION"):
            current_section = procedure
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        current_section.append(line)

    return "\n".join(begin + working_storage + linkage + procedure)


def clean_response_for_eval(code: str) -> str:
    """Clean up model-generated COBOL code for evaluation."""
    # Remove duplicate WORKING-STORAGE SECTION
    tmp = "WORKING-STORAGE SECTION.\n       WORKING-STORAGE SECTION."
    code = code.replace(tmp, "WORKING-STORAGE SECTION.")

    try:
        # Ensure proper COBOL indentation (column 8+)
        if code[0] != " ":
            code = "       " + code

        working_index = code.index("WORKING-STORAGE SECTION.")
        try:
            procedure_index = code.index("PROCEDURE DIVISION USING")
        except ValueError:
            procedure_index = code.index("PROCEDURE DIVISION.")
        linkage_index = code.index("LINKAGE SECTION.")

        if working_index > linkage_index:
            working_division = code[working_index:procedure_index]
            prefix = code[:linkage_index]
            procedure = code[procedure_index:]
            linkage_section = code[linkage_index:working_index]
            code = prefix + working_division + linkage_section + procedure
    except Exception:
        pass

    return code


def clean_java_response(code: str) -> str:
    """Clean up model-generated Java code for evaluation."""
    # Truncate at duplicate import block (model repeating itself)
    idx = code[100:].find("import java")
    if idx != -1:
        return code[: idx + 100]
    return code


# ---------------------------------------------------------------------------
# Comment removal
# ---------------------------------------------------------------------------

def remove_block_comments(java_code: str) -> str:
    """Remove all block comments (/* ... */) from Java code."""
    return re.sub(r"/\*.*?\*/", "", java_code, flags=re.DOTALL)


def remove_cobol_comments(cobol_code: str) -> str:
    """Remove COBOL comment lines (lines starting with '*')."""
    cleaned_lines = []
    for line in cobol_code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("*"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Shell execution helpers
# ---------------------------------------------------------------------------

def cmd(command: str, timeout: int = 5):
    """Run a shell command and return (success, stderr)."""
    process = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return process.returncode == 0, process.stderr


def cleanup_file(name: str):
    """Remove a file if it exists."""
    try:
        os.remove(name)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class Model:
    name: str
    saved_name: str = ""
    temp: float = 0.0
    samples_per_task: int = 1
    tokenizer: str = None
    prefix_token: str = None
    suffix_token: str = None
    middle_token: str = None
    eos_token: str = None
