"""
Validates repo structure + minimum data counts against the assignment spec.
Does NOT validate notebook correctness, model quality, or whether DPO
actually improved anything -- that requires reading the notebooks/reports
yourself. This only catches "deliverable missing or under minimum size."

Usage (from repo root, e.g. F:\\fine_tunning\\files):
    python validate_assignment.py
"""

import json
import os
import sys

ROOT = os.getcwd()

REQUIRED_FILES = {
    "data/non_instruction_data.txt": "Step 2 deliverable",
    "data/instruction_dataset.jsonl": "Step 4 deliverable",
    "data/preference_dataset.jsonl": "Step 8 deliverable",
    "notebooks/non_instruction_finetuning.ipynb": "Step 3 deliverable",
    "notebooks/instruction_finetuning.ipynb": "Step 6 deliverable",
    "notebooks/dpo_alignment.ipynb": "Step 9 deliverable",
    "reports/base_model_evaluation.md": "Step 5 deliverable",
    "reports/sft_model_comparison.md": "Step 7 deliverable",
    "reports/final_evaluation.md": "Step 10 deliverable",
    "reports/fine_tuning_explanation.md": "Step 11 deliverable",
    "src/inference.py": "Step 12 deliverable",
    "README.md": "Final submission requirement",
    "requirements.txt": "Final submission requirement",
}

README_REQUIRED_SECTIONS = [
    "domain", "business problem", "dataset", "base model",
    "non-instruction", "instruction", "dpo", "lora", "qlora",
    "before", "after", "observation", "challenge", "future",
]

MIN_COUNTS = {
    "data/non_instruction_data.txt": ("paragraphs", 50),
    "data/instruction_dataset.jsonl": ("jsonl_lines", 100),
    "data/preference_dataset.jsonl": ("jsonl_lines", 50),
}

REQUIRED_JSONL_KEYS = {
    "data/instruction_dataset.jsonl": {"instruction", "response"},
    "data/preference_dataset.jsonl": {"prompt", "chosen", "rejected"},
}

TABLE_REQUIRED_REPORTS = {
    "reports/base_model_evaluation.md": ["Question", "Base Model Answer", "Problem"],
    "reports/sft_model_comparison.md": [
        "Question", "Base Model Answer", "Fine-Tuned Model Answer",
        "Which is Better", "Reason",
    ],
    "reports/final_evaluation.md": [
        "Question", "Base Model Answer", "SFT Model Answer",
        "DPO Model Answer", "Best Answer", "Reason",
    ],
}

NOTEBOOK_REQUIRED_SIGNALS = {
    "notebooks/non_instruction_finetuning.ipynb": [
        "FastLanguageModel", "lora", "save_pretrained",
    ],
    "notebooks/instruction_finetuning.ipynb": [
        "FastLanguageModel", "lora", "SFTTrainer", "save_pretrained",
    ],
    "notebooks/dpo_alignment.ipynb": [
        "DPOTrainer", "chosen", "rejected", "save_pretrained",
    ],
}


def count_paragraphs(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return len(blocks)


def count_jsonl_lines(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def check_jsonl_keys(path, required_keys):
    bad_rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_rows.append((i, "invalid JSON"))
                continue
            missing = required_keys - set(row.keys())
            if missing:
                bad_rows.append((i, f"missing keys: {missing}"))
    return bad_rows


def check_notebook_signals(path, signals):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [s for s in signals if s.lower() not in content.lower()]


def check_report_table(path, required_columns):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    missing = [c for c in required_columns if c.lower() not in content.lower()]
    return missing


def main():
    failures = []
    warnings = []

    print("=== File presence ===")
    for rel_path, label in REQUIRED_FILES.items():
        full_path = os.path.join(ROOT, rel_path)
        exists = os.path.isfile(full_path)
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {rel_path}  ({label})")
        if not exists:
            failures.append(f"Missing required file: {rel_path}")

    print("\n=== Minimum count checks ===")
    for rel_path, (kind, minimum) in MIN_COUNTS.items():
        full_path = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue
        if kind == "paragraphs":
            n = count_paragraphs(full_path)
        else:
            n = count_jsonl_lines(full_path)
        status = "OK" if n >= minimum else "BELOW MINIMUM"
        print(f"[{status}] {rel_path}: {n} (need >= {minimum})")
        if n < minimum:
            failures.append(f"{rel_path} has {n}, needs >= {minimum}")

    print("\n=== JSONL schema checks ===")
    for rel_path, required_keys in REQUIRED_JSONL_KEYS.items():
        full_path = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue
        bad_rows = check_jsonl_keys(full_path, required_keys)
        if bad_rows:
            print(f"[FAIL] {rel_path}: {len(bad_rows)} malformed rows")
            for i, reason in bad_rows[:5]:
                print(f"   line {i}: {reason}")
            failures.append(f"{rel_path} has {len(bad_rows)} malformed rows")
        else:
            print(f"[OK] {rel_path}: all rows have required keys {required_keys}")

    print("\n=== Notebook content signal checks (heuristic only) ===")
    for rel_path, signals in NOTEBOOK_REQUIRED_SIGNALS.items():
        full_path = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue
        missing = check_notebook_signals(full_path, signals)
        if missing:
            print(f"[WARN] {rel_path}: no mention of {missing}")
            warnings.append(f"{rel_path} missing expected signals: {missing}")
        else:
            print(f"[OK] {rel_path}: contains expected signals {signals}")

    print("\n=== Report table column checks ===")
    for rel_path, columns in TABLE_REQUIRED_REPORTS.items():
        full_path = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue
        missing = check_report_table(full_path, columns)
        if missing:
            print(f"[FAIL] {rel_path}: missing table columns {missing}")
            failures.append(f"{rel_path} missing columns: {missing}")
        else:
            print(f"[OK] {rel_path}: has expected table columns")

    print("\n=== README section checks ===")
    readme_path = os.path.join(ROOT, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            readme_lower = f.read().lower()
        missing_sections = [s for s in README_REQUIRED_SECTIONS if s not in readme_lower]
        if missing_sections:
            print(f"[WARN] README.md missing mentions of: {missing_sections}")
            warnings.append(f"README missing topics: {missing_sections}")
        else:
            print("[OK] README.md mentions all required topics")

    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: {len(failures)} FAILURES -- not complete\n")
        for f_ in failures:
            print(f" - {f_}")
    else:
        print("RESULT: all structural/count checks passed")

    if warnings:
        print(f"\n{len(warnings)} WARNINGS (review manually, script can't judge these):")
        for w in warnings:
            print(f" - {w}")

    print(
        "\nNOTE: this script cannot verify notebook execution actually "
        "succeeded, that comparison tables contain real (non-placeholder) "
        "rows, that DPO genuinely improved outputs, or that your safety/"
        "trap-question eval exists. Check those manually."
    )

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
