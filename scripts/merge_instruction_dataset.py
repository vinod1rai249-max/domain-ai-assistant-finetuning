"""
Merges the filtered medalpaca flashcards with the hand-written patient-phrased
set into the final data/instruction_dataset.jsonl, deduplicated, with a
final count check against the 100-example minimum.

Run after both scrape_medlineplus.py-equivalent step (filter_medalpaca.py)
and after you've expanded patient_phrased_examples.jsonl if needed.
"""

import json

FLASHCARD_FILE = "data/instruction_dataset.jsonl"          # output of filter_medalpaca.py
PATIENT_FILE = "data/patient_phrased_examples.jsonl"       # hand-written seed set
OUTPUT_FILE = "data/instruction_dataset.jsonl"

MIN_EXAMPLES = 100
MIN_PATIENT_PHRASED = 20  # don't let flashcard phrasing dominate the dataset


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    flashcards = load_jsonl(FLASHCARD_FILE)
    patient = load_jsonl(PATIENT_FILE)

    print(f"Flashcard-derived examples: {len(flashcards)}")
    print(f"Patient-phrased examples: {len(patient)}")

    if len(patient) < MIN_PATIENT_PHRASED:
        print(
            f"WARNING: only {len(patient)} patient-phrased examples. "
            f"Add more to data/patient_phrased_examples.jsonl -- aim for "
            f">= {MIN_PATIENT_PHRASED} so the assistant doesn't sound purely "
            "like a flashcard deck."
        )

    combined = flashcards + patient
    seen = set()
    deduped = []
    for row in combined:
        key = row["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    print(f"\nFinal combined, deduplicated: {len(deduped)} examples")
    if len(deduped) < MIN_EXAMPLES:
        print(
            f"WARNING: under the {MIN_EXAMPLES}-example minimum by "
            f"{MIN_EXAMPLES - len(deduped)}. Loosen the medalpaca keyword "
            "filter or add more patient-phrased examples."
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
