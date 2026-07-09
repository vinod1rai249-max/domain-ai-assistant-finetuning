"""
Step 4: Instruction dataset (diabetes domain).

Filters medalpaca/medical_meadow_medical_flashcards down to diabetes-relevant
cards, reformats to {"instruction": ..., "response": ...}, and writes
data/instruction_dataset.jsonl.

Run this in Colab:
    !pip install datasets

IMPORTANT: this dataset is exam/textbook-phrased ("Define...", "What is the
mechanism of..."). Real patients don't ask like that. This script gets you
the domain-correct content fast, but you MUST add the manual patient-phrased
set below or your assistant will sound like a med-school flashcard, not an
FAQ bot. That's the single biggest quality risk in this step -- don't skip it.
"""

import json
import re
from datasets import load_dataset

# Keywords covering type 1/2, gestational, hypo/hyperglycemia, insulin,
# complications, diet/management -- matches the MedlinePlus page spread above
# so the two datasets reinforce the same sub-topics.
DIABETES_KEYWORDS = [
    "diabet", "insulin", "hypoglycemi", "hyperglycemi", "blood sugar",
    "blood glucose", "a1c", "hba1c", "ketoacidosis", "metformin",
    "glucose tolerance", "pancrea", "glycemic", "sulfonylurea",
]

MIN_RESPONSE_WORDS = 4   # drop one-word flashcard answers, too thin to train on


def is_diabetes_related(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in DIABETES_KEYWORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    ds = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards", split="train"
    )
    print(f"Loaded {len(ds)} total flashcards")

    filtered = []
    seen_instructions = set()
    for row in ds:
        instruction = clean_text(row.get("input", row.get("instruction", "")))
        response = clean_text(row.get("output", row.get("response", "")))

        if not instruction or not response:
            continue
        if len(response.split()) < MIN_RESPONSE_WORDS:
            continue
        if instruction in seen_instructions:
            continue
        if is_diabetes_related(instruction) or is_diabetes_related(response):
            filtered.append({"instruction": instruction, "response": response})
            seen_instructions.add(instruction)

    print(f"Filtered to {len(filtered)} diabetes-related cards")

    out_path = "data/instruction_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    print(
        f"\nNext: you need ~{max(0, 100 - len(filtered))} more examples to hit "
        "the 100 minimum, AND you need patient-phrased pairs regardless of "
        "count (see patient_phrased_template.jsonl). Aim for at least "
        "20-30 patient-style questions mixed in even if the flashcard count "
        "alone clears 100."
    )


if __name__ == "__main__":
    main()
