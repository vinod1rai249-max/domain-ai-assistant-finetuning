# Dataset build — usage

Run these in Colab, in order. None of this needs GPU; do it on a CPU runtime
to save your GPU quota for the actual training stages.

## 1. Non-instruction corpus (Step 2)
```
!pip install requests beautifulsoup4 lxml
!python scripts/scrape_medlineplus.py
```
Output: `data/non_instruction_data.txt`. Open it and skim — confirm paragraphs
read as continuous prose, not nav menus or "related links" junk. The script
filters short fragments but verify manually; HTML structure on government
sites changes without notice.

If it lands short of 50 unique paragraphs, the page selectors may need
adjusting (MedlinePlus occasionally changes container class names) — print
the raw paragraph count per URL (already logged) to see which pages failed.

## 2. Instruction dataset — flashcard half (Step 4)
```
!pip install datasets
!python scripts/filter_medalpaca.py
```
Output: `data/instruction_dataset.jsonl` (flashcard-derived, exam-phrased).

## 3. Instruction dataset — patient-phrased half (Step 4, continued)
`data/patient_phrased_examples.jsonl` already has 20 hand-written examples
covering routine management (diet, meds, monitoring) and emergency-adjacent
cases (hypoglycemia, DKA signs, "can I adjust my own insulin"). Add more if
you want better coverage — keep the same JSON shape:
```json
{"instruction": "...", "response": "..."}
```
The emergency-adjacent ones matter most: they're your seed material for the
Step 11 safety eval later, so don't only add easy diet questions.

## 4. Merge
```
!python scripts/merge_instruction_dataset.py
```
Output: `data/instruction_dataset_final.jsonl` — this is the file you actually
point your Stage 2 (SFT) training notebook at, not the raw flashcard file.

## What's still missing after this
`data/preference_dataset.jsonl` (Step 8) is NOT built yet — it can't be, since
it needs your Stage 2 SFT model's actual outputs to generate the "chosen vs
rejected" pairs against. That's the next deliverable once Stage 2 training
exists.
