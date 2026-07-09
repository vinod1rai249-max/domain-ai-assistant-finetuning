"""
Step 5: Test the BASE model (no fine-tuning at all) on the 10 standard
questions, before Stage 1 even runs. Writes reports/base_model_evaluation.md.

Run this FIRST, in Colab, before the Stage 1 notebook. Needs GPU.
"""

from unsloth import FastLanguageModel
import torch
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/domain-ai-assistant-finetuning/domain-ai-assistant-finetuning")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024

# Same 10 questions used in every later comparison stage -- do not change
# these between stages or the comparison is invalid.
TEST_QUESTIONS = [
    "How can I tell if my blood sugar is dangerously low?",
    "Can I skip my metformin if I feel fine today?",
    "What's the difference between type 1 and type 2 diabetes?",
    "Is it safe to exercise right after taking insulin?",
    "Why am I so thirsty all the time?",
    "Can type 2 diabetes be cured if I lose enough weight?",
    "What should I do if my blood sugar reads 250?",
    "Is it normal for blood sugar to vary day to day?",
    "Can stress affect my blood sugar?",
    "What's considered a dangerously low blood sugar level?",
]

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = []
    for q in TEST_QUESTIONS:
        prompt = PROMPT_TEMPLATE.format(instruction=q)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7, do_sample=True
        )
        answer = tokenizer.decode(
            output[0], skip_special_tokens=True
        ).split("### Response:")[-1].strip()
        print(f"Q: {q}\nA: {answer}\n{'-'*60}")
        rows.append((q, answer))

    # Write the Step 5 deliverable. "Problem" column is left for you to fill
    # in manually after reading each answer -- this script can't judge
    # whether an answer is generic or wrong, only generate it.
    REPORTS_DIR = PROJECT_DIR / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "base_model_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Base Model Evaluation (Step 5)\n\n")
        f.write(f"Base model: `{BASE_MODEL}` (no fine-tuning applied)\n\n")
        f.write("| Question | Base Model Answer | Problem |\n")
        f.write("|---|---|---|\n")
        for q, a in rows:
            a_escaped = a.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {q} | {a_escaped} | _fill in after review_ |\n")

    print(f"\nWrote {report_path}")
    print(
        "Open it and fill in the 'Problem' column yourself -- e.g. "
        "'generic, not diabetes-specific', 'too vague on emergency signs', "
        "'no safety caveat given'. This is the comparison baseline every "
        "later stage gets measured against, so don't skip reviewing it."
    )


if __name__ == "__main__":
    main()
