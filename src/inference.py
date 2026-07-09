"""
Step 12 deliverable. Loads your DPO-trained model from a LOCAL folder
(what you'll have after downloading the saved adapter from Colab/Drive to
F:\\fine_tunning\\files or wherever your repo lives).

Auto-detects whether a CUDA GPU is available:
  - GPU present  -> uses Unsloth's FastLanguageModel (fast path)
  - No GPU       -> falls back to plain transformers + peft (CPU, slower,
                    but works without an NVIDIA card)

Usage:
    python src/inference.py "How can I tell if my blood sugar is dangerously low?"

Before running, set MODEL_PATH below to the folder containing your
downloaded adapter (the same folder your DPO notebook called
model.save_pretrained(...) on).
"""

import sys
import torch

# Path to the LOCAL folder with your saved adapter/model, e.g.:
# r"F:\fine_tunning\files\outputs\dpo_model"
MODEL_PATH = r"/content/drive/MyDrive/domain-ai-assistant-finetuning/checkpoints/stage3_dpo_final"

# Base model your adapter was trained on top of (only used by CPU fallback).
BASE_MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # match whatever you trained on

MAX_SEQ_LENGTH = 2048

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def has_cuda() -> bool:
    return torch.cuda.is_available()


def load_model_gpu():
    """Fast path: Unsloth, requires CUDA."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_model_cpu():
    """Fallback path: plain transformers + peft, no GPU required.
    Slower (expect 10-60s per answer), but doesn't need Unsloth's CUDA-only
    kernels.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    return model, tokenizer


def load_model():
    if has_cuda():
        print("CUDA GPU detected -- using Unsloth fast path.")
        return load_model_gpu()
    else:
        print("No CUDA GPU detected -- using CPU fallback (slower).")
        return load_model_cpu()


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/inference.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    model, tokenizer = load_model()
    answer = generate_answer(model, tokenizer, question)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
