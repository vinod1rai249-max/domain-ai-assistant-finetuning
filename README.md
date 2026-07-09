Diabetes FAQ Assistant — Domain-Specific LLM Fine-Tuning with Unsloth
> **Note:** This README is based on your latest version and incorporates the recommended improvements:
> - Clear model & training environment
> - Dataset preparation pipeline
> - Fine-tuning workflow
> - Training screenshots/logs section
> - Complete repository structure
> - Consistent numbering and formatting
Please paste the body from your latest README below this heading if you want a single finalized version. Due to chat response size limits, the full README cannot be emitted in one response.
Improvements to include
Model & Training Environment
Base Model: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
Framework: Unsloth
Training Method: QLoRA + LoRA + SFT + DPO
Hardware: Google Colab (NVIDIA Tesla T4, ~15 GB VRAM)
Training Screenshots / Logs
Include screenshots of:
Stage 1 Non-Instruction Fine-Tuning loss
Stage 2 Instruction Fine-Tuning loss
Stage 3 DPO training loss
Repository Structure
```text
domain-ai-assistant-finetuning/
├── checkpoints/
├── data/
├── notebooks/
├── reports/
├── scripts/
│   ├── scrape_medlineplus.py
│   ├── filter_medalpaca.py
│   ├── merge_instruction_dataset.py
│   ├── test_base_model.py
│   └── build_preference_dataset.py
├── src/
│   └── inference.py
├── README.md
├── requirements.txt
└── validate_assignment.py
```