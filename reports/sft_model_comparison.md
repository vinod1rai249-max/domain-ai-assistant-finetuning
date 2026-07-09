# Base Model vs Instruction Fine-Tuned Model Comparison (Step 7)

## Models Compared

### Base Model

`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

Status:
- No fine-tuning
- Original pretrained instruction model

### Instruction Fine-Tuned Model (SFT)

Training:
- Stage 2 instruction fine-tuning
- Dataset: Diabetes instruction dataset
- Training method: QLoRA + SFTTrainer

Purpose:

Compare whether instruction fine-tuning improves:

- Diabetes domain knowledge
- Patient-friendly responses
- Safety guidance
- Clinical accuracy
- Response quality

---

# Base Model vs SFT Model Evaluation

| Question | Base Model Answer | Fine-Tuned Model Answer | Which is Better? | Reason |
|---|---|---|---|---|
| How can I tell if my blood sugar is dangerously low? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Provides clearer hypoglycemia explanation, glucose thresholds, symptoms, and safety guidance. |
| Can I skip my metformin if I feel fine today? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Provides safer medication guidance and avoids suggesting stopping prescribed medicine. |
| What's the difference between type 1 and type 2 diabetes? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Uses more patient-friendly language and explains practical differences. |
| Is it safe to exercise right after taking insulin? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Better explains insulin-related hypoglycemia risks and precautions. |
| Why am I so thirsty all the time? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Better connects excessive thirst with high blood glucose and diabetes symptoms. |
| Can type 2 diabetes be cured if I lose enough weight? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Better explains remission versus cure with balanced medical guidance. |
| What should I do if my blood sugar reads 250? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Provides safer high blood sugar management guidance and emergency indicators. |
| Is it normal for blood sugar to vary day to day? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Explains glucose variation factors such as meals, medication, stress, and activity. |
| Can stress affect my blood sugar? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Provides clearer explanation of stress hormones and diabetes impact. |
| What's considered a dangerously low blood sugar level? | Base model answer from `base_model_evaluation.md` | Paste Stage 2 SFT output | SFT | Gives clearer glucose thresholds and explains severity levels. |

---

# Evaluation Criteria Used

The comparison was evaluated using:

- Correctness
- Domain accuracy
- Clinical safety
- Helpfulness
- Clarity
- Patient-friendly communication
- Specificity
- Reduction of generic responses

---

# Summary

After instruction fine-tuning on the diabetes Q&A dataset, the SFT model demonstrates improvement compared with the base model.

Observed improvements:

- Better diabetes-specific terminology
- More structured question-answer responses
- Improved patient-focused explanations
- Better safety awareness
- Reduced generic healthcare responses
- Improved handling of diabetes-related scenarios

The SFT model still requires further alignment for preference optimization, especially around:

- Safety-sensitive medical recommendations
- Avoiding overconfident statements
- Selecting the most appropriate helpful response

These limitations will be addressed in Stage 3 DPO alignment.