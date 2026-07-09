# Fine-Tuning Explanation (Step 11)

## Why Full Fine-Tuning is Expensive

Full fine-tuning updates every parameter in a pretrained language model. A model with billions of parameters requires a large amount of GPU memory to store the model weights, gradients, and optimizer states during training. This significantly increases computational cost, training time, and storage requirements. Because of these hardware requirements, full fine-tuning is generally not practical on free Google Colab GPUs and is mostly used by organizations with access to high-end computing infrastructure.

---

## What LoRA Does

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating every parameter in the model, LoRA freezes the original pretrained weights and inserts small trainable matrices into selected transformer layers. During training, only these adapter weights are updated while the original model remains unchanged.

This approach provides several advantages:

* Reduces GPU memory usage
* Speeds up training
* Produces very small checkpoint files
* Preserves the original model's general language capabilities

---

## What QLoRA Does

QLoRA combines LoRA with 4-bit quantization of the pretrained model. The frozen base model is loaded in a compressed 4-bit format while the LoRA adapter weights continue to train in higher precision.

Because the largest part of the model is stored using much less memory, QLoRA makes it possible to fine-tune modern language models on hardware with limited GPU memory.

---

## Why QLoRA is Useful on Limited GPU Resources

This project was completed using a free Google Colab Tesla T4 GPU with approximately 15 GB of VRAM.

Without QLoRA, loading and fine-tuning the base model would require significantly more GPU memory than available. Using QLoRA through Unsloth reduced memory requirements enough to complete all three fine-tuning stages on the available hardware while maintaining good model quality.

For students and individual developers, QLoRA provides an efficient and cost-effective way to fine-tune large language models without requiring expensive GPUs.

---

## What is Non-Instruction Fine-Tuning?

Non-instruction fine-tuning trains the model on raw domain-specific text rather than question-answer pairs.

In this project, diabetes-related educational content from MedlinePlus was used to teach the model medical terminology, writing style, and background knowledge about diabetes.

The objective of this stage was not to teach the model how to answer questions but to improve its understanding of diabetes-related information before instruction tuning.

---

## What is Instruction Fine-Tuning (SFT)?

Instruction Fine-Tuning, also called Supervised Fine-Tuning (SFT), trains the model using instruction-response pairs.

Each training example contains:

* A user question (instruction)
* A high-quality expected response

This stage teaches the model how to generate structured, patient-friendly answers instead of simply predicting the next token.

For this project, the instruction dataset consisted of diabetes-related question-answer pairs created from filtered Medical Meadow flashcards together with additional patient-style examples.

---

## What is DPO?

Direct Preference Optimization (DPO) is a preference alignment technique.

Instead of training only on correct answers, DPO learns from pairs of responses:

* **Chosen response** – the preferred, safer, and more helpful answer.
* **Rejected response** – a less useful, generic, or potentially unsafe answer.

The model learns to prefer responses that better match human preferences.

In this project, DPO was used to improve:

* Patient safety
* Professional tone
* Medical caution
* Overall response quality

---

## Difference Between SFT and DPO

Supervised Fine-Tuning (SFT) teaches the model how to answer questions correctly by learning from instruction-response examples.

Direct Preference Optimization (DPO) further improves the model by teaching it which good answers should be preferred when multiple possible responses exist.

In simple terms:

* **SFT teaches the model what to say.**
* **DPO teaches the model how to say it more safely, professionally, and helpfully.**

Therefore, SFT improves domain knowledge, while DPO improves alignment with human preferences and safer response generation.

---

# Hyperparameters Used in This Project

| Parameter               | Stage 1 (Non-Instruction FT) | Stage 2 (SFT) | Stage 3 (DPO)         |
| ----------------------- | ---------------------------- | ------------- | --------------------- |
| LoRA Rank (r)           | 16                           | 16            | Existing LoRA adapter |
| LoRA Alpha              | 32                           | 32            | Existing LoRA adapter |
| LoRA Dropout            | 0.05                         | 0.05          | Existing LoRA adapter |
| Learning Rate           | 2e-4                         | 2e-4          | 5e-6                  |
| Batch Size (per device) | 2*                           | 2*            | 1                     |
| Gradient Accumulation   | 4                            | 4             | 4                     |
| Training Steps / Epochs | 300 steps                    | 400 steps     | 2 epochs              |
| DPO Beta                | —                            | —             | 0.1                   |
| Optimizer               | AdamW 8-bit                  | AdamW 8-bit   | AdamW 8-bit           |

*If the batch size was reduced during training to avoid GPU memory issues, the values in this table should be updated to reflect the actual training configuration.

---

# Conclusion

In this project, I built a domain-specific Diabetes FAQ Assistant using a three-stage fine-tuning pipeline with Unsloth and QLoRA.

First, I performed non-instruction fine-tuning on diabetes-related medical text to improve the model's domain knowledge. Next, I used supervised fine-tuning (SFT) with question-answer pairs to teach the model how to respond accurately and clearly to patient queries. Finally, I applied Direct Preference Optimization (DPO) to align the model with safer, more professional, and more helpful responses.

Using LoRA and QLoRA allowed the entire training process to be completed efficiently on a free Google Colab Tesla T4 GPU while significantly reducing memory usage compared to full fine-tuning.
