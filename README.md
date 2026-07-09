# Diabetes FAQ Assistant --- Domain-Specific LLM Fine-Tuning with Unsloth

## 1. Project Title

# Diabetes FAQ Assistant

A domain-specific healthcare AI assistant fine-tuned using a three-stage
LLM adaptation pipeline:

1.  Non-instruction fine-tuning
2.  Instruction fine-tuning (SFT)
3.  Direct Preference Optimization (DPO) alignment

The final model is designed to answer diabetes-related patient questions
with improved domain knowledge, safety awareness, helpfulness, and
professional medical communication compared with the original base
model.

------------------------------------------------------------------------

# 2. Domain Selected

## Healthcare FAQ Assistant --- Diabetes Domain

The selected domain is healthcare, specifically diabetes management.

The assistant covers:

-   Type 1 diabetes
-   Type 2 diabetes
-   Gestational diabetes
-   Diabetes symptoms
-   Blood glucose management
-   HbA1c understanding
-   Insulin and medications
-   Hypoglycemia
-   Diabetic ketoacidosis (DKA)
-   Diabetes complications
-   Emergency warning signs

------------------------------------------------------------------------

# 3. Business Problem

Healthcare organizations receive thousands of repetitive
diabetes-related questions from patients.

A domain-specific AI assistant can help provide:

-   Faster patient education
-   Consistent healthcare information
-   Better FAQ automation
-   Improved accessibility of medical knowledge

The objective of this project was to build a healthcare assistant that
provides:

-   Clinically grounded answers
-   Patient-friendly explanations
-   Safe medical guidance
-   Reduced hallucination compared with the base model

The final system demonstrates improvement across:

-   Correctness
-   Helpfulness
-   Domain accuracy
-   Safety
-   Tone
-   Clarity
-   Professional response quality

------------------------------------------------------------------------

# 4. Dataset Details

## Dataset Sources

  -----------------------------------------------------------------------------------------
  Dataset          Source                                          Purpose         Format
  ---------------- ----------------------------------------------- --------------- --------
  MedlinePlus      NIH/NLM public health information               Domain          Plain
  Diabetes                                                         knowledge       text
  Articles                                                         pretraining     

  Medical Meadow   Hugging Face                                    Instruction     JSONL
  Flashcards       `medalpaca/medical_meadow_medical_flashcards`   tuning          

  Patient          Manually created diabetes questions             Real-world      JSONL
  Generated                                                        patient         
  Examples                                                         language        

  Preference       Generated from SFT outputs + safety examples    DPO alignment   JSONL
  Dataset                                                                          
  -----------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Dataset Preparation

The data preparation pipeline included:

1.  Collecting diabetes-related medical content
2.  Cleaning and removing unnecessary text
3.  Filtering medical flashcards for diabetes topics
4.  Adding patient-style questions
5.  Creating instruction-response pairs
6.  Creating preference pairs:

```{=html}
<!-- -->
```
    Prompt
     |
     |---- Chosen answer (safe/high quality)
     |
     |---- Rejected answer (unsafe/low quality)

------------------------------------------------------------------------

# 5. Base Model Used

Model:

    unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit

Reason for selection:

-   Small enough for free-tier Colab GPU
-   Good instruction following capability
-   Compatible with QLoRA
-   Suitable for sequential fine-tuning experiments

Hardware used:

  Resource          Value
  ----------------- -----------------
  GPU               NVIDIA Tesla T4
  VRAM              \~15GB
  Framework         Unsloth
  Training Method   QLoRA

------------------------------------------------------------------------

# 6. Fine-Tuning Approach

The model was trained in three sequential stages.

------------------------------------------------------------------------

# Stage 1: Non-Instruction Fine-Tuning

## Objective

Teach the model diabetes-specific vocabulary and domain patterns before
instruction training.

## Approach

Raw diabetes documents from MedlinePlus were used.

Pipeline:

    Medical Articles
            |
            |
    Text Cleaning
            |
            |
    Chunking
            |
            |
    QLoRA Fine-Tuning

Configuration:

  Parameter               Value
  ----------------------- --------
  Parameter               Value
  -----------             ------
  Method                  QLoRA
  LoRA Rank               16
  Alpha                   32
  Dropout                 0.05
  Batch Size              2
  Gradient Accumulation   4
  Training Steps          300

Expected improvement:

-   Better diabetes terminology
-   Improved domain awareness

------------------------------------------------------------------------

# Stage 2: Instruction Fine-Tuning (SFT)

## Objective

Teach the model how to answer user questions.

Dataset format:

``` json
{
"instruction": "What are symptoms of diabetes?",
"response": "Common symptoms include..."
}
```

Training approach:

-   Supervised Fine-Tuning (SFT)
-   Instruction-response format
-   Patient-oriented questions

Configuration:

  Parameter               Value
  ----------------------- ------------
  Parameter               Value
  -----------             ------
  Method                  SFTTrainer
  Learning Rate           2e-4
  Batch Size              2
  Gradient Accumulation   4
  Training Steps          400

Expected improvement:

-   Better question answering
-   Improved explanation style
-   Better instruction following

------------------------------------------------------------------------

# Stage 3: DPO Alignment

## Objective

Improve response preference, safety, and communication style.

DPO trains the model using:

    Prompt

    Chosen Answer  → Preferred response

    Rejected Answer → Poor response

Example:

Rejected:

> Patient should immediately change medication dosage.

Chosen:

> Medication changes should be discussed with a healthcare professional.

Configuration:

  Parameter               Value
  ----------------------- ------------
  Method                  DPOTrainer
  Beta                    0.1
  Learning Rate           5e-6
  Epochs                  2
  Batch Size              1
  Gradient Accumulation   4

Expected improvement:

-   Safer answers
-   Reduced hallucination
-   Better professional tone

------------------------------------------------------------------------

# 7. LoRA / QLoRA Configuration

  ------------------------------------------------------------------------
  Parameter      Value
  -------------- ---------------------------------------------------------
  Quantization   4-bit

  Framework      Unsloth

  LoRA Rank      16

  LoRA Alpha     32

  LoRA Dropout   0.05

  Target Modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
                 down_proj
  ------------------------------------------------------------------------

------------------------------------------------------------------------

# 8. Why LoRA / QLoRA?

## Full Fine-Tuning Problem

Full model fine-tuning requires:

-   Updating billions of parameters
-   Large GPU memory
-   High compute cost

## LoRA

LoRA freezes the original model weights and trains small adapter layers.

Benefits:

-   Less GPU memory
-   Faster training
-   Smaller checkpoints

## QLoRA

QLoRA combines:

-   4-bit quantized base model
-   LoRA adapters

Benefits:

-   Enables fine-tuning on limited GPUs
-   Makes LLM customization possible on Colab T4 GPUs

------------------------------------------------------------------------

# 9. Training Notebooks

The complete training pipeline is available:

    notebooks/

    ├── non_instruction_finetuning.ipynb

    ├── instruction_finetuning.ipynb

    └── dpo_alignment.ipynb

------------------------------------------------------------------------

# 10. Evaluation Strategy

The model was evaluated at three stages:

## Models Compared

1.  Base Qwen2.5 model
2.  SFT fine-tuned model
3.  DPO aligned model

Evaluation criteria:

  Criteria
  -------------------------
  Correctness
  Helpfulness
  Domain accuracy
  Safety
  Tone
  Clarity
  Hallucination reduction
  Professional quality

Evaluation reports:

    reports/

    ├── base_model_evaluation.md

    ├── sft_model_comparison.md

    └── final_evaluation.md

------------------------------------------------------------------------

# 11. Before vs After Comparison

Example evaluation flow:

Question:

    What should I do if my blood sugar is high?

Compared outputs:

  Model        Result
  ------------ -----------------------------------------
  Base Model   General medical explanation
  SFT Model    More diabetes-specific answer
  DPO Model    Safer, clearer, patient-friendly answer

Detailed comparison:

    reports/final_evaluation.md

------------------------------------------------------------------------

# 12. Final Observations

Key observations:

-   Non-instruction fine-tuning improved diabetes vocabulary and domain
    familiarity.
-   SFT created stronger question-answering behavior.
-   DPO improved safety, tone, and response preference alignment.
-   The final assistant produced more professional healthcare responses
    compared with the base model.

------------------------------------------------------------------------

# 13. Challenges Faced

During implementation:

-   Limited GPU memory on free Colab environment
-   Managing multi-stage adapter training
-   Cleaning medical datasets
-   Converting flashcard data into patient-style questions
-   Creating meaningful preference pairs for DPO
-   Recovering checkpoints between Colab sessions

------------------------------------------------------------------------

# 14. Future Improvements

Possible improvements:

-   Train with larger healthcare datasets
-   Use larger models (7B+)
-   Add human evaluation
-   Add retrieval augmented generation (RAG)
-   Add medical knowledge verification layer
-   Build Gradio/Streamlit interface
-   Add safety guardrails and monitoring

------------------------------------------------------------------------

# 15. Repository Structure

    domain-ai-assistant-finetuning/

    ├── data/
    │   ├── non_instruction_data.txt
    │   ├── instruction_dataset.jsonl
    │   └── preference_dataset.jsonl

    ├── notebooks/
    │   ├── non_instruction_finetuning.ipynb
    │   ├── instruction_finetuning.ipynb
    │   └── dpo_alignment.ipynb

    ├── reports/
    │   ├── base_model_evaluation.md
    │   ├── sft_model_comparison.md
    │   ├── final_evaluation.md
    │   └── fine_tuning_explanation.md

    ├── scripts/

    ├── src/
    │   └── inference.py

    ├── README.md

    └── requirements.txt

------------------------------------------------------------------------

# 16. Running the Project

## Install Dependencies

``` bash
pip install -r requirements.txt
```

## Run Inference

``` bash
python src/inference.py "What are symptoms of diabetes?"
```

The inference script loads the final DPO-aligned model and allows users
to ask diabetes-related questions from the command line.

**Inference Script**

    src/inference.py

Example:

``` python
question = "What is HbA1c?"

answer = generate_answer(question)

print(answer)
```

------------------------------------------------------------------------

# 17. Final Project Summary

This project demonstrates the complete lifecycle of building a
domain-specific AI assistant:

    Raw Medical Data

            ↓

    Non-Instruction Fine-Tuning

            ↓

    Instruction Fine-Tuning (SFT)

            ↓

    Preference Alignment (DPO)

            ↓

    Healthcare AI Assistant

Interview explanation:

> "I built a diabetes-focused AI assistant using Unsloth and QLoRA. I
> first adapted the model using domain text, then trained it on
> instruction-response pairs, and finally aligned it using DPO
> preference optimization. I evaluated base, SFT, and DPO models and
> demonstrated improved healthcare response quality."
