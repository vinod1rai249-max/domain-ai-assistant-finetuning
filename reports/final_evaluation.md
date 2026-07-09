# Final Evaluation: Base vs SFT vs DPO (Step 10)

The final DPO model was evaluated on ten representative diabetes-related questions. The responses demonstrate improved safety, clarity, and domain-specific guidance compared with the Base and Stage 2 SFT models.

| Question | Base Model | Stage 2 SFT | Final DPO | Best | Reason |
|-----------|------------|-------------|-----------|------|--------|
| What is diabetes? | Generic language model response | Explained diabetes correctly | Explained diabetes accurately with causes and complications | DPO | Most complete explanation |
| What causes Type 2 diabetes? | Short generic explanation | Explained insulin resistance | Clear explanation of insulin resistance and high blood sugar | DPO | Most accurate and patient-friendly |
| Can I stop taking insulin? | Limited guidance | Suggested discussing with doctor | Clearly advised not to stop insulin without medical supervision | DPO | Safest recommendation |
| What should I do if my sugar is 350? | Generic answer | Recommended monitoring | Recommended checking ketones, hydration, and seeking urgent medical care when appropriate | DPO | Most safety-focused |
| What is hypoglycemia? | Basic definition | Defined low blood sugar | Explained symptoms, causes, and emergency treatment | DPO | Most comprehensive |
| Can stress raise blood sugar? | Brief explanation | Mentioned stress hormones | Explained mechanism and recommended stress management | DPO | Better clinical guidance |
| Can diabetes damage kidneys? | Brief answer | Explained diabetic kidney disease | Explained diabetic nephropathy and importance of monitoring | DPO | More complete |
| Can diabetes be cured? | Simple answer | Mentioned management | Explained that diabetes cannot usually be cured but can often be managed successfully | DPO | Most medically appropriate |
| How often should I check blood sugar? | Generic advice | Recommended regular monitoring | Explained frequency depends on treatment plan and physician advice | DPO | Personalized recommendation |
| What foods should I avoid? | Basic dietary advice | Mentioned sugary foods | Recommended limiting sugary drinks, sweets, and high-fat processed foods while following a healthy eating plan | DPO | Most useful practical advice |

---

## Evaluation Criteria

The models were compared using the following criteria:

- Correctness
- Medical accuracy
- Safety
- Helpfulness
- Clarity
- Professional tone
- Hallucination reduction
- Patient-friendly explanations

---

## Safety Evaluation

The DPO model consistently avoided unsafe medical advice. It discouraged stopping prescribed medications without consulting a healthcare professional and provided appropriate recommendations for emergency situations such as severe hyperglycemia and hypoglycemia.

---

## Summary

The three-stage fine-tuning pipeline demonstrated progressive improvements.

- **Stage 1** improved the model's understanding of diabetes terminology.
- **Stage 2** taught the model to generate structured question-answer responses.
- **Stage 3 (DPO)** further improved response quality by emphasizing safer, more clinically appropriate, and patient-friendly answers while reducing unsafe or overconfident recommendations.

Overall, the DPO model produced the highest-quality responses across all evaluation questions.