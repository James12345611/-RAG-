# RAG Eval Summary

- run_time: `2026-04-04T18:54:15`
- dataset: `E:\RAG_projects\eval\eval_set_template.jsonl`
- sample_count: `36`
- generation_model: `Qwen/Qwen3-14B`
- judge_model: `zai-org/GLM-4.6`

## Overall

- Hit@3: `0.7500`
- MRR@5: `0.6641`
- Evidence Recall@5: `0.8281`
- Answer Accuracy: `0.9062`
- Key Point Coverage: `0.9583`
- Faithfulness: `1.0000`
- Refusal F1: `1.0000`
- Over-Refusal Rate: `0.0000`

## By Question Type

### assumption

- sample_count: `6`
- Hit@3: `0.5000`
- Answer Accuracy: `0.9167`
- Faithfulness: `1.0000`
- Refusal F1: `0.0000`

### definition

- sample_count: `8`
- Hit@3: `0.8750`
- Answer Accuracy: `0.9375`
- Faithfulness: `1.0000`
- Refusal F1: `0.0000`

### formula_interpretation

- sample_count: `4`
- Hit@3: `0.7500`
- Answer Accuracy: `1.0000`
- Faithfulness: `1.0000`
- Refusal F1: `0.0000`

### method_comparison

- sample_count: `6`
- Hit@3: `1.0000`
- Answer Accuracy: `0.9167`
- Faithfulness: `1.0000`
- Refusal F1: `0.0000`

### method_selection

- sample_count: `8`
- Hit@3: `0.6250`
- Answer Accuracy: `0.8125`
- Faithfulness: `1.0000`
- Refusal F1: `0.0000`

### refusal

- sample_count: `4`
- Hit@3: `n/a`
- Answer Accuracy: `n/a`
- Faithfulness: `n/a`
- Refusal F1: `1.0000`

## By Difficulty

### easy

- sample_count: `5`
- Hit@3: `1.0000`
- Answer Accuracy: `0.9375`
- Faithfulness: `1.0000`
- Refusal F1: `1.0000`

### hard

- sample_count: `9`
- Hit@3: `0.6250`
- Answer Accuracy: `0.8438`
- Faithfulness: `1.0000`
- Refusal F1: `1.0000`

### medium

- sample_count: `22`
- Hit@3: `0.7500`
- Answer Accuracy: `0.9250`
- Faithfulness: `1.0000`
- Refusal F1: `1.0000`
