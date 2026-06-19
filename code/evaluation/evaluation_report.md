# Evaluation Report — Multi-Modal Evidence Review

## Approach

This solution uses a local Ollama LLaVA vision-language model to review claim images and conversations.

## Sample Evaluation Results

| Field | Accuracy |
|---|---|
| claim_status | 10.0% |
| issue_type | 20.0% |
| object_part | 30.0% |
| severity | 15.0% |
| evidence_standard_met | 80.0% |
| valid_image | 70.0% |
| **Overall** | **0.0%** |

## Operational Analysis

- Model used: Ollama LLaVA
- API cost: $0, local model
- Sample claims processed: 20
- Model calls: 20
- Runtime: 2321.4 seconds
- Average latency: 116.1 seconds per claim
- Images are converted to JPEG and resized before processing.
- Output fields are sanitized in `main.py`.
- If processing fails, fallback output uses `not_enough_information` and `manual_review_required`.

## Risk Handling

- Image evidence is treated as the primary source of truth.
- User history is used only for risk context.
- Prompt injection or suspicious text should be flagged for manual review.
