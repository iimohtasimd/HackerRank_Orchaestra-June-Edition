#!/usr/bin/env python3

import csv
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    load_csv,
    call_vlm,
    post_process,
    fallback_row,
    build_prompt,
    DATASET_DIR,
    OUTPUT_COLUMNS,
)

EVAL_OUTPUT = Path(__file__).parent / "eval_output.csv"
REPORT_PATH = Path(__file__).parent / "evaluation_report.md"

EVAL_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
]


def evaluate():
    sample = load_csv(DATASET_DIR / "sample_claims.csv")
    history = {r["user_id"]: r for r in load_csv(DATASET_DIR / "user_history.csv")}
    reqs = load_csv(DATASET_DIR / "evidence_requirements.csv")

    print(f"Evaluating {len(sample)} sample claims...")

    predictions = []
    comparisons = []
    start = time.time()

    for i, row in enumerate(sample, 1):
        uid = row["user_id"]
        print(f"  [{i}/{len(sample)}] {uid} | {row['claim_object']}")

        prompt, images, _ = build_prompt(row, history.get(uid, {}), reqs)

        if not images:
            raw = {}
        else:
            raw = call_vlm(prompt, images)

        processed = post_process(raw, row["claim_object"]) if raw else fallback_row()

        result = {
            "user_id": uid,
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            **processed,
        }

        predictions.append(result)

        match = {
            f: processed.get(f, "").strip().lower() == row.get(f, "").strip().lower()
            for f in EVAL_FIELDS
        }

        comparisons.append(match)

        wrong = [f for f, v in match.items() if not v]
        correct = [f for f, v in match.items() if v]

        print(f"    ✓ {correct}  ✗ {wrong}")
        time.sleep(0.5)

    elapsed = time.time() - start
    n = len(comparisons)

    field_acc = {
        f: sum(c[f] for c in comparisons) / n
        for f in EVAL_FIELDS
    }

    overall = sum(all(c.values()) for c in comparisons) / n

    with open(EVAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)

    print("\n=== RESULTS ===")
    for f, a in field_acc.items():
        print(f"  {f}: {a:.1%}")

    print(f"  Overall: {overall:.1%}")
    print(f"  Time: {elapsed:.1f}s")

    write_report(field_acc, overall, n, elapsed)


def write_report(field_acc, overall, n_samples, elapsed):
    report = f"""# Evaluation Report — Multi-Modal Evidence Review

## Approach

This solution uses a local Ollama LLaVA vision-language model to review claim images and conversations.

## Sample Evaluation Results

| Field | Accuracy |
|---|---|
""" + "\n".join(
        f"| {f} | {a:.1%} |" for f, a in field_acc.items()
    ) + f"""
| **Overall** | **{overall:.1%}** |

## Operational Analysis

- Model used: Ollama LLaVA
- API cost: $0, local model
- Sample claims processed: {n_samples}
- Model calls: {n_samples}
- Runtime: {elapsed:.1f} seconds
- Average latency: {elapsed / n_samples:.1f} seconds per claim
- Images are converted to JPEG and resized before processing.
- Output fields are sanitized in `main.py`.
- If processing fails, fallback output uses `not_enough_information` and `manual_review_required`.

## Risk Handling

- Image evidence is treated as the primary source of truth.
- User history is used only for risk context.
- Prompt injection or suspicious text should be flagged for manual review.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()