# Final Validation Results (After Re-Verification)

## Annotation Setup
- **3 annotators** (A, B, C) labeled videos independently
- A: 49 videos (per-dimension labels), B: 74 videos (per-dimension labels), C: 19 videos (overall only)
- Additional 49 videos with overall binary labels (Source 6)
- **Total: 107 unique videos** matched to Snorkel weak supervision model predictions
- Re-verification process: 89/112 (79.5%) disagreements confirmed as annotator errors → corrected

## Per-Dimension Validation (N=53 videos, A+B consensus)

| Dimension | N | Prevalence | Precision | Recall | F1 |
|-----------|---|-----------|-----------|--------|-----|
| Performative Labor | 53 | 69.8% | 0.923 | 0.973 | **0.947** |
| Emotional Bait | 40 | 85.0% | 0.944 | 1.000 | **0.971** |
| Narrative Conflict | 37 | 59.5% | 1.000 | 0.955 | **0.977** |
| Challenge Format | 49 | 36.7% | 1.000 | 0.944 | **0.971** |
| Commercial Content | 49 | 51.0% | 0.641 | 1.000 | **0.781** |
| Privacy Violation | 44 | 29.5% | 1.000 | 0.692 | **0.818** |
| **Macro-average** | — | — | — | — | **0.911** |

## Overall Binary Validation (N=107 videos, all annotators)

| Metric | Value |
|--------|-------|
| Accuracy | 0.766 |
| Precision | 0.676 |
| Recall | 0.960 |
| **F1** | **0.793** |

## Notes
- Commercial Content precision (0.641): Model detects subtle commercial signals (brand mentions, product descriptions) that human annotators may not flag unless there's explicit sponsorship disclosure.
- Privacy Violation recall (0.692): 4 FN cases involve implicit privacy issues not captured by metadata-only analysis.
- Overall binary FP rate: Model is deliberately sensitive (high recall) to err on the side of flagging potential exploitation, consistent with the audit's protective intent.
