# Pipeline v4 Results: Vision + LLM Combined Classification

**Date:** 2026-05-12

## Method

Weighted combination of two signal sources:
- **LLM text-based** (title only, binary): weight = 0.33
- **GPT-4 Vision** (title + thumbnail + description, continuous [0,1]): weight = 0.67

Weights based on validation against 23 human annotations (κ_vision=0.617 vs κ_llm=0.309).

## Classification Summary

| Metric | v3 (Snorkel LM) | v4 (Vision+LLM) |
|--------|-----------------|------------------|
| Total videos | 4,685 | 5,051 |
| Vision coverage | 217 (4.6%) | 4,673 (92.5%) |
| Exploitative | ~35% | 19.7% (997) |
| Clean | ~65% | 80.3% (4,054) |
| Mean score | 0.51 | 0.327 |

## Overall Engagement Premium

| Metric | v3 | v4 |
|--------|-----|-----|
| Channels analyzed | 48 | 54 |
| Wilcoxon p-value | 0.0003 | **0.000066** |
| Median view ratio | 1.18x | **1.23x** |
| Positive premium channels | 69% | **74.1%** |
| Mean log premium | 0.34 | **0.388** |

## Per-Dimension Results (FDR-Corrected)

| Dimension | n_channels | Mean Premium | p_raw | p_fdr | Significant |
|-----------|-----------|-------------|-------|-------|-------------|
| performative_labor | 67 | +0.354 | 0.000046 | **0.000279** | YES |
| emotional_bait | 60 | +0.281 | 0.000484 | **0.001452** | YES |
| narrative_conflict | 50 | +0.153 | 0.007609 | **0.012188** | YES |
| challenge_format | 49 | +0.208 | 0.023008 | **0.023008** | YES |
| commercial_content | 44 | -0.325 | 0.010157 | **0.012188** | YES (negative!) |
| privacy_violation | 20 | +0.401 | 0.008308 | **0.012188** | YES |

## Key Changes from v3

1. **All 6 dimensions now significant after FDR correction** (v3 had only 3)
2. **Commercial content shows NEGATIVE premium** (-0.325): videos with commercial content get FEWER views. This is a new and interesting finding.
3. **Privacy violation now significant** (p_fdr=0.012): was previously underpowered (p=0.114 in v3)
4. **Performative labor** has the strongest signal (p_fdr=0.000279)
5. **Same-year robustness** still holds (p=0.006)

## Robustness: Same-Year Within-Channel

| Metric | Value |
|--------|-------|
| Channel-year pairs | 37 |
| Mean premium | +0.404 |
| Wilcoxon p | 0.006 ** |
| Positive pairs | 24/37 (64.9%) |

## Spearman Correlation

ρ = 0.328, p = 2.83e-127 (exploitation score vs log views)

## Notable Finding: Commercial Content Negative Premium

The negative premium for commercial content (-0.325) suggests that **overt commercial content (product placements, sponsored videos) actually reduces engagement**. This aligns with audience research showing viewers dislike obvious advertising. The exploitation premium is driven by emotional/performative dimensions, not commercial ones.
