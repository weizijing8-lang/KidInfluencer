# Robustness Check Results Summary

## Key Findings

### Check 1: Views/Day (mechanical age control)
- Only 21 channels have enough publishedAt data
- Median boost: -2.7% (NOT significant)
- Mean boost: +174.3% (driven by outliers)
- Wilcoxon p=0.585 — NOT SIGNIFICANT
- Problem: very few channels have publishedAt data after merge

### Check 2: Same-Year Within-Channel Comparison
- 26 channel-year groups
- Median boost: +26.2% (SIGNIFICANT)
- Wilcoxon p=0.0018 ***
- t-test p=0.0148 *
- 73% of groups show positive boost
- THIS IS THE KEY ROBUSTNESS CHECK — even comparing videos from the same year, exploitative content gets more views

### Check 4: Within-Channel by Dimension (raw views, FDR-corrected)
- performative: +42.0% median, p_FDR < 0.001 ***
- emotional_bait: +13.3% median, p_FDR = 0.007 **
- narrative_conflict: +32.0% median, p_FDR = 0.002 **
- challenge_format: +14.8% median, p_FDR = 0.007 **
- commercial_content: +3.7% median, p_FDR = 0.560 n.s.
- privacy_violation: -7.4% median, p_FDR = 0.114 n.s.

### Check 5: Views/Day by Dimension (FDR-corrected)
- ALL dimensions become non-significant after FDR correction
- But sample sizes are very small (5-19 channels) due to missing publishedAt
- This is a DATA LIMITATION, not necessarily evidence against the effect

## Interpretation for Paper

1. Main analysis (raw views, within-channel): 4 of 6 dimensions significant after FDR
2. Same-year robustness: overall effect HOLDS (p=0.0018)
3. Views/day robustness: inconclusive due to data limitations (only 21 channels have dates)

## Recommended Framing
- Report main analysis as primary results
- Report same-year comparison as key robustness check (holds!)
- Report views/day as additional check, note data limitation
- Be transparent that views/day analysis is underpowered
