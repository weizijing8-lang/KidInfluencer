# Verified Statistics from Snorkel Proper Pipeline (Filtered, Kid-Centric)

## Dataset
- Videos: 4,208
- Channels: 56 (kid-centric only, 23 adult channels removed)
- Labeling Functions: 33 (heterogeneous: LLM, VLM, keyword rules, description, duration, tags)
- Overall threshold: 0.7
- Per-dimension threshold: 0.6 (privacy: 0.5)

## Overall Exploit Rate
- 26.5% (1,114/4,208 videos classified as exploitative at threshold 0.7)

## Spearman Correlation (continuous score vs views)
- rho = 0.229, p = 4.00e-51

## Per-Dimension Prevalence (at threshold 0.6 / 0.5 for privacy)
- Performative Labor: 69.5%
- Emotional Bait: 56.2%
- Narrative Conflict: 34.9%
- Challenge Format: 32.4%
- Commercial Content: 41.4%
- Privacy Violation: 22.9%

## Mixed-Effects Model 1: Overall Score
- Beta = 0.647, SE = 0.059, z = 10.91, p = 1.00e-27
- Random intercept variance: 0.741
- ICC: 0.738
- Multiplier: 10^0.647 = 4.4x

## Mixed-Effects Model 2: All Dimensions (continuous prob)
- Performative Labor: beta=0.091, p=0.0005 ***
- Emotional Bait: beta=0.264, p<0.0001 ***
- Narrative Conflict: beta=0.153, p=0.0001 ***
- Challenge Format: beta=0.003, p=0.9156 n.s.
- Commercial Content: beta=0.042, p=0.2448 n.s.
- Privacy Violation: beta=0.226, p<0.0001 ***

## Within-Channel Premiums (percentage, FDR-corrected)
- Performative Labor: +56.0%, d=0.524, fdr_p=0.0004 ***
- Emotional Bait: +65.6%, d=0.751, fdr_p<0.0001 ***
- Narrative Conflict: +39.7%, d=0.511, fdr_p=0.0004 ***
- Challenge Format: +20.9%, d=0.320, fdr_p=0.0814 n.s.
- Commercial Content: -3.8%, d=-0.073, fdr_p=0.5156 n.s.
- Privacy Violation: +40.3%, d=0.650, fdr_p<0.0001 ***

## Within-Channel Robustness (overall exploit flag)
- Mean premium: +0.123 log10 views = +32.8%
- Channels with positive premium: 36/47 (76.6%)
- Wilcoxon p: 0.0011

## Same-Year Robustness (subset with valid dates, n=907)
- Mean premium: +0.160 log10 views = +44.4%
- Groups: 23, positive: 16/23
- Wilcoxon p: 0.0301

## Validation (3 annotators, 107 unique videos)
- Overall F1: 0.884
- Recall: 0.983
- Primary annotator (A): F1=0.920, κ=0.838
- Inter-rater agreement (A vs B overlap): 3/3 = 100%
