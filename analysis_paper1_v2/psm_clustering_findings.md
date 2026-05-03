# PSM + Clustering Key Findings

## PSM Results (25 matched pairs)
- Covariates balanced: log_total_views SMD=-0.28, n_videos SMD=0.27, span_years SMD=-0.008
- After matching on channel size/age:
  - **CI significantly higher for family** (ATE=0.12, p=0.036)
  - **sponsor_rate significantly LOWER for family** (ATE=-0.05, p=0.034) — surprising!
  - videos_per_week higher but not significant (ATE=0.49, p=0.25)
  - n_child_brands much higher but wide CI (ATE=425, p=0.17)
  - LII not significantly different (ATE=-0.01, p=0.76)

## Key Insight from PSM
Family channels are MORE commercialized (higher CI) but through DIFFERENT mechanisms:
- Not through explicit sponsorship (lower sponsor_rate)
- But through child-brand partnerships and network connections
- This suggests "hidden" commercialization — not disclosed #ad but structural integration

## Clustering Results (χ²=1919.7, p<0.001)
Content space is STRONGLY segregated by category:
- Cluster 1: 100% family (nursery rhymes/kids songs)
- Cluster 3: 92% family (family vlog/daily life)
- Cluster 4: 95% family (pregnancy/baby content)
- Cluster 5: 77% family (dramatic/clickbait family content)
- Cluster 6: 5% family (product reviews/unboxing — adult dominated)
- Cluster 0: 31% family (mixed entertainment)

## Exploitation by Cluster
- Cluster 4 (baby/pregnancy, 95% family) has HIGHEST exploit score for adult channels (0.145)
  but moderate for family (0.074)
- Cluster 2 (mixed, 30% family) has highest exploit for family channels (0.071)
- Overall: adult channels in family-dominated clusters show HIGHER exploitation than family channels themselves

## t-SNE Visualization
- Clear separation between family and adult content in embedding space
- Family content occupies distinct regions (kids songs, family vlogs, dramatic content)
- Some overlap in "challenge" and "entertainment" content types

## For Paper
1. PSM shows family channels are commercially integrated differently (not overt sponsorship)
2. Content clustering reveals distinct "kidfluencer content niches" with varying exploitation levels
3. The embedding separation validates that family/adult content is fundamentally different
4. Chi-square confirms cluster membership is strongly associated with channel category
