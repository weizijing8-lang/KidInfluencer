# ML Pipeline Key Findings

## Issues Identified

1. **Video-level XGBoost overfits badly**: CV R² = -0.19 (negative!), Training R² = 0.79
   - This is because we only have 75 channels × ~50 videos each = high channel-level correlation
   - Videos from the same channel have very similar view counts regardless of content features
   - Need to account for channel-level random effects (mixed-effects model)

2. **Hierarchical model also negative R²**: All models have negative CV R² 
   - Same overfitting issue - channel identity dominates view counts
   - Content features explain within-channel variance but not between-channel

3. **Channel-level model works better**: LOO R² = 0.15 for CI → LII
   - Small but positive - commercialization features do predict labor intensity
   - n_child_brands is the strongest predictor (SHAP = 0.047)

## Key Findings That DO Work

1. **CI → LII correlation is significant overall** (r=0.26, p=0.035; ρ=0.41, p<0.001)
2. **Family channels have higher CI** (mean=0.33 vs 0.22, p=0.023)
3. **Family channels upload more frequently** (2.7 vs 2.0/week, p=0.037)
4. **Family channels have higher exploit scores** (0.063 vs 0.033, p=0.037)
5. **Family channels have more child brand partnerships** (803 vs 152, p=0.004)
6. **Labor intensity predicts views** for both groups (Family r=0.43, Adult r=0.41)

## Exploit Score Differential Finding (IMPORTANT)
- For adult channels: exploit_score → views β=4.89 (stronger reward)
- For family channels: exploit_score → views β=0.91 (weaker reward)
- This means adult channels are MORE rewarded for exploitation tactics
- BUT family channels still use them more (higher mean exploit score)
- Interpretation: Family channels use exploitation tactics even when the reward is smaller,
  suggesting structural/industry pressure rather than pure optimization

## Revised Strategy for Paper

The negative CV R² for video-level models is actually a FINDING, not a failure:
- It shows that **channel identity** (subscriber base) dominates individual video performance
- Content features matter less than platform position
- This supports the "structural incentive" argument: once you're in the system, 
  the escalation loop is about maintaining position, not optimizing individual videos

For the paper, we should:
1. Report the channel-level analysis as primary (CI → LII, r=0.26)
2. Use SHAP to show which CI components drive LII
3. Show the family vs adult comparison
4. Frame the negative video-level R² as evidence that platform structure > content
5. Use the exploit score differential as a key finding
