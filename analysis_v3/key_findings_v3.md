# Kidfluencer Analysis V3 — Key Findings (LLM-Annotated)

## Dataset
- 3,034 annotated videos across 75 channels (69.8% annotation success rate)
- 6 structured labels per video: content_type, emotional_manipulation, commercial_signals, child_role, privacy_concern, clickbait_level

## Prevalence of Concerning Patterns
| Metric | Rate |
|--------|------|
| Any emotional manipulation | 34.1% |
| Any commercial signals | 5.9% |
| Child as protagonist | 27.2% |
| Any privacy concern | 53.0% |
| Any clickbait | 65.9% |

## Regression Results (R² = 0.564)
| Variable | Coefficient | p-value | Interpretation |
|----------|------------|---------|----------------|
| log_subs | +0.70 | <0.001*** | Channel size (control) |
| commercial_binary | +0.44 | 0.001** | Commercial videos get 55% more views |
| clickbait_score | +0.27 | <0.001*** | Each clickbait level → 31% more views |
| child_protagonist | +0.20 | 0.002** | Child-centric videos get 22% more views |
| privacy_score | +0.17 | 0.016* | Privacy-exposing content gets more views |
| emotional_score | -0.005 | 0.944 | NOT significant after controlling for clickbait |
| duration_min | +0.02 | <0.001*** | Longer = more views (small effect) |

### Critical Insight:
**Emotional manipulation becomes non-significant when clickbait is in the model.** This means what we were calling "emotional manipulation" in V2 was actually capturing clickbait effects. The real story is:
1. **Clickbait** (sensational titles, caps, excessive punctuation) → strongly rewarded
2. **Commercial content** → strongly rewarded
3. **Child as protagonist** → rewarded
4. **Privacy exposure** → rewarded (borderline)
5. **Emotional manipulation per se** → NOT independently rewarded

## Platform Incentive (Within-Channel Analysis)
- 58 channels had both emotional and non-emotional content
- Mean ratio: emotional videos get 1.26x more views than non-emotional (same channel)
- BUT paired t-test is NOT significant (p=0.697)
- **Conclusion: The platform incentive operates through clickbait/commercial signals, not emotional manipulation specifically**

## Top Risk Channels (Data-Driven Weights)
1. Not Enough Nelsons (0.504) — high emotional + privacy + clickbait + child protagonist
2. JesssFam (0.486) — highest emotional + clickbait scores
3. CKN (0.484) — 100% commercial rate
4. Beast Family Vlogs (0.434) — high across all dimensions
5. Forever Family Vlogs (0.433)
6. Jordan Matter (0.419)
7. Ryan's World (0.400) — 78% child protagonist rate
8. Daily Bumps (0.385)
9. Piper Rockelle (0.373) — known controversial channel
10. The Bucket List Family (0.372)

## Methodological Improvements from V2
1. LLM annotation vs keyword matching: detected 34.1% emotional manipulation vs 2% before
2. Commercial detection: 5.9% vs 0.2% before (30x improvement)
3. New dimensions: privacy concern, clickbait, child role — all significant predictors
4. Risk weights now data-driven (from regression coefficients) instead of arbitrary
5. Within-channel analysis provides causal-adjacent evidence

## Remaining Weaknesses
1. Within-channel paired test is non-significant — need more data or better design
2. Still no temporal analysis (trends over time)
3. 75 channels is still small for a publication
4. No comment/engagement data beyond views
5. LLM annotations need inter-rater reliability validation
