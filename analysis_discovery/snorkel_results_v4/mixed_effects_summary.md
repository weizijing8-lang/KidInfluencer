# Mixed-Effects Regression Results

## Model 1: Overall Exploitation Score → Log Views

| Parameter | Value |
|-----------|-------|
| Coefficient (β) | 0.681 |
| 95% CI | [0.575, 0.786] |
| z-value | 12.613 |
| p-value | < 0.001 |
| N observations | 5,051 |
| N groups (channels) | 79 |
| Random effect variance | 0.763 |
| Residual scale | 0.272 |

**Interpretation:** A one-unit increase in exploitation score is associated with a 0.68 increase in log₁₀(views), controlling for channel-level variation. This translates to approximately 4.8x more views (10^0.681 ≈ 4.8).

## Model 2: Per-Dimension Fixed Effects

| Dimension | β | SE | z | p | Sig |
|-----------|---|-----|---|---|-----|
| performative_labor | 0.291 | 0.049 | 5.95 | 2.70e-09 | *** |
| emotional_bait | 0.205 | 0.048 | 4.28 | 1.87e-05 | *** |
| narrative_conflict | 0.061 | 0.041 | 1.46 | 0.144 | n.s. |
| challenge_format | -0.017 | 0.032 | -0.53 | 0.595 | n.s. |
| commercial_content | 0.005 | 0.042 | 0.12 | 0.906 | n.s. |
| privacy_violation | 0.316 | 0.062 | 5.08 | 3.71e-07 | *** |

**Key finding:** When controlling for all dimensions simultaneously:
- **performative_labor, emotional_bait, privacy_violation** remain highly significant
- **narrative_conflict, challenge_format, commercial_content** lose significance
- This suggests the Wilcoxon results for narrative_conflict and challenge_format may be partially driven by correlation with the three significant dimensions

## Effect Sizes (Cohen's d)

| Dimension | Cohen's d | Interpretation | Cliff's δ |
|-----------|-----------|----------------|-----------|
| performative_labor | 0.613 | **medium** | 0.339 |
| narrative_conflict | 0.359 | small | 0.214 |
| emotional_bait | 0.281 | small | 0.175 |
| challenge_format | 0.151 | small | 0.085 |
| commercial_content | 0.108 | small | 0.063 |
| privacy_violation | 0.092 | small | 0.053 |

**Note:** Performative labor has a medium effect size (d=0.61), the strongest among all dimensions.
