# Key Findings from Full Analysis

## Figure Observations

### Fig 1 - Channel Drift Comparison
- Clear separation between family (red) and adult (blue) channels
- Family mean drift: 0.119, Adult mean drift: 0.015
- Only 1 adult channel (David Dobrik, 0.146) overlaps with family range
- All family channels are positive; many adult channels are negative

### Fig 2 - Drift Distribution
- t=154.3, p<0.001: Highly significant difference between groups
- Family distribution is right-skewed (long tail toward exploitation)
- Adult distribution is more symmetric around zero

### Fig 5 - Temporal Deep Dive (KEY FINDING)
- Cocomelon shows dramatic drift increase from 0.08 (2011) to 0.30 (2016-2022)
- Ryan's World peaked at 0.25 around 2017-2018
- Family Fun Pack shows gradual decline from 0.22 to 0.10 over time
- Adult channels (Casey Neistat, MrBeast) stay flat in [-0.10, 0.08] range
- Mark Wiens stays consistently negative (-0.10 to -0.05)

### Fig 6 - Event Study (CAUSAL RESULT)
- Family channels: slight upward bump after viral hit (t=0)
- Adult channels: completely flat, no response to viral hits
- The gap between family and adult is consistent (~0.11) throughout
- Effect is small but visible in family channels post-viral

## Statistical Results

### DiD Regression
- β(is_family) = 0.003272, p = 0.0848
- Direction correct: family channels drift MORE after viral hits
- Marginally significant (p < 0.10 but not < 0.05)
- 95% CI: [-0.000449, 0.006993]

### Potential improvements for significance:
1. Larger window (20 instead of 10)
2. Channel fixed effects
3. Continuous treatment intensity (log views)
4. Exclude channels with < 100 videos
5. Use "super viral" threshold (μ + 3σ)
