# Pilot Study: Key Findings from Embedding Analysis

## Mean Exploitation Drift Score by Channel

| Channel | Type | N Videos | Mean Drift | Std |
|---------|------|----------|-----------|-----|
| The ACE Family | Family | 712 | **0.1135** | 0.1147 |
| Casey Neistat | Control | 1119 | 0.0614 | 0.0888 |
| Bratayley | Family | 2602 | 0.0163 | 0.0795 |
| Family Fun Pack | Family | 2806 | 0.0067 | 0.1104 |
| Ryan's World | Family | 2992 | -0.0037 | 0.0829 |
| Mark Wiens | Control | 1553 | **-0.0813** | 0.0659 |

## Initial Observations

1. **ACE Family has the highest drift score (0.1135)** - consistent with their known clickbait/drama-heavy content
2. **Mark Wiens (food vlogger) has the lowest (-0.0813)** - makes sense, food content is far from exploitation
3. **Ryan's World is near zero** - their content is mostly toy reviews, not drama-based
4. **Casey Neistat is surprisingly high (0.0614)** - his clickbait-style titles may be triggering this
5. **View count data is missing for some channels** (ACE Family, Ryan's World, Family Fun Pack show median=0)

## Issues to Address
- View count data is 0 for some channels (flat-playlist mode doesn't always get view counts)
- Need to get actual view counts for viral video analysis
- The direction vector seems to work! ACE Family (known for exploitation) > Bratayley > Ryan's World
