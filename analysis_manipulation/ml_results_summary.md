# ML Classification Results: Manipulation Detection in Kidfluencer Titles

## Dataset
- 41,157 videos from family/kid channels
- 16.5% classified as manipulative (6,787 videos)
- 5 manipulation categories + NEUTRAL

## Binary Classification (Manipulative vs Neutral)

| Model | Accuracy | F1 (macro) |
|-------|----------|-----------|
| SGD-LR (TF-IDF word) | 0.9723 | 0.9481 |
| **SGD-LR (word+char)** | **0.9837** | **0.9701** |
| SGD-LR (full: word+char+hand) | 0.8243 | 0.7600 |
| SGD-LR (handcrafted only) | 0.5642 | 0.4887 |

**Key insight:** TF-IDF word+char achieves 98.4% accuracy and 0.97 F1. Adding handcrafted features HURTS performance (likely overfitting on small feature set with balanced class weights).

## Multi-class Classification (6 categories)

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| CHALLENGE_DARE | 0.95 | 0.91 | 0.93 | 3,371 |
| DECEPTION_NARRATIVE | 0.65 | 0.85 | 0.73 | 1,359 |
| EMOTIONAL_BAIT | 0.87 | 0.52 | 0.65 | 522 |
| FAKE_EMERGENCY | 0.94 | 0.72 | 0.82 | 511 |
| NEUTRAL | 0.97 | 0.98 | 0.98 | 34,370 |
| STAGED_CONFLICT | 0.94 | 0.72 | 0.81 | 1,024 |

**Overall: Acc=0.9556, F1_macro=0.8210, F1_weighted=0.9553**

## View Boost by Manipulation Category

| Category | Median Views | Boost vs Neutral | n |
|----------|-------------|-----------------|---|
| CHALLENGE_DARE | 1,776,111 | **+290%** | 3,371 |
| STAGED_CONFLICT | 1,200,890 | **+164%** | 1,024 |
| FAKE_EMERGENCY | 756,820 | +66% | 511 |
| DECEPTION_NARRATIVE | 554,735 | +22% | 1,359 |
| EMOTIONAL_BAIT | 411,978 | -10% | 522 |
| NEUTRAL (baseline) | 455,150 | — | 34,370 |

## Top Predictive Words

**For MANIPULATIVE:** challenge (15.4), secret (8.3), prank (6.8), caught (6.3), lost (6.1), don't (5.7), believe (5.2), trick (5.2), 24 hours (4.6), extreme (4.5), hospital (4.4), missing (4.3), broke (4.2), truth (4.1)

**For NEUTRAL:** cocomelon (-1.1), party (-1.1), baby (-1.1), wk (-1.1), new (-1.0), tour (-1.0), routine (-0.9), play (-0.9), christmas (-0.9), hair (-0.8)

## Channel-Level Analysis

| Channel | Videos | Manip Rate | Mean Views |
|---------|--------|-----------|-----------|
| Jordan Matter | 571 | 54.5% | 24.6M |
| Rebecca Zamolo | 1,170 | 48.0% | 8.7M |
| Piper Rockelle | 816 | 45.6% | 4.4M |
| itsyeboi | 587 | 41.2% | 675K |
| EhBee | 1,150 | 29.0% | 2.9M |
| ACE Family | 713 | 26.4% | 6.4M |
| Ryan's World | 3,685 | 25.5% | 17.2M |

**Spearman correlation (manipulation_rate vs mean_views): ρ=0.550, p=0.004**

## Key Findings for Paper

1. **Detection works:** Simple TF-IDF can detect manipulation in titles with 98% accuracy (binary) and 96% accuracy (multi-class)
2. **Platform rewards manipulation:** Challenge/Dare videos get +290% views, Staged Conflict gets +164%
3. **More manipulative channels are more successful:** ρ=0.550, p=0.004
4. **Manipulation is linguistically distinct:** Top words (challenge, prank, caught, secret) form clear semantic clusters
5. **Handcrafted features alone are insufficient:** Need actual word content, not just surface features (caps, length)
