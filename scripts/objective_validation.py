"""
Objective Validation: Compare LLM annotations against hard/verifiable labels.
This validates LLM annotation quality without needing human annotators.

Validation strategies:
1. Commercial signals: Compare LLM's commercial_binary with presence of #ad/#sponsored in description
2. Clickbait features: Compare LLM's clickbait_level with objective title features (ALL CAPS, !, ?)
3. Content type: Compare LLM's content_type with video metadata patterns
4. Self-consistency: Run LLM on same videos twice and measure agreement
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import re
import os

# Load data
am = pd.read_csv('data/annotations_merged.csv')
vids = pd.read_csv('data/combined_videos.csv')

# Merge
merged = am.merge(vids, on='video_id', how='inner')
valid = merged[merged['emotional_manipulation'] != 'error'].copy()
print(f"Valid annotations for validation: {len(valid)}")

os.makedirs('annotation_task/validation_results', exist_ok=True)

# ============================================================
# VALIDATION 1: Commercial Signals vs Hard Labels
# ============================================================
print("\n" + "="*60)
print("VALIDATION 1: Commercial Signals")
print("="*60)

# Hard label: is_commercial from combined_videos (based on description keywords)
valid['hard_commercial'] = valid['is_commercial'].astype(bool)
valid['llm_commercial'] = valid['commercial_signals'].isin(['brand_mention', 'sponsored', 'product_placement'])

# Agreement
agree_commercial = (valid['hard_commercial'] == valid['llm_commercial']).mean()
print(f"Agreement (LLM vs hard label): {agree_commercial:.3f}")

# Confusion matrix
cm = confusion_matrix(valid['hard_commercial'], valid['llm_commercial'])
print(f"Confusion matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Cohen's Kappa
try:
    kappa = cohen_kappa_score(valid['hard_commercial'], valid['llm_commercial'])
    print(f"Cohen's Kappa: {kappa:.3f}")
except:
    kappa = None
    print("Could not compute Kappa (possibly no variance)")

# ============================================================
# VALIDATION 2: Clickbait Level vs Objective Title Features
# ============================================================
print("\n" + "="*60)
print("VALIDATION 2: Clickbait Level vs Title Features")
print("="*60)

# Compute objective clickbait indicators from title
def compute_title_clickbait_score(title):
    if pd.isna(title):
        return 0
    title = str(title)
    score = 0
    # ALL CAPS words (more than 2 consecutive caps words)
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', title))
    if caps_words >= 3:
        score += 2
    elif caps_words >= 1:
        score += 1
    # Exclamation marks
    excl = title.count('!')
    if excl >= 3:
        score += 2
    elif excl >= 1:
        score += 1
    # Question marks (curiosity gap)
    if '?' in title:
        score += 1
    # Emoji
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]")
    if emoji_pattern.search(title):
        score += 1
    # Clickbait phrases
    clickbait_phrases = ['you won\'t believe', 'gone wrong', 'not clickbait', 'omg', 'shocking', 'exposed']
    if any(p in title.lower() for p in clickbait_phrases):
        score += 2
    return min(score, 6)  # cap at 6

valid['objective_clickbait_score'] = valid['title'].apply(compute_title_clickbait_score)

# Map LLM clickbait to numeric
clickbait_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
valid['llm_clickbait_numeric'] = valid['clickbait_level'].map(clickbait_map)

# Correlation
corr = valid[['objective_clickbait_score', 'llm_clickbait_numeric']].dropna().corr().iloc[0,1]
print(f"Pearson correlation (objective score vs LLM score): {corr:.3f}")

# Spearman
from scipy.stats import spearmanr
valid_cb = valid[['objective_clickbait_score', 'llm_clickbait_numeric']].dropna()
rho, p = spearmanr(valid_cb['objective_clickbait_score'], valid_cb['llm_clickbait_numeric'])
print(f"Spearman rho: {rho:.3f}, p={p:.2e}")

# Binary comparison: objective_score >= 3 vs LLM moderate/severe
valid['obj_high_clickbait'] = valid['objective_clickbait_score'] >= 3
valid['llm_high_clickbait'] = valid['clickbait_level'].isin(['moderate', 'severe'])
agree_clickbait = (valid['obj_high_clickbait'] == valid['llm_high_clickbait']).mean()
kappa_clickbait = cohen_kappa_score(valid['obj_high_clickbait'], valid['llm_high_clickbait'])
print(f"Binary agreement (high clickbait): {agree_clickbait:.3f}")
print(f"Cohen's Kappa (high clickbait): {kappa_clickbait:.3f}")

# ============================================================
# VALIDATION 3: Emotional Title (has_emotional_title) vs LLM emotional_manipulation
# ============================================================
print("\n" + "="*60)
print("VALIDATION 3: Emotional Title Flag vs LLM Emotional Manipulation")
print("="*60)

valid['hard_emotional'] = valid['has_emotional_title'].astype(bool)
valid['llm_emotional'] = valid['emotional_manipulation'].isin(['moderate', 'severe'])

agree_emotional = (valid['hard_emotional'] == valid['llm_emotional']).mean()
kappa_emotional = cohen_kappa_score(valid['hard_emotional'], valid['llm_emotional'])
print(f"Agreement: {agree_emotional:.3f}")
print(f"Cohen's Kappa: {kappa_emotional:.3f}")

# Also check: does LLM emotional != none correlate with has_emotional_title?
valid['llm_any_emotional'] = valid['emotional_manipulation'] != 'none'
agree_any = (valid['hard_emotional'] == valid['llm_any_emotional']).mean()
kappa_any = cohen_kappa_score(valid['hard_emotional'], valid['llm_any_emotional'])
print(f"\nRelaxed (any emotional vs title flag):")
print(f"Agreement: {agree_any:.3f}")
print(f"Cohen's Kappa: {kappa_any:.3f}")

# ============================================================
# VALIDATION 4: Cross-dimension consistency
# ============================================================
print("\n" + "="*60)
print("VALIDATION 4: Cross-dimension Logical Consistency")
print("="*60)

# If child_role == 'absent', emotional_manipulation should be 'none' (no child to manipulate)
absent_mask = valid['child_role'] == 'absent'
if absent_mask.sum() > 0:
    absent_emotional = valid.loc[absent_mask, 'emotional_manipulation']
    pct_none = (absent_emotional == 'none').mean()
    print(f"When child_role='absent', emotional_manipulation='none': {pct_none:.1%} ({absent_mask.sum()} videos)")
else:
    print("No videos with child_role='absent'")

# If clickbait_level == 'severe', we expect more emotional_manipulation
severe_cb = valid[valid['clickbait_level'] == 'severe']
if len(severe_cb) > 0:
    pct_emotional = (severe_cb['emotional_manipulation'].isin(['moderate', 'severe'])).mean()
    print(f"When clickbait='severe', emotional_manipulation is moderate/severe: {pct_emotional:.1%} ({len(severe_cb)} videos)")

# If commercial_signals != 'none', check if clickbait is higher
commercial_vids = valid[valid['commercial_signals'] != 'none']
non_commercial = valid[valid['commercial_signals'] == 'none']
if len(commercial_vids) > 0:
    comm_clickbait = commercial_vids['llm_clickbait_numeric'].mean()
    non_comm_clickbait = non_commercial['llm_clickbait_numeric'].mean()
    print(f"Mean clickbait score - commercial: {comm_clickbait:.2f}, non-commercial: {non_comm_clickbait:.2f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY: LLM Annotation Validation Results")
print("="*60)

results = {
    'Validation': ['Commercial (binary)', 'Clickbait (Spearman)', 'Clickbait (binary Kappa)', 
                   'Emotional (strict Kappa)', 'Emotional (relaxed Kappa)'],
    'Metric': ['Agreement', 'Spearman ρ', "Cohen's κ", "Cohen's κ", "Cohen's κ"],
    'Value': [f"{agree_commercial:.3f}", f"{rho:.3f}", f"{kappa_clickbait:.3f}",
              f"{kappa_emotional:.3f}", f"{kappa_any:.3f}"],
    'Interpretation': [
        'Good' if agree_commercial > 0.8 else 'Fair' if agree_commercial > 0.6 else 'Poor',
        'Good' if rho > 0.5 else 'Fair' if rho > 0.3 else 'Poor',
        'Good' if kappa_clickbait > 0.4 else 'Fair' if kappa_clickbait > 0.2 else 'Poor',
        'Good' if kappa_emotional > 0.4 else 'Fair' if kappa_emotional > 0.2 else 'Poor',
        'Good' if kappa_any > 0.4 else 'Fair' if kappa_any > 0.2 else 'Poor'
    ]
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv('annotation_task/validation_results/objective_validation_summary.csv', index=False)
print("\nResults saved to annotation_task/validation_results/")
