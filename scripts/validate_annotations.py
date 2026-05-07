"""
Compare user's human annotations with Snorkel pipeline predictions.
Compute accuracy, F1, Cohen's kappa per dimension and overall.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix

# Load user annotations
user_raw = pd.read_csv('/home/ubuntu/upload/pasted_content.txt', sep='\t')
print(f"User annotations loaded: {len(user_raw)} videos")
print(f"Columns: {user_raw.columns.tolist()}")
print()

# Clean up: replace empty/NaN with 0 for dimension columns
dim_cols = ['annotator_performative_labor', 'annotator_emotional_bait', 'annotator_narrative_conflict',
            'annotator_challenge_format', 'annotator_commercial_content', 'annotator_privacy_violation',
            'annotator_overall_exploitative']

for col in dim_cols:
    user_raw[col] = pd.to_numeric(user_raw[col], errors='coerce').fillna(0).astype(int)

print("User annotation summary:")
print(user_raw[dim_cols].sum())
print(f"\nOverall exploitative rate: {user_raw['annotator_overall_exploitative'].mean():.1%}")

# Load Snorkel predictions
snorkel_df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results/videos_with_exploitation_scores.csv')
print(f"\nSnorkel dataset: {len(snorkel_df)} videos")

# Merge
merged = user_raw.merge(
    snorkel_df[['id', 'performative', 'emotional_bait', 'narrative_conflict', 
                'challenge_format', 'commercial_content', 'exploitation_score', 'exploitation_pred']],
    left_on='video_id', right_on='id', how='left'
)

matched = merged[merged['id'].notna()].copy()
unmatched = merged[merged['id'].isna()].copy()

print(f"\nMatched with Snorkel: {len(matched)} videos")
print(f"Not in Snorkel sample: {len(unmatched)} videos")

if len(unmatched) > 0:
    print(f"\nUnmatched videos:")
    for _, row in unmatched.iterrows():
        print(f"  - {row['video_id']}: {row['title'][:50]}... ({row['channel']})")

# ============================================================
# OVERALL BINARY COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("OVERALL BINARY: Human 'overall_exploitative' vs Snorkel 'exploitation_pred'")
print("=" * 80)

if len(matched) > 0:
    y_human = matched['annotator_overall_exploitative'].values
    y_snorkel = matched['exploitation_pred'].values
    
    acc = accuracy_score(y_human, y_snorkel)
    f1 = f1_score(y_human, y_snorkel, zero_division=0)
    prec = precision_score(y_human, y_snorkel, zero_division=0)
    rec = recall_score(y_human, y_snorkel, zero_division=0)
    kappa = cohen_kappa_score(y_human, y_snorkel)
    
    cm = confusion_matrix(y_human, y_snorkel)
    
    print(f"\n  Accuracy:  {acc:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  Cohen's κ: {kappa:.3f}")
    print(f"\n  Confusion Matrix (rows=human, cols=snorkel):")
    print(f"              Snorkel=0  Snorkel=1")
    print(f"  Human=0     {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Human=1     {cm[1][0]:>5}      {cm[1][1]:>5}")

# ============================================================
# PER-DIMENSION COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("PER-DIMENSION COMPARISON")
print("=" * 80)

dim_mapping = {
    'performative': ('annotator_performative_labor', 'performative'),
    'emotional_bait': ('annotator_emotional_bait', 'emotional_bait'),
    'narrative_conflict': ('annotator_narrative_conflict', 'narrative_conflict'),
    'challenge_format': ('annotator_challenge_format', 'challenge_format'),
    'commercial_content': ('annotator_commercial_content', 'commercial_content'),
}

if len(matched) > 0:
    print(f"\n{'Dimension':<22} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'κ':<8} {'H=1':<6} {'S=1':<6}")
    print("─" * 75)
    
    for dim_name, (h_col, s_col) in dim_mapping.items():
        y_h = matched[h_col].values
        y_s = matched[s_col].values
        
        acc = accuracy_score(y_h, y_s)
        f1 = f1_score(y_h, y_s, zero_division=0)
        prec = precision_score(y_h, y_s, zero_division=0)
        rec = recall_score(y_h, y_s, zero_division=0)
        kappa = cohen_kappa_score(y_h, y_s)
        
        print(f"{dim_name:<22} {acc:.3f}   {f1:.3f}   {prec:.3f}   {rec:.3f}   {kappa:+.3f}  {y_h.sum():<6} {y_s.sum():<6}")

# ============================================================
# SCORE DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("SNORKEL SCORE DISTRIBUTION BY HUMAN LABEL")
print("=" * 80)

if len(matched) > 0:
    exploit_scores = matched[matched['annotator_overall_exploitative'] == 1]['exploitation_score']
    clean_scores = matched[matched['annotator_overall_exploitative'] == 0]['exploitation_score']
    
    print(f"\n  Human=EXPLOITATIVE (n={len(exploit_scores)}):")
    print(f"    Mean score: {exploit_scores.mean():.3f}")
    print(f"    Median:     {exploit_scores.median():.3f}")
    print(f"    Range:      [{exploit_scores.min():.3f}, {exploit_scores.max():.3f}]")
    
    print(f"\n  Human=CLEAN (n={len(clean_scores)}):")
    print(f"    Mean score: {clean_scores.mean():.3f}")
    print(f"    Median:     {clean_scores.median():.3f}")
    print(f"    Range:      [{clean_scores.min():.3f}, {clean_scores.max():.3f}]")
    
    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    if len(exploit_scores) > 0 and len(clean_scores) > 0:
        stat, p = mannwhitneyu(exploit_scores, clean_scores, alternative='greater')
        print(f"\n  Mann-Whitney U test (exploit > clean): U={stat:.0f}, p={p:.4f}")

# ============================================================
# DETAILED DISAGREEMENTS
# ============================================================
print("\n" + "=" * 80)
print("DISAGREEMENTS (Human vs Snorkel Overall)")
print("=" * 80)

if len(matched) > 0:
    disagree = matched[matched['annotator_overall_exploitative'] != matched['exploitation_pred']]
    print(f"\nTotal disagreements: {len(disagree)}/{len(matched)}")
    
    # False Positives (Snorkel says exploit, human says clean)
    fp = disagree[disagree['exploitation_pred'] == 1]
    print(f"\n  FALSE POSITIVES (Snorkel=EXPLOIT, Human=CLEAN): {len(fp)}")
    for _, row in fp.iterrows():
        print(f"    Score={row['exploitation_score']:.3f} | {row['title'][:60]}")
    
    # False Negatives (Snorkel says clean, human says exploit)
    fn = disagree[disagree['exploitation_pred'] == 0]
    print(f"\n  FALSE NEGATIVES (Snorkel=CLEAN, Human=EXPLOIT): {len(fn)}")
    for _, row in fn.iterrows():
        print(f"    Score={row['exploitation_score']:.3f} | {row['title'][:60]}")

# ============================================================
# ALL VIDEOS: FULL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 80)
print("FULL COMPARISON TABLE (all matched videos)")
print("=" * 80)

if len(matched) > 0:
    print(f"\n{'Title':<50} {'H_ov':<5} {'S_pred':<6} {'S_score':<8} {'Match'}")
    print("─" * 80)
    for _, row in matched.sort_values('exploitation_score').iterrows():
        title_short = row['title'][:47] + "..." if len(row['title']) > 47 else row['title']
        h = int(row['annotator_overall_exploitative'])
        s = int(row['exploitation_pred'])
        score = row['exploitation_score']
        match = "✅" if h == s else "❌"
        print(f"{title_short:<50} {h:<5} {s:<6} {score:.3f}   {match}")
