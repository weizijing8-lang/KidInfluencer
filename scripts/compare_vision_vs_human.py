"""
Compare GPT-4 Vision analysis results with user's human annotations.
Compute metrics at various thresholds.
"""
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, accuracy_score

# Load vision results
with open('/home/ubuntu/KidInfluencer/data/vision_analysis_23.json') as f:
    vision = json.load(f)

# Load user annotations
user = pd.read_csv('/home/ubuntu/upload/pasted_content.txt', sep='\t')
dim_cols = ['annotator_performative_labor', 'annotator_emotional_bait', 'annotator_narrative_conflict',
            'annotator_challenge_format', 'annotator_commercial_content', 'annotator_privacy_violation',
            'annotator_overall_exploitative']
for col in dim_cols:
    user[col] = pd.to_numeric(user[col], errors='coerce').fillna(0).astype(int)

# Merge
vision_df = pd.DataFrame(vision)
merged = user.merge(vision_df[['video_id', 'performative_labor', 'emotional_bait', 'narrative_conflict',
                                'challenge_format', 'commercial_content', 'privacy_violation', 
                                'overall_exploitative', 'reasoning']], on='video_id', how='inner')

print(f"Matched videos: {len(merged)}")
print()

# ============================================================
# OVERALL EXPLOITATIVE
# ============================================================
print("=" * 80)
print("OVERALL EXPLOITATIVE: GPT-4 Vision (title+thumbnail+desc) vs Human")
print("=" * 80)

# Try different thresholds
print(f"\n{'Thresh':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'κ':<8}")
print("─" * 48)

best_f1 = 0
best_thresh = 0.5

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_true = merged['annotator_overall_exploitative'].values
    y_pred = (merged['overall_exploitative'] >= thresh).astype(int).values
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"{thresh:<8} {acc:<8.3f} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} {kappa:<8.3f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nBest threshold: {best_thresh} (F1={best_f1:.3f})")

# ============================================================
# PER-DIMENSION COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("PER-DIMENSION COMPARISON (threshold=0.5)")
print("=" * 80)

dimensions = [
    ('performative_labor', 'annotator_performative_labor'),
    ('emotional_bait', 'annotator_emotional_bait'),
    ('narrative_conflict', 'annotator_narrative_conflict'),
    ('challenge_format', 'annotator_challenge_format'),
    ('commercial_content', 'annotator_commercial_content'),
    ('privacy_violation', 'annotator_privacy_violation'),
]

print(f"\n{'Dimension':<22} {'Acc':<7} {'F1':<7} {'Prec':<7} {'Rec':<7} {'κ':<7} {'H=1':<5} {'V=1':<5}")
print("─" * 70)

for vision_col, human_col in dimensions:
    y_true = merged[human_col].values
    y_pred = (merged[vision_col] >= 0.5).astype(int).values
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    h1 = y_true.sum()
    v1 = y_pred.sum()
    
    print(f"{vision_col:<22} {acc:<7.3f} {f1:<7.3f} {prec:<7.3f} {rec:<7.3f} {kappa:<7.3f} {h1:<5} {v1:<5}")

# ============================================================
# VIDEO-BY-VIDEO COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("VIDEO-BY-VIDEO: Overall Exploitative")
print("=" * 80)

print(f"\n{'#':<3} {'Title':<40} {'Human':<6} {'Vision':<7} {'Match'}")
print("─" * 65)

correct = 0
for idx, row in merged.iterrows():
    title = row['title'][:37] + "..." if len(row['title']) > 37 else row['title']
    h = row['annotator_overall_exploitative']
    v = row['overall_exploitative']
    v_bin = 1 if v >= best_thresh else 0
    match = "✅" if h == v_bin else "❌"
    if h == v_bin:
        correct += 1
    print(f"{idx+1:<3} {title:<40} {h:<6} {v:.2f}{'*' if v_bin != h else ' ':<3} {match}")

print(f"\nAccuracy at threshold {best_thresh}: {correct}/{len(merged)} = {correct/len(merged):.1%}")

# ============================================================
# COMPARISON WITH PREVIOUS METHODS
# ============================================================
print("\n" + "=" * 80)
print("METHOD COMPARISON SUMMARY")
print("=" * 80)

# Previous Snorkel results (from earlier analysis)
print(f"""
{'Method':<45} {'Acc':<8} {'F1':<8} {'κ':<8}
{'─' * 70}
{'Title keywords only (rule-based)':<45} {'0.583':<8} {'0.286':<8} {'0.102':<8}
{'Snorkel Label Model (title+thumb CV)':<45} {'0.696':<8} {'0.533':<8} {'0.309':<8}
{'Description-based LF (performative only)':<45} {'0.739':<8} {'0.750':<8} {'0.481':<8}
""")

# Add current vision result
y_true = merged['annotator_overall_exploitative'].values
y_pred = (merged['overall_exploitative'] >= best_thresh).astype(int).values
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"{'GPT-4 Vision (title+thumb+desc) overall':<45} {acc:<8.3f} {f1:<8.3f} {kappa:<8.3f}")

# Per-dimension best
y_true_p = merged['annotator_performative_labor'].values
y_pred_p = (merged['performative_labor'] >= 0.5).astype(int).values
f1_p = f1_score(y_true_p, y_pred_p, zero_division=0)
kappa_p = cohen_kappa_score(y_true_p, y_pred_p)
print(f"{'GPT-4 Vision performative_labor':<45} {accuracy_score(y_true_p, y_pred_p):<8.3f} {f1_p:<8.3f} {kappa_p:<8.3f}")

y_true_e = merged['annotator_emotional_bait'].values
y_pred_e = (merged['emotional_bait'] >= 0.5).astype(int).values
f1_e = f1_score(y_true_e, y_pred_e, zero_division=0)
kappa_e = cohen_kappa_score(y_true_e, y_pred_e)
print(f"{'GPT-4 Vision emotional_bait':<45} {accuracy_score(y_true_e, y_pred_e):<8.3f} {f1_e:<8.3f} {kappa_e:<8.3f}")

# ============================================================
# KEY DISAGREEMENTS ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("KEY DISAGREEMENTS (Human ≠ Vision at threshold 0.5)")
print("=" * 80)

for idx, row in merged.iterrows():
    h = row['annotator_overall_exploitative']
    v = row['overall_exploitative']
    v_bin = 1 if v >= 0.5 else 0
    if h != v_bin:
        print(f"\n  📹 {row['title'][:55]}")
        print(f"     Human={h}, Vision={v:.2f}")
        print(f"     Vision scores: perf={row['performative_labor']:.2f} emo={row['emotional_bait']:.2f} "
              f"conflict={row['narrative_conflict']:.2f} challenge={row['challenge_format']:.2f} "
              f"commercial={row['commercial_content']:.2f} privacy={row['privacy_violation']:.2f}")
        # Get reasoning
        for vr in vision:
            if vr['video_id'] == row['video_id']:
                print(f"     Reasoning: {vr.get('reasoning', 'N/A')[:120]}")
                break
