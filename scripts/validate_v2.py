"""
Compare user's 23 human annotations against model predictions from annotation_sheet_full.csv
and LLM classifications from llm_classifications_v2.csv.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix

# ============================================================
# LOAD DATA
# ============================================================

# User annotations (23 videos)
user = pd.read_csv('/home/ubuntu/upload/pasted_content.txt', sep='\t')
dim_cols = ['annotator_performative_labor', 'annotator_emotional_bait', 'annotator_narrative_conflict',
            'annotator_challenge_format', 'annotator_commercial_content', 'annotator_privacy_violation',
            'annotator_overall_exploitative']
for col in dim_cols:
    user[col] = pd.to_numeric(user[col], errors='coerce').fillna(0).astype(int)

# Model predictions from annotation_sheet_full.csv (200-video sheet with model scores)
model_sheet = pd.read_csv('/home/ubuntu/KidInfluencer/data/annotation_sheet_full.csv')

# LLM classifications (full 4685 videos)
llm = pd.read_csv('/home/ubuntu/KidInfluencer/data/llm_classifications_v2.csv')

print(f"User annotations: {len(user)} videos")
print(f"Model sheet (200): {len(model_sheet)} videos")
print(f"LLM classifications: {len(llm)} videos")

# ============================================================
# MERGE: User annotations with model predictions
# ============================================================

merged = user.merge(
    model_sheet[['video_id', 'model_exploitation_score', 'model_performative', 
                 'model_emotional_bait', 'model_narrative_conflict', 'model_challenge_format',
                 'model_commercial_content', 'model_privacy_violation']],
    on='video_id', how='left'
)

# Also merge with LLM classifications (which cover all 4685 videos)
merged = merged.merge(
    llm[['id', 'performative_labor', 'emotional_bait', 'narrative_conflict', 
         'challenge_format', 'commercial_content', 'privacy_violation']],
    left_on='video_id', right_on='id', how='left', suffixes=('', '_llm')
)

has_model = merged['model_exploitation_score'].notna()
has_llm = merged['id'].notna()

print(f"\nMatched with model predictions: {has_model.sum()}")
print(f"Matched with LLM classifications: {has_llm.sum()}")

# ============================================================
# COMPARISON 1: Model exploitation score vs human overall
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON 1: Model Exploitation Score vs Human Overall Label")
print("=" * 80)

m = merged[has_model].copy()

if len(m) > 0:
    # The model gives a continuous score; we need a threshold
    # Try threshold = 0.5
    m['model_pred_05'] = (m['model_exploitation_score'] >= 0.5).astype(int)
    
    y_human = m['annotator_overall_exploitative'].values
    y_model = m['model_pred_05'].values
    
    acc = accuracy_score(y_human, y_model)
    f1 = f1_score(y_human, y_model, zero_division=0)
    prec = precision_score(y_human, y_model, zero_division=0)
    rec = recall_score(y_human, y_model, zero_division=0)
    kappa = cohen_kappa_score(y_human, y_model)
    cm = confusion_matrix(y_human, y_model)
    
    print(f"\n  Threshold = 0.5 | n = {len(m)}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  Cohen's κ: {kappa:.3f}")
    print(f"\n  Confusion Matrix (rows=human, cols=model):")
    print(f"              Model=0  Model=1")
    print(f"  Human=0     {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"  Human=1     {cm[1][0]:>5}    {cm[1][1]:>5}")
    
    # Try other thresholds
    print(f"\n  Threshold Sensitivity:")
    print(f"  {'Thresh':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'κ':<8}")
    print(f"  {'─' * 48}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (m['model_exploitation_score'] >= thresh).astype(int).values
        a = accuracy_score(y_human, pred)
        f = f1_score(y_human, pred, zero_division=0)
        p = precision_score(y_human, pred, zero_division=0)
        r = recall_score(y_human, pred, zero_division=0)
        k = cohen_kappa_score(y_human, pred)
        print(f"  {thresh:<8.1f} {a:<8.3f} {f:<8.3f} {p:<8.3f} {r:<8.3f} {k:<+8.3f}")

# ============================================================
# COMPARISON 2: LLM per-dimension vs human per-dimension
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON 2: LLM Per-Dimension Classifications vs Human")
print("=" * 80)

l = merged[has_llm].copy()

if len(l) > 0:
    dim_mapping = {
        'performative_labor': ('annotator_performative_labor', 'performative_labor'),
        'emotional_bait': ('annotator_emotional_bait', 'emotional_bait'),
        'narrative_conflict': ('annotator_narrative_conflict', 'narrative_conflict'),
        'challenge_format': ('annotator_challenge_format', 'challenge_format'),
        'commercial_content': ('annotator_commercial_content', 'commercial_content'),
        'privacy_violation': ('annotator_privacy_violation', 'privacy_violation'),
    }
    
    print(f"\n  n = {len(l)} videos")
    print(f"\n  {'Dimension':<22} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'κ':<8} {'H=1':<6} {'LLM=1':<6}")
    print(f"  {'─' * 75}")
    
    for dim_name, (h_col, llm_col) in dim_mapping.items():
        y_h = l[h_col].values
        y_l = l[llm_col].values
        
        acc = accuracy_score(y_h, y_l)
        f1 = f1_score(y_h, y_l, zero_division=0)
        prec = precision_score(y_h, y_l, zero_division=0)
        rec = recall_score(y_h, y_l, zero_division=0)
        kappa = cohen_kappa_score(y_h, y_l)
        
        print(f"  {dim_name:<22} {acc:<8.3f} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} {kappa:<+8.3f} {y_h.sum():<6} {y_l.sum():<6}")

# ============================================================
# COMPARISON 3: Model per-dimension vs human (from annotation_sheet_full)
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON 3: Model (Snorkel) Per-Dimension vs Human")
print("=" * 80)

if len(m) > 0:
    model_dim_mapping = {
        'performative': ('annotator_performative_labor', 'model_performative'),
        'emotional_bait': ('annotator_emotional_bait', 'model_emotional_bait'),
        'narrative_conflict': ('annotator_narrative_conflict', 'model_narrative_conflict'),
        'challenge_format': ('annotator_challenge_format', 'model_challenge_format'),
        'commercial_content': ('annotator_commercial_content', 'model_commercial_content'),
        'privacy_violation': ('annotator_privacy_violation', 'model_privacy_violation'),
    }
    
    print(f"\n  n = {len(m)} videos")
    print(f"\n  {'Dimension':<22} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'κ':<8} {'H=1':<6} {'M=1':<6}")
    print(f"  {'─' * 75}")
    
    for dim_name, (h_col, m_col) in model_dim_mapping.items():
        y_h = m[h_col].values
        y_m = m[m_col].values
        
        acc = accuracy_score(y_h, y_m)
        f1 = f1_score(y_h, y_m, zero_division=0)
        prec = precision_score(y_h, y_m, zero_division=0)
        rec = recall_score(y_h, y_m, zero_division=0)
        kappa = cohen_kappa_score(y_h, y_m)
        
        print(f"  {dim_name:<22} {acc:<8.3f} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} {kappa:<+8.3f} {y_h.sum():<6} {y_m.sum():<6}")

# ============================================================
# FULL VIDEO-BY-VIDEO TABLE
# ============================================================
print("\n" + "=" * 80)
print("FULL VIDEO-BY-VIDEO COMPARISON")
print("=" * 80)

print(f"\n{'#':<3} {'Title':<45} {'H_ov':<5} {'M_score':<8} {'M_pred':<7} {'Match'}")
print("─" * 80)

for i, row in merged.iterrows():
    title_short = row['title'][:42] + "..." if len(row['title']) > 42 else row['title']
    h = int(row['annotator_overall_exploitative'])
    
    if pd.notna(row.get('model_exploitation_score')):
        score = row['model_exploitation_score']
        pred = 1 if score >= 0.5 else 0
        match = "✅" if h == pred else "❌"
        print(f"{i+1:<3} {title_short:<45} {h:<5} {score:<8.3f} {pred:<7} {match}")
    else:
        print(f"{i+1:<3} {title_short:<45} {h:<5} {'N/A':<8} {'N/A':<7} {'—'}")

# ============================================================
# SCORE DISTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("SCORE DISTRIBUTION BY HUMAN LABEL")
print("=" * 80)

if len(m) > 0:
    exploit_scores = m[m['annotator_overall_exploitative'] == 1]['model_exploitation_score']
    clean_scores = m[m['annotator_overall_exploitative'] == 0]['model_exploitation_score']
    
    print(f"\n  Human=EXPLOITATIVE (n={len(exploit_scores)}):")
    if len(exploit_scores) > 0:
        print(f"    Mean: {exploit_scores.mean():.3f}, Median: {exploit_scores.median():.3f}")
        print(f"    Range: [{exploit_scores.min():.3f}, {exploit_scores.max():.3f}]")
    
    print(f"\n  Human=CLEAN (n={len(clean_scores)}):")
    if len(clean_scores) > 0:
        print(f"    Mean: {clean_scores.mean():.3f}, Median: {clean_scores.median():.3f}")
        print(f"    Range: [{clean_scores.min():.3f}, {clean_scores.max():.3f}]")
    
    from scipy.stats import mannwhitneyu
    if len(exploit_scores) > 0 and len(clean_scores) > 0:
        stat, p = mannwhitneyu(exploit_scores, clean_scores, alternative='greater')
        print(f"\n  Mann-Whitney U (exploit > clean): U={stat:.0f}, p={p:.4f}")
