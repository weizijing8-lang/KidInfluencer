"""
Compute per-dimension validation metrics from Annotator B's per-dimension labels.
B's data format (tab-separated):
video_id, title, channel, url, views, perf_labor, emot_bait, narr_conflict, challenge, commercial, privacy, overall_label
"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Parse B's original annotations (pasted_content_7)
cols = ['video_id', 'title', 'channel', 'url', 'views', 
        'performative_labor', 'emotional_bait', 'narrative_conflict', 
        'challenge_format', 'commercial_content', 'privacy_violation', 'overall_label']

# Read B's original data
rows_b = []
with open('/home/ubuntu/upload/pasted_content_7.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            row = {}
            row['video_id'] = parts[0]
            row['title'] = parts[1]
            row['channel'] = parts[2]
            row['url'] = parts[3]
            row['views'] = parts[4]
            # Dimensions: positions 5-10, overall: position 11
            dims = parts[5:11] if len(parts) >= 11 else [''] * 6
            overall = parts[11] if len(parts) >= 12 else parts[-1]
            row['performative_labor'] = int(dims[0]) if dims[0].strip() in ['0','1'] else np.nan
            row['emotional_bait'] = int(dims[1]) if dims[1].strip() in ['0','1'] else np.nan
            row['narrative_conflict'] = int(dims[2]) if dims[2].strip() in ['0','1'] else np.nan
            row['challenge_format'] = int(dims[3]) if dims[3].strip() in ['0','1'] else np.nan
            row['commercial_content'] = int(dims[4]) if dims[4].strip() in ['0','1'] else np.nan
            row['privacy_violation'] = int(dims[5]) if dims[5].strip() in ['0','1'] else np.nan
            row['overall_label'] = int(overall) if overall.strip() in ['0','1'] else np.nan
            rows_b.append(row)

df_b = pd.DataFrame(rows_b)
print(f"Annotator B: {len(df_b)} videos total")

# Read B's updated annotations (pasted_content_9) - these override
rows_update = []
with open('/home/ubuntu/upload/pasted_content_9.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            row = {}
            row['video_id'] = parts[0]
            row['title'] = parts[1]
            row['channel'] = parts[2]
            row['url'] = parts[3]
            row['views'] = parts[4]
            dims = parts[5:11] if len(parts) >= 11 else [''] * 6
            overall = parts[11] if len(parts) >= 12 else parts[-1]
            row['performative_labor'] = int(dims[0]) if dims[0].strip() in ['0','1'] else np.nan
            row['emotional_bait'] = int(dims[1]) if dims[1].strip() in ['0','1'] else np.nan
            row['narrative_conflict'] = int(dims[2]) if dims[2].strip() in ['0','1'] else np.nan
            row['challenge_format'] = int(dims[3]) if dims[3].strip() in ['0','1'] else np.nan
            row['commercial_content'] = int(dims[4]) if dims[4].strip() in ['0','1'] else np.nan
            row['privacy_violation'] = int(dims[5]) if dims[5].strip() in ['0','1'] else np.nan
            row['overall_label'] = int(overall) if overall.strip() in ['0','1'] else np.nan
            rows_update.append(row)

df_update = pd.DataFrame(rows_update)
print(f"Annotator B updates: {len(df_update)} videos")

# Apply updates: replace rows in df_b with df_update where video_id matches
for _, upd_row in df_update.iterrows():
    mask = df_b['video_id'] == upd_row['video_id']
    if mask.any():
        for col in ['performative_labor', 'emotional_bait', 'narrative_conflict', 
                    'challenge_format', 'commercial_content', 'privacy_violation', 'overall_label']:
            df_b.loc[mask, col] = upd_row[col]
    else:
        df_b = pd.concat([df_b, pd.DataFrame([upd_row])], ignore_index=True)

# Also fix Candle Blowing Challenge (sS89FnQWwvA) - B verbally confirmed exploit
mask_candle = df_b['video_id'] == 'sS89FnQWwvA'
if mask_candle.any():
    df_b.loc[mask_candle, 'overall_label'] = 1

print(f"\nAfter updates: {len(df_b)} videos total")

# Filter to kid-centric channels only
adult_channels = [
    'blippi', 'crazygorilla', 'itsyeboi', 'itsrucka', 'dafuqboom',
    'mrbeast', 'sssniperwolf', 'markiplier', 'pewdiepie', 'dude perfect',
    'unspeakable', 'preston', 'brianna', 'zhong', 'ben azelart',
    'stokes twins', 'alan chikin chow', 'matt and abby', 'the royalty family gaming',
    'faze rug', 'carter sharer', 'rebecca zamolo', 'chad wild clay'
]

df_b_kid = df_b[~df_b['channel'].str.lower().isin([c.lower() for c in adult_channels])].copy()
print(f"Kid-centric only: {len(df_b_kid)} videos")

# Now load model predictions for these videos
snorkel_data = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv')
print(f"\nSnorkel data: {len(snorkel_data)} videos")

# Merge B's annotations with model predictions
merged = df_b_kid.merge(snorkel_data, left_on='video_id', right_on='id', how='inner')
print(f"Merged (B annotations with model): {len(merged)} videos")

# Define dimension columns
dimensions = ['performative_labor', 'emotional_bait', 'narrative_conflict', 
              'challenge_format', 'commercial_content', 'privacy_violation']

# Per-dimension validation
print("\n" + "="*70)
print("PER-DIMENSION VALIDATION (Annotator B vs Snorkel Model)")
print("Note: Snorkel -1 = ABSTAIN (excluded), blank human = 0 (not present)")
print("="*70)

results = []
for dim in dimensions:
    human_col = dim  # from B's annotations
    model_col = f"{dim}_pred"  # from Snorkel predictions
    
    # Get valid subset: human labeled AND model did not abstain
    subset = merged[['video_id', human_col, model_col]].copy()
    # Treat blank human labels as 0 (not present)
    subset[human_col] = subset[human_col].fillna(0).astype(int)
    # Exclude model abstains (-1)
    subset = subset[subset[model_col] != -1]
    
    if len(subset) == 0:
        print(f"\n{dim}: No valid annotations (all abstains)")
        continue
    
    y_true = subset[human_col].values
    y_pred = subset[model_col].astype(int).values
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    n_abstain = (merged[model_col] == -1).sum()
    
    print(f"\n{dim.upper()}:")
    print(f"  N={len(subset)} (abstains excluded: {n_abstain}), Prevalence={y_true.mean():.1%}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    
    results.append({
        'dimension': dim,
        'n': len(subset),
        'n_abstain': n_abstain,
        'prevalence': y_true.mean(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Dimension':<25} {'N':>4} {'Prev':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*70)
for r in results:
    print(f"{r['dimension']:<25} {r['n']:>4} {r['prevalence']:>6.1%} {r['accuracy']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

# Also check: INCLUDING abstains (treat abstain as 0 = not flagged)
print("\n" + "="*70)
print("INCLUDING ABSTAINS AS 0 (model abstain = not flagged)")
print("="*70)

results2 = []
for dim in dimensions:
    human_col = dim
    model_col = f"{dim}_pred"
    
    # Treat NaN in human labels as 0, treat model -1 as 0
    valid = merged[['video_id', human_col, model_col]].copy()
    valid[human_col] = valid[human_col].fillna(0).astype(int)
    valid[model_col] = valid[model_col].replace(-1, 0).astype(int)
    
    y_true = valid[human_col].values
    y_pred = valid[model_col].values
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    results2.append({
        'dimension': dim,
        'n': len(valid),
        'prevalence': y_true.mean(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

print(f"{'Dimension':<25} {'N':>4} {'Prev':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*70)
for r in results2:
    print(f"{r['dimension']:<25} {r['n']:>4} {r['prevalence']:>6.1%} {r['accuracy']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
