"""Analyze user's 50 new annotations (kid-centric channels only)"""
import pandas as pd
import numpy as np

# Read the tab-separated file
df = pd.read_csv('/home/ubuntu/upload/pasted_content_6.txt', sep='\t', header=None)

# Assign column names based on the annotation sheet structure
cols = ['video_id', 'youtube_link', 'title', 'channel', 'views', 'sample_group',
        'MODEL_performative_labor_prob', 'MODEL_performative_labor_pred',
        'MODEL_emotional_bait_prob', 'MODEL_emotional_bait_pred',
        'MODEL_narrative_conflict_prob', 'MODEL_narrative_conflict_pred',
        'MODEL_challenge_format_prob', 'MODEL_challenge_format_pred',
        'MODEL_commercial_content_prob', 'MODEL_commercial_content_pred',
        'MODEL_privacy_violation_prob', 'MODEL_privacy_violation_pred',
        'MODEL_exploitation_score', 'MODEL_overall_prediction', 'MODEL_confidence',
        'MODEL_n_dims_flagged', 'Human_Labeled', 'Note']

df.columns = cols[:len(df.columns)]
print(f'Total annotated videos: {len(df)}')
print()

# Extract model prediction and human label
df['model_pred'] = (df['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df['human_label'] = df['Human_Labeled'].astype(int)

# Overall accuracy
correct = (df['model_pred'] == df['human_label']).sum()
total = len(df)
accuracy = correct / total
print(f'{"="*60}')
print(f'  OVERALL METRICS (n={total})')
print(f'{"="*60}')
print(f'  Accuracy: {correct}/{total} = {accuracy:.1%}')
print()

# Confusion matrix
tp = ((df['model_pred'] == 1) & (df['human_label'] == 1)).sum()
fp = ((df['model_pred'] == 1) & (df['human_label'] == 0)).sum()
tn = ((df['model_pred'] == 0) & (df['human_label'] == 0)).sum()
fn = ((df['model_pred'] == 0) & (df['human_label'] == 1)).sum()

print(f'  Confusion Matrix:')
print(f'                  Human=Exploit  Human=Clean')
print(f'    Model=Exploit     TP={tp:<5}      FP={fp}')
print(f'    Model=Clean       FN={fn:<5}      TN={tn}')
print()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f'  Precision: {precision:.3f}')
print(f'  Recall:    {recall:.3f}')
print(f'  F1 Score:  {f1:.3f}')
print()

# Cohen's Kappa
po = accuracy
pe = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp)) / (total**2)
kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
print(f'  Cohen\'s Kappa: {kappa:.3f}')
print()

# By sample group
print(f'{"="*60}')
print(f'  BY SAMPLE GROUP')
print(f'{"="*60}')
for group in ['high_conf_exploit', 'high_conf_clean', 'low_confidence', 'random']:
    g = df[df['sample_group'] == group]
    if len(g) == 0:
        continue
    acc = (g['model_pred'] == g['human_label']).mean()
    n = len(g)
    tp_g = ((g['model_pred'] == 1) & (g['human_label'] == 1)).sum()
    fp_g = ((g['model_pred'] == 1) & (g['human_label'] == 0)).sum()
    tn_g = ((g['model_pred'] == 0) & (g['human_label'] == 0)).sum()
    fn_g = ((g['model_pred'] == 0) & (g['human_label'] == 1)).sum()
    print(f'  {group} (n={n}): accuracy={acc:.1%}  TP={tp_g} FP={fp_g} TN={tn_g} FN={fn_g}')

print()

# Confidence analysis
print(f'{"="*60}')
print(f'  CONFIDENCE ANALYSIS')
print(f'{"="*60}')
correct_mask = (df['model_pred'] == df['human_label'])
df['MODEL_confidence'] = df['MODEL_confidence'].astype(float)
print(f'  Mean confidence (correct predictions): {df[correct_mask]["MODEL_confidence"].mean():.3f}')
print(f'  Mean confidence (incorrect predictions): {df[~correct_mask]["MODEL_confidence"].mean():.3f}')
print()

# Error analysis
print(f'{"="*60}')
print(f'  ERROR ANALYSIS')
print(f'{"="*60}')
print(f'\n  False Positives (model=EXPLOIT, human=CLEAN):')
fps = df[(df['model_pred'] == 1) & (df['human_label'] == 0)]
for _, row in fps.iterrows():
    note = str(row.get('Note', ''))
    note_str = f' | Note: {note}' if note and note != 'nan' else ''
    print(f'    - [{row["channel"]}] {row["title"][:55]}{note_str}')

print(f'\n  False Negatives (model=CLEAN, human=EXPLOIT):')
fns = df[(df['model_pred'] == 0) & (df['human_label'] == 1)]
for _, row in fns.iterrows():
    note = str(row.get('Note', ''))
    note_str = f' | Note: {note}' if note and note != 'nan' else ''
    print(f'    - [{row["channel"]}] {row["title"][:55]}{note_str}')

# Summary for paper
print()
print(f'{"="*60}')
print(f'  SUMMARY FOR PAPER')
print(f'{"="*60}')
print(f'  n = {total} videos from {df["channel"].nunique()} kid-centric channels')
print(f'  Human prevalence of exploitation: {df["human_label"].mean():.1%}')
print(f'  Model prevalence of exploitation: {df["model_pred"].mean():.1%}')
print(f'  Accuracy = {accuracy:.1%}, Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}')
print(f'  Cohen\'s Kappa = {kappa:.3f}')
