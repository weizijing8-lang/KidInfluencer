"""Analyze user's human annotations vs model predictions"""
import pandas as pd
import numpy as np

# Read the tab-separated file
df = pd.read_csv('/home/ubuntu/upload/pasted_content_5.txt', sep='\t')
print(f'Total annotated videos: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()

# Extract model prediction and human label
df['model_pred'] = (df['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df['human_label'] = df['Human Labeled'].astype(int)

# Overall accuracy
correct = (df['model_pred'] == df['human_label']).sum()
total = len(df)
accuracy = correct / total
print(f'=== OVERALL METRICS (n={total}) ===')
print(f'Accuracy: {correct}/{total} = {accuracy:.1%}')
print()

# Confusion matrix
tp = ((df['model_pred'] == 1) & (df['human_label'] == 1)).sum()
fp = ((df['model_pred'] == 1) & (df['human_label'] == 0)).sum()
tn = ((df['model_pred'] == 0) & (df['human_label'] == 0)).sum()
fn = ((df['model_pred'] == 0) & (df['human_label'] == 1)).sum()

print(f'Confusion Matrix:')
print(f'  TP={tp}, FP={fp}')
print(f'  FN={fn}, TN={tn}')
print()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}')
print()

# By sample group
print('=== BY SAMPLE GROUP ===')
for group in df['sample_group'].unique():
    g = df[df['sample_group'] == group]
    acc = (g['model_pred'] == g['human_label']).mean()
    print(f'  {group}: accuracy={acc:.1%} (n={len(g)})')
print()

# Error analysis
print('=== ERROR ANALYSIS ===')
print('\nFalse Positives (model says EXPLOIT, human says CLEAN):')
fps = df[(df['model_pred'] == 1) & (df['human_label'] == 0)]
for _, row in fps.iterrows():
    note = row.get('Note', '')
    print(f'  - [{row["channel"]}] {row["title"][:60]}')
    if note and str(note) != 'nan':
        print(f'    Note: {note}')

print(f'\nFalse Negatives (model says CLEAN, human says EXPLOIT):')
fns = df[(df['model_pred'] == 0) & (df['human_label'] == 1)]
for _, row in fns.iterrows():
    note = row.get('Note', '')
    print(f'  - [{row["channel"]}] {row["title"][:60]}')
    if note and str(note) != 'nan':
        print(f'    Note: {note}')

# Key insight: what patterns do FPs share?
print('\n=== KEY PATTERNS IN ERRORS ===')
print(f'\nFP pattern: Model flags as exploit but human says clean')
print(f'  Common theme in notes: {[str(row.get("Note","")) for _, row in fps.iterrows() if str(row.get("Note","")) != "nan"]}')
print(f'\n  Model confidence for FPs: mean={fps["MODEL_confidence"].astype(float).mean():.3f}')
print(f'  Model confidence for TPs: mean={df[(df["model_pred"]==1)&(df["human_label"]==1)]["MODEL_confidence"].astype(float).mean():.3f}')
