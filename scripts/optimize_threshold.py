"""
Optimize threshold and compute alternative metrics for paper validation.
Computes: AUC-ROC, AUC-PR, optimal threshold (Youden's J), 
rank correlation, top-k precision, and per-dimension analysis.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             roc_curve, precision_recall_curve,
                             f1_score, cohen_kappa_score, accuracy_score,
                             confusion_matrix)
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import json

# Load data
human = pd.read_csv('/home/ubuntu/KidInfluencer/data/human_annotations_50.tsv', sep='\t')
merged_data = pd.read_csv('/home/ubuntu/KidInfluencer/data/merged_llm_vision.csv')
merged_data = merged_data.drop_duplicates(subset='video_id', keep='first')

# Merge
merged = human.merge(merged_data, on='video_id', how='left', suffixes=('', '_pipe'))
merged = merged.dropna(subset=['overall_exploitative'])
print(f"Matched videos: {len(merged)}")

# Human labels
y_true = merged['annotator_overall_exploitative'].values.astype(int)

# Pipeline continuous scores - compute from dimensions
dim_vision_cols = ['performative_labor_vision', 'emotional_bait_vision', 
                   'narrative_conflict_vision', 'challenge_format_vision',
                   'commercial_content_vision', 'privacy_violation_vision']
dim_llm_cols = ['performative_labor_llm', 'emotional_bait_llm',
                'narrative_conflict_llm', 'challenge_format_llm', 
                'commercial_content_llm', 'privacy_violation_llm']

# Compute combined score per dimension (0.67*vision + 0.33*llm)
combined_dims = []
for v_col, l_col in zip(dim_vision_cols, dim_llm_cols):
    v = pd.to_numeric(merged[v_col], errors='coerce').fillna(0).values
    l = pd.to_numeric(merged[l_col], errors='coerce').fillna(0).values
    combined_dims.append(0.67 * v + 0.33 * l)

# Overall score = mean of combined dimensions
y_score = np.mean(combined_dims, axis=0)

# Also use the pipeline's overall_exploitative directly
y_pipe = pd.to_numeric(merged['overall_exploitative'], errors='coerce').fillna(0).values

print(f"\n{'='*70}")
print("1. AUC-ROC AND AUC-PR ANALYSIS")
print(f"{'='*70}")

# AUC-ROC
auc_roc = roc_auc_score(y_true, y_score)
auc_roc_pipe = roc_auc_score(y_true, y_pipe)
print(f"AUC-ROC (combined dim scores): {auc_roc:.3f}")
print(f"AUC-ROC (pipeline overall):    {auc_roc_pipe:.3f}")

# AUC-PR
auc_pr = average_precision_score(y_true, y_score)
auc_pr_pipe = average_precision_score(y_true, y_pipe)
print(f"AUC-PR (combined dim scores):  {auc_pr:.3f}")
print(f"AUC-PR (pipeline overall):     {auc_pr_pipe:.3f}")

print(f"\n{'='*70}")
print("2. OPTIMAL THRESHOLD ANALYSIS (Youden's J)")
print(f"{'='*70}")

# ROC curve and Youden's J
fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold_j = thresholds_roc[best_idx]
print(f"Optimal threshold (Youden's J): {best_threshold_j:.3f}")
print(f"  TPR at optimal: {tpr[best_idx]:.3f}")
print(f"  FPR at optimal: {fpr[best_idx]:.3f}")

# F1-optimal threshold
thresholds_to_try = np.arange(0.1, 0.9, 0.01)
f1_scores = []
kappa_scores = []
acc_scores = []
for t in thresholds_to_try:
    pred = (y_score >= t).astype(int)
    f1_scores.append(f1_score(y_true, pred, zero_division=0))
    kappa_scores.append(cohen_kappa_score(y_true, pred))
    acc_scores.append(accuracy_score(y_true, pred))

best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_to_try[best_f1_idx]
best_kappa_idx = np.argmax(kappa_scores)
best_kappa_threshold = thresholds_to_try[best_kappa_idx]

print(f"\nOptimal threshold (max F1):    {best_f1_threshold:.3f}")
print(f"  F1 at optimal: {f1_scores[best_f1_idx]:.3f}")
print(f"  κ at optimal:  {kappa_scores[best_f1_idx]:.3f}")
print(f"  Acc at optimal: {acc_scores[best_f1_idx]:.3f}")

print(f"\nOptimal threshold (max κ):     {best_kappa_threshold:.3f}")
print(f"  κ at optimal:  {kappa_scores[best_kappa_idx]:.3f}")
print(f"  F1 at optimal: {f1_scores[best_kappa_idx]:.3f}")
print(f"  Acc at optimal: {acc_scores[best_kappa_idx]:.3f}")

# Compare with default 0.5
pred_05 = (y_score >= 0.5).astype(int)
print(f"\nAt default threshold 0.50:")
print(f"  F1:  {f1_score(y_true, pred_05):.3f}")
print(f"  κ:   {cohen_kappa_score(y_true, pred_05):.3f}")
print(f"  Acc: {accuracy_score(y_true, pred_05):.3f}")

# At optimal threshold
pred_opt = (y_score >= best_f1_threshold).astype(int)
cm = confusion_matrix(y_true, pred_opt)
print(f"\nConfusion matrix at optimal threshold ({best_f1_threshold:.2f}):")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print(f"\n{'='*70}")
print("3. RANK CORRELATION (Spearman & Kendall)")
print(f"{'='*70}")

rho, p_rho = spearmanr(y_true, y_score)
tau, p_tau = kendalltau(y_true, y_score)
print(f"Spearman ρ: {rho:.3f} (p={p_rho:.4f})")
print(f"Kendall τ:  {tau:.3f} (p={p_tau:.4f})")

print(f"\n{'='*70}")
print("4. TOP-K PRECISION")
print(f"{'='*70}")

# Sort by pipeline score descending
sorted_idx = np.argsort(-y_score)
for k_pct in [10, 20, 25, 30, 50]:
    k = max(1, int(len(y_true) * k_pct / 100))
    top_k_idx = sorted_idx[:k]
    top_k_precision = y_true[top_k_idx].mean()
    print(f"Top-{k_pct}% precision (n={k}): {top_k_precision:.3f}")

# Bottom-k (should be mostly clean)
for k_pct in [10, 20, 25, 30, 50]:
    k = max(1, int(len(y_true) * k_pct / 100))
    bottom_k_idx = sorted_idx[-k:]
    bottom_k_clean = (1 - y_true[bottom_k_idx]).mean()
    print(f"Bottom-{k_pct}% clean rate (n={k}): {bottom_k_clean:.3f}")

print(f"\n{'='*70}")
print("5. PER-DIMENSION AUC-ROC")
print(f"{'='*70}")

dim_names = ['performative_labor', 'emotional_bait', 'narrative_conflict',
             'challenge_format', 'commercial_content', 'privacy_violation']
human_dim_cols = ['annotator_performative_labor', 'annotator_emotional_bait',
                  'annotator_narrative_conflict', 'annotator_challenge_format',
                  'annotator_commercial_content', 'annotator_privacy_violation']

for dim, h_col, combined in zip(dim_names, human_dim_cols, combined_dims):
    h_raw = pd.to_numeric(merged[h_col], errors='coerce').fillna(0).values
    h = h_raw.astype(int)
    n_pos = int(h.sum())
    n_neg = int((h == 0).sum())
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(h, combined)
        rho_d, p_d = spearmanr(h, combined)
        # Optimal threshold for this dimension
        f1s = [f1_score(h, (combined >= t).astype(int), zero_division=0) for t in thresholds_to_try]
        best_t = thresholds_to_try[np.argmax(f1s)]
        best_f1 = max(f1s)
        kappa_at_best = cohen_kappa_score(h, (combined >= best_t).astype(int))
        print(f"  {dim}:")
        print(f"    AUC-ROC={auc:.3f}, Spearman={rho_d:.3f}, Best F1={best_f1:.3f} (t={best_t:.2f}), κ={kappa_at_best:.3f}")
        print(f"    Human pos: {n_pos}/{len(h)}")
    else:
        print(f"  {dim}: SKIPPED (pos={n_pos}, neg={n_neg})")

print(f"\n{'='*70}")
print("6. ALTERNATIVE: EXCLUDE EMOTIONAL_BAIT FROM OVERALL SCORE")
print(f"{'='*70}")

# What if we exclude emotional_bait (the weakest dimension) from the overall score?
# Use only: performative, narrative, challenge, commercial, privacy
strong_dims = [0, 2, 3, 4, 5]  # exclude emotional_bait (index 1)
y_score_no_eb = np.mean([combined_dims[i] for i in strong_dims], axis=0)
auc_no_eb = roc_auc_score(y_true, y_score_no_eb)
f1s_no_eb = [f1_score(y_true, (y_score_no_eb >= t).astype(int), zero_division=0) for t in thresholds_to_try]
best_t_no_eb = thresholds_to_try[np.argmax(f1s_no_eb)]
pred_no_eb = (y_score_no_eb >= best_t_no_eb).astype(int)
kappa_no_eb = cohen_kappa_score(y_true, pred_no_eb)
print(f"Without emotional_bait:")
print(f"  AUC-ROC: {auc_no_eb:.3f}")
print(f"  Best F1: {max(f1s_no_eb):.3f} (t={best_t_no_eb:.2f})")
print(f"  κ at best: {kappa_no_eb:.3f}")

# What about using only performative + privacy (the two strongest)?
y_score_2dim = np.mean([combined_dims[0], combined_dims[5]], axis=0)
if y_true.sum() > 0 and y_true.sum() < len(y_true):
    auc_2dim = roc_auc_score(y_true, y_score_2dim)
    f1s_2dim = [f1_score(y_true, (y_score_2dim >= t).astype(int), zero_division=0) for t in thresholds_to_try]
    best_t_2dim = thresholds_to_try[np.argmax(f1s_2dim)]
    kappa_2dim = cohen_kappa_score(y_true, (y_score_2dim >= best_t_2dim).astype(int))
    print(f"\nPerformative + Privacy only:")
    print(f"  AUC-ROC: {auc_2dim:.3f}")
    print(f"  Best F1: {max(f1s_2dim):.3f} (t={best_t_2dim:.2f})")
    print(f"  κ at best: {kappa_2dim:.3f}")

print(f"\n{'='*70}")
print("7. VISION-ONLY vs LLM-ONLY vs COMBINED")
print(f"{'='*70}")

# Vision only
v_scores = np.mean([pd.to_numeric(merged[c], errors='coerce').fillna(0).values for c in dim_vision_cols], axis=0)
l_scores = np.mean([pd.to_numeric(merged[c], errors='coerce').fillna(0).values for c in dim_llm_cols], axis=0)

auc_v = roc_auc_score(y_true, v_scores)
auc_l = roc_auc_score(y_true, l_scores)
auc_c = roc_auc_score(y_true, y_score)

# Best κ for each
f1s_v = [f1_score(y_true, (v_scores >= t).astype(int), zero_division=0) for t in thresholds_to_try]
f1s_l = [f1_score(y_true, (l_scores >= t).astype(int), zero_division=0) for t in thresholds_to_try]

best_t_v = thresholds_to_try[np.argmax(f1s_v)]
best_t_l = thresholds_to_try[np.argmax(f1s_l)]

kappa_v = cohen_kappa_score(y_true, (v_scores >= best_t_v).astype(int))
kappa_l = cohen_kappa_score(y_true, (l_scores >= best_t_l).astype(int))
kappa_c = cohen_kappa_score(y_true, (y_score >= best_f1_threshold).astype(int))

print(f"{'Method':<20} {'AUC-ROC':<10} {'Best F1':<10} {'Best κ':<10} {'Threshold':<10}")
print(f"{'-'*60}")
print(f"{'Vision only':<20} {auc_v:<10.3f} {max(f1s_v):<10.3f} {kappa_v:<10.3f} {best_t_v:<10.2f}")
print(f"{'LLM only':<20} {auc_l:<10.3f} {max(f1s_l):<10.3f} {kappa_l:<10.3f} {best_t_l:<10.2f}")
print(f"{'Combined (0.67+0.33)':<20} {auc_c:<10.3f} {max(f1_scores):<10.3f} {kappa_c:<10.3f} {best_f1_threshold:<10.2f}")

# Generate figures
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROC curve
ax = axes[0, 0]
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Combined (AUC={auc_roc:.3f})')
fpr_v, tpr_v, _ = roc_curve(y_true, v_scores)
fpr_l, tpr_l, _ = roc_curve(y_true, l_scores)
ax.plot(fpr_v, tpr_v, 'g--', linewidth=1.5, label=f'Vision only (AUC={auc_v:.3f})')
ax.plot(fpr_l, tpr_l, 'r--', linewidth=1.5, label=f'LLM only (AUC={auc_l:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: Pipeline vs Human Annotations')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Threshold vs metrics
ax = axes[0, 1]
ax.plot(thresholds_to_try, f1_scores, 'b-', linewidth=2, label='F1')
ax.plot(thresholds_to_try, kappa_scores, 'r-', linewidth=2, label="Cohen's κ")
ax.plot(thresholds_to_try, acc_scores, 'g-', linewidth=2, label='Accuracy')
ax.axvline(x=best_f1_threshold, color='b', linestyle='--', alpha=0.5, label=f'Best F1 (t={best_f1_threshold:.2f})')
ax.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default (t=0.50)')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# PR curve
ax = axes[1, 0]
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
ax.plot(rec_curve, prec_curve, 'b-', linewidth=2, label=f'Combined (AP={auc_pr:.3f})')
prec_v, rec_v, _ = precision_recall_curve(y_true, v_scores)
prec_l, rec_l, _ = precision_recall_curve(y_true, l_scores)
ax.plot(rec_v, prec_v, 'g--', linewidth=1.5, label=f'Vision (AP={average_precision_score(y_true, v_scores):.3f})')
ax.plot(rec_l, prec_l, 'r--', linewidth=1.5, label=f'LLM (AP={average_precision_score(y_true, l_scores):.3f})')
baseline = y_true.mean()
ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Score distribution by class
ax = axes[1, 1]
scores_exploit = y_score[y_true == 1]
scores_clean = y_score[y_true == 0]
ax.hist(scores_clean, bins=15, alpha=0.6, color='green', label=f'Human=Clean (n={len(scores_clean)})')
ax.hist(scores_exploit, bins=15, alpha=0.6, color='red', label=f'Human=Exploit (n={len(scores_exploit)})')
ax.axvline(x=best_f1_threshold, color='blue', linestyle='--', linewidth=2, label=f'Optimal threshold ({best_f1_threshold:.2f})')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Default threshold (0.50)')
ax.set_xlabel('Pipeline Exploitation Score')
ax.set_ylabel('Count')
ax.set_title('Score Distribution by Human Label')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/KidInfluencer/figures_v4/fig_validation_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to figures_v4/fig_validation_metrics.png")

# Summary JSON
summary = {
    'n_matched': len(merged),
    'auc_roc_combined': float(auc_roc),
    'auc_roc_vision': float(auc_v),
    'auc_roc_llm': float(auc_l),
    'auc_pr_combined': float(auc_pr),
    'optimal_threshold_f1': float(best_f1_threshold),
    'metrics_at_optimal': {
        'f1': float(f1_scores[best_f1_idx]),
        'kappa': float(kappa_scores[best_f1_idx]),
        'accuracy': float(acc_scores[best_f1_idx])
    },
    'metrics_at_default_05': {
        'f1': float(f1_score(y_true, pred_05)),
        'kappa': float(cohen_kappa_score(y_true, pred_05)),
        'accuracy': float(accuracy_score(y_true, pred_05))
    },
    'spearman_rho': float(rho),
    'spearman_p': float(p_rho),
    'kendall_tau': float(tau),
    'top_20_precision': float(y_true[sorted_idx[:max(1, int(len(y_true)*0.2))]].mean()),
    'bottom_20_clean_rate': float((1-y_true[sorted_idx[-max(1, int(len(y_true)*0.2)):]]).mean())
}
with open('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v4/threshold_optimization.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to threshold_optimization.json")
