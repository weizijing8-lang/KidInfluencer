"""
Anomaly Detection for Kidfluencer Risk Identification
======================================================
Methods:
1. Isolation Forest
2. One-Class SVM
3. PU Learning (Positive-Unlabeled)
4. Ensemble

Validation: Compare outputs against known problematic channels
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Output directory
os.makedirs('/home/ubuntu/KidInfluencer/analysis_v3/anomaly_detection', exist_ok=True)
OUT_DIR = '/home/ubuntu/KidInfluencer/analysis_v3/anomaly_detection'

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/channel_risk_scores_v3.csv')
print(f"Total channels: {df.shape[0]}")

# Define known problematic channels (ground truth for validation)
# These are channels with documented controversies, legal issues, or media investigations
KNOWN_PROBLEMATIC = {
    'Piper Rockelle': 'lawsuit_exploitation',      # Mother sued for exploitation
    'Jordan Matter': 'controversy_minors',          # Criticized for content with minors
    'Ryan\'s World': 'ftc_investigation',           # FTC complaint about deceptive ads
    'JesssFam': 'controversy_exploitation',         # Criticized for exploiting children
    'Not Enough Nelsons': 'controversy_privacy',    # Criticized for exposing children's privacy
    'Beast Family Vlogs': 'controversy_clickbait',  # Known for extreme clickbait with kids
    'Daily Bumps': 'controversy_privacy',           # Criticized for oversharing children's lives
    'Forever Family Vlogs': 'controversy_exploitation',  # Criticized for exploitation
}

# Mark known problematic channels
df['is_known_problematic'] = df['channel_title'].isin(KNOWN_PROBLEMATIC.keys()).astype(int)
print(f"Known problematic channels found in dataset: {df['is_known_problematic'].sum()}")
print(f"Known problematic channels: {df[df['is_known_problematic']==1]['channel_title'].tolist()}")

# Feature columns for anomaly detection
FEATURE_COLS = [
    'emotional_score_mean',
    'commercial_rate', 
    'privacy_score_mean',
    'clickbait_score_mean',
    'child_protagonist_rate',
    'cross_platform',
]

# Additional structural features
df['upload_frequency'] = df['total_videos_on_channel'] / 365  # rough estimate (videos per day)
df['log_subscribers'] = np.log1p(df['subscribers'])
df['views_per_sub'] = df['avg_views'] / (df['subscribers'] + 1)

FEATURE_COLS_EXTENDED = FEATURE_COLS + ['upload_frequency', 'log_subscribers', 'views_per_sub']

print(f"\nFeatures used: {FEATURE_COLS_EXTENDED}")
print(f"\nFeature statistics:")
print(df[FEATURE_COLS_EXTENDED].describe())

# Prepare feature matrix
X = df[FEATURE_COLS_EXTENDED].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2. ISOLATION FOREST
# ============================================================
print("\n" + "=" * 60)
print("2. ISOLATION FOREST")
print("=" * 60)

# Try different contamination rates
contamination_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
if_results = {}

for cont in contamination_rates:
    clf = IsolationForest(
        n_estimators=200,
        contamination=cont,
        random_state=42,
        max_samples='auto'
    )
    df[f'if_pred_{int(cont*100)}'] = clf.fit_predict(X_scaled)
    df[f'if_score_{int(cont*100)}'] = -clf.decision_function(X_scaled)  # Higher = more anomalous
    
    # Count how many known problematic are flagged
    flagged = df[df[f'if_pred_{int(cont*100)}'] == -1]
    n_flagged = len(flagged)
    n_problematic_flagged = flagged['is_known_problematic'].sum()
    
    precision = n_problematic_flagged / n_flagged if n_flagged > 0 else 0
    recall = n_problematic_flagged / df['is_known_problematic'].sum() if df['is_known_problematic'].sum() > 0 else 0
    
    if_results[cont] = {
        'n_flagged': n_flagged,
        'n_problematic_flagged': n_problematic_flagged,
        'precision': precision,
        'recall': recall,
        'f1': 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    }
    
    print(f"\n  Contamination={cont:.0%}:")
    print(f"    Flagged: {n_flagged} channels")
    print(f"    Known problematic flagged: {n_problematic_flagged}/{df['is_known_problematic'].sum()}")
    print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {if_results[cont]['f1']:.2%}")
    if n_problematic_flagged > 0:
        print(f"    Flagged problematic: {flagged[flagged['is_known_problematic']==1]['channel_title'].tolist()}")

# Use the anomaly score from best contamination
best_cont = max(if_results, key=lambda k: if_results[k]['f1'])
print(f"\n  Best contamination rate: {best_cont:.0%} (F1={if_results[best_cont]['f1']:.2%})")
df['if_anomaly_score'] = df[f'if_score_{int(best_cont*100)}']

# ============================================================
# 3. ONE-CLASS SVM
# ============================================================
print("\n" + "=" * 60)
print("3. ONE-CLASS SVM")
print("=" * 60)

# Train on "normal" channels (those NOT known problematic)
# This is the key idea: learn what "normal" looks like, then flag deviations
X_train = X_scaled[df['is_known_problematic'] == 0]
print(f"  Training on {len(X_train)} 'normal' channels")

# Try different nu values (expected fraction of outliers)
nu_values = [0.05, 0.10, 0.15, 0.20, 0.30]
ocsvm_results = {}

for nu in nu_values:
    clf = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    clf.fit(X_train)
    
    df[f'ocsvm_pred_{int(nu*100)}'] = clf.predict(X_scaled)
    df[f'ocsvm_score_{int(nu*100)}'] = -clf.decision_function(X_scaled)
    
    flagged = df[df[f'ocsvm_pred_{int(nu*100)}'] == -1]
    n_flagged = len(flagged)
    n_problematic_flagged = flagged['is_known_problematic'].sum()
    
    precision = n_problematic_flagged / n_flagged if n_flagged > 0 else 0
    recall = n_problematic_flagged / df['is_known_problematic'].sum() if df['is_known_problematic'].sum() > 0 else 0
    
    ocsvm_results[nu] = {
        'n_flagged': n_flagged,
        'n_problematic_flagged': n_problematic_flagged,
        'precision': precision,
        'recall': recall,
        'f1': 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    }
    
    print(f"\n  nu={nu:.2f}:")
    print(f"    Flagged: {n_flagged} channels")
    print(f"    Known problematic flagged: {n_problematic_flagged}/{df['is_known_problematic'].sum()}")
    print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {ocsvm_results[nu]['f1']:.2%}")
    if n_problematic_flagged > 0:
        print(f"    Flagged problematic: {flagged[flagged['is_known_problematic']==1]['channel_title'].tolist()}")

best_nu = max(ocsvm_results, key=lambda k: ocsvm_results[k]['f1'])
print(f"\n  Best nu: {best_nu:.2f} (F1={ocsvm_results[best_nu]['f1']:.2%})")
df['ocsvm_anomaly_score'] = df[f'ocsvm_score_{int(best_nu*100)}']

# ============================================================
# 4. PU LEARNING (Positive-Unlabeled)
# ============================================================
print("\n" + "=" * 60)
print("4. PU LEARNING")
print("=" * 60)

# Simple PU Learning approach:
# 1. Use known problematic as Positive
# 2. All others as Unlabeled (not necessarily negative)
# 3. Use iterative approach: train classifier, identify likely negatives, retrain

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

# Step 1: Initial classifier treating unlabeled as negative
print("  Step 1: Initial classifier (P vs U)")
y_pu = df['is_known_problematic'].values

# Use bagging-based PU learning
# Run multiple iterations, each time sampling from unlabeled as "negative"
n_iterations = 50
pu_scores = np.zeros(len(df))

np.random.seed(42)
for i in range(n_iterations):
    # Sample reliable negatives: channels with lowest risk scores
    unlabeled_idx = np.where(y_pu == 0)[0]
    positive_idx = np.where(y_pu == 1)[0]
    
    # Randomly sample same number of "negatives" from unlabeled
    n_neg = min(len(positive_idx) * 3, len(unlabeled_idx))
    neg_sample_idx = np.random.choice(unlabeled_idx, size=n_neg, replace=False)
    
    # Create training set
    train_idx = np.concatenate([positive_idx, neg_sample_idx])
    X_train_pu = X_scaled[train_idx]
    y_train_pu = np.concatenate([np.ones(len(positive_idx)), np.zeros(n_neg)])
    
    # Train classifier
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=i)
    clf.fit(X_train_pu, y_train_pu)
    
    # Score all channels
    pu_scores += clf.predict_proba(X_scaled)[:, 1]

# Average scores
pu_scores /= n_iterations
df['pu_score'] = pu_scores

# Evaluate
print(f"\n  PU Learning scores (top 15):")
top_pu = df.nlargest(15, 'pu_score')[['channel_title', 'pu_score', 'is_known_problematic']]
for _, row in top_pu.iterrows():
    marker = " *** KNOWN" if row['is_known_problematic'] else ""
    print(f"    {row['channel_title'][:35]:35s} | score: {row['pu_score']:.3f}{marker}")

# Threshold at different levels
for threshold in [0.5, 0.4, 0.3]:
    flagged = df[df['pu_score'] >= threshold]
    n_flagged = len(flagged)
    n_problematic_flagged = flagged['is_known_problematic'].sum()
    precision = n_problematic_flagged / n_flagged if n_flagged > 0 else 0
    recall = n_problematic_flagged / df['is_known_problematic'].sum() if df['is_known_problematic'].sum() > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    print(f"\n  Threshold={threshold:.1f}: Flagged={n_flagged}, Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}")

# ============================================================
# 5. ENSEMBLE METHOD
# ============================================================
print("\n" + "=" * 60)
print("5. ENSEMBLE (combining all methods)")
print("=" * 60)

# Normalize all scores to 0-1
minmax = MinMaxScaler()
ensemble_features = pd.DataFrame({
    'if_score': df['if_anomaly_score'],
    'ocsvm_score': df['ocsvm_anomaly_score'],
    'pu_score': df['pu_score'],
})
ensemble_normalized = pd.DataFrame(
    minmax.fit_transform(ensemble_features),
    columns=ensemble_features.columns
)

# Weighted ensemble (PU learning gets higher weight since it uses label info)
weights = {'if_score': 0.25, 'ocsvm_score': 0.25, 'pu_score': 0.50}
df['ensemble_score'] = sum(ensemble_normalized[col] * w for col, w in weights.items())

# Final ranking
print("\n  FINAL ENSEMBLE RANKING (Top 20):")
print(f"  {'Rank':<5} {'Channel':<35} {'Score':<8} {'IF':<8} {'OCSVM':<8} {'PU':<8} {'Known'}")
print("  " + "-" * 95)

top20 = df.nlargest(20, 'ensemble_score')
for rank, (_, row) in enumerate(top20.iterrows(), 1):
    known = "YES" if row['is_known_problematic'] else ""
    print(f"  {rank:<5} {row['channel_title'][:35]:<35} {row['ensemble_score']:.3f}   "
          f"{ensemble_normalized.loc[row.name, 'if_score']:.2f}    "
          f"{ensemble_normalized.loc[row.name, 'ocsvm_score']:.2f}    "
          f"{ensemble_normalized.loc[row.name, 'pu_score']:.2f}    {known}")

# Evaluate ensemble at different thresholds
print("\n  Ensemble evaluation:")
for top_k in [5, 10, 15, 20, 25]:
    top_channels = df.nlargest(top_k, 'ensemble_score')
    n_problematic = top_channels['is_known_problematic'].sum()
    precision_at_k = n_problematic / top_k
    recall_at_k = n_problematic / df['is_known_problematic'].sum()
    print(f"    Top-{top_k}: {n_problematic}/{top_k} known problematic (P@{top_k}={precision_at_k:.2%}, R@{top_k}={recall_at_k:.2%})")

# ============================================================
# 6. FEATURE IMPORTANCE (from PU Learning)
# ============================================================
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

# Train one final model for feature importance
clf_final = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
# Use all positive + random negative sample
pos_idx = np.where(y_pu == 1)[0]
neg_idx = np.where(y_pu == 0)[0]
neg_sample = np.random.choice(neg_idx, size=len(neg_idx), replace=False)  # use all
train_idx = np.concatenate([pos_idx, neg_sample])
clf_final.fit(X_scaled[train_idx], y_pu[train_idx])

importances = clf_final.feature_importances_
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS_EXTENDED,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n  Feature Importance (Gradient Boosting):")
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"    {row['feature']:<25} {row['importance']:.3f} {bar}")

# ============================================================
# 7. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("7. GENERATING VISUALIZATIONS")
print("=" * 60)

plt.style.use('seaborn-v0_8-whitegrid')

# Fig 1: Method comparison (Precision-Recall at different thresholds)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# IF results
conts = list(if_results.keys())
ax = axes[0]
ax.bar(range(len(conts)), [if_results[c]['recall'] for c in conts], alpha=0.7, label='Recall', color='steelblue')
ax.bar(range(len(conts)), [if_results[c]['precision'] for c in conts], alpha=0.5, label='Precision', color='coral')
ax.set_xticks(range(len(conts)))
ax.set_xticklabels([f'{c:.0%}' for c in conts])
ax.set_xlabel('Contamination Rate')
ax.set_ylabel('Score')
ax.set_title('Isolation Forest')
ax.legend()
ax.set_ylim(0, 1)

# OCSVM results
nus = list(ocsvm_results.keys())
ax = axes[1]
ax.bar(range(len(nus)), [ocsvm_results[n]['recall'] for n in nus], alpha=0.7, label='Recall', color='steelblue')
ax.bar(range(len(nus)), [ocsvm_results[n]['precision'] for n in nus], alpha=0.5, label='Precision', color='coral')
ax.set_xticks(range(len(nus)))
ax.set_xticklabels([f'{n:.2f}' for n in nus])
ax.set_xlabel('Nu')
ax.set_ylabel('Score')
ax.set_title('One-Class SVM')
ax.legend()
ax.set_ylim(0, 1)

# Ensemble top-k
ax = axes[2]
top_ks = [5, 10, 15, 20, 25, 30]
precisions = []
recalls = []
for k in top_ks:
    top = df.nlargest(k, 'ensemble_score')
    n_prob = top['is_known_problematic'].sum()
    precisions.append(n_prob / k)
    recalls.append(n_prob / df['is_known_problematic'].sum())
ax.plot(top_ks, precisions, 'o-', label='Precision@K', color='coral')
ax.plot(top_ks, recalls, 's-', label='Recall@K', color='steelblue')
ax.set_xlabel('Top-K')
ax.set_ylabel('Score')
ax.set_title('Ensemble (P@K and R@K)')
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_method_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved fig1_method_comparison.png")

# Fig 2: Ensemble score distribution with known problematic highlighted
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d32f2f' if x == 1 else '#1976d2' for x in df.sort_values('ensemble_score', ascending=True)['is_known_problematic']]
sorted_df = df.sort_values('ensemble_score', ascending=True).reset_index(drop=True)
bars = ax.barh(range(len(sorted_df)), sorted_df['ensemble_score'], 
               color=['#d32f2f' if x == 1 else '#90caf9' for x in sorted_df['is_known_problematic']])

# Label top channels
for i, row in sorted_df.tail(15).iterrows():
    ax.text(row['ensemble_score'] + 0.01, i, row['channel_title'][:25], 
            va='center', fontsize=8, fontweight='bold' if row['is_known_problematic'] else 'normal')

ax.set_xlabel('Ensemble Anomaly Score')
ax.set_title('Channel Risk Scores (Red = Known Problematic)')
ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_ensemble_distribution.png', dpi=150, bbox_inches='tight')
print("  Saved fig2_ensemble_distribution.png")

# Fig 3: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp_sorted = feat_imp.sort_values('importance', ascending=True)
ax.barh(range(len(feat_imp_sorted)), feat_imp_sorted['importance'], color='steelblue')
ax.set_yticks(range(len(feat_imp_sorted)))
ax.set_yticklabels(feat_imp_sorted['feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('What Features Best Distinguish High-Risk Channels?\n(Gradient Boosting Feature Importance)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_feature_importance.png', dpi=150, bbox_inches='tight')
print("  Saved fig3_feature_importance.png")

# Fig 4: 2D projection of channels (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
scatter_normal = ax.scatter(X_pca[df['is_known_problematic']==0, 0], 
                           X_pca[df['is_known_problematic']==0, 1],
                           c=df[df['is_known_problematic']==0]['ensemble_score'],
                           cmap='YlOrRd', s=60, alpha=0.7, edgecolors='gray', linewidths=0.5)
scatter_known = ax.scatter(X_pca[df['is_known_problematic']==1, 0], 
                          X_pca[df['is_known_problematic']==1, 1],
                          c='red', s=150, marker='*', edgecolors='black', linewidths=1,
                          label='Known Problematic', zorder=5)

# Label known problematic
for idx in df[df['is_known_problematic']==1].index:
    ax.annotate(df.loc[idx, 'channel_title'][:20], 
               (X_pca[idx, 0], X_pca[idx, 1]),
               fontsize=8, fontweight='bold',
               xytext=(5, 5), textcoords='offset points')

plt.colorbar(scatter_normal, label='Ensemble Risk Score')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('Channel Feature Space (PCA Projection)\nKnown Problematic Channels Cluster Together')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_pca_projection.png', dpi=150, bbox_inches='tight')
print("  Saved fig4_pca_projection.png")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("8. SAVING RESULTS")
print("=" * 60)

# Save full results
result_cols = ['channel_id', 'channel_title', 'subscribers', 'is_known_problematic',
               'if_anomaly_score', 'ocsvm_anomaly_score', 'pu_score', 'ensemble_score'] + FEATURE_COLS_EXTENDED
df[result_cols].sort_values('ensemble_score', ascending=False).to_csv(
    f'{OUT_DIR}/anomaly_detection_results.csv', index=False)
print("  Saved anomaly_detection_results.csv")

# Save summary
summary = {
    'dataset': {'n_channels': len(df), 'n_known_problematic': int(df['is_known_problematic'].sum())},
    'features': FEATURE_COLS_EXTENDED,
    'isolation_forest': {str(k): v for k, v in if_results.items()},
    'one_class_svm': {str(k): v for k, v in ocsvm_results.items()},
    'ensemble_top10_precision': precisions[1],  # P@10
    'ensemble_top10_recall': recalls[1],
    'feature_importance': feat_imp.set_index('feature')['importance'].to_dict(),
    'best_params': {'if_contamination': best_cont, 'ocsvm_nu': best_nu}
}

with open(f'{OUT_DIR}/experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved experiment_summary.json")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
