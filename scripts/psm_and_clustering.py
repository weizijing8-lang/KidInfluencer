"""
Phase 4: Propensity Score Matching + Embedding-based Clustering
Paper 1 - AI-powered computational audit of kidfluencer channels

This script:
1. Propensity Score Matching: Match family vs adult channels on observables
2. Estimate Average Treatment Effect (ATE) of being a family channel on labor metrics
3. Sentence-BERT embeddings of video titles for content clustering
4. BERTopic-style topic modeling to find latent content strategies
5. t-SNE/UMAP visualization of content space
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

import subprocess
subprocess.run(['sudo', 'pip3', 'install', 'sentence-transformers', 'umap-learn', '-q'],
               capture_output=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

BASE = '/home/ubuntu/KidInfluencer'
OUTPUT = f'{BASE}/analysis_paper1_v2'
FIG_DIR = f'{OUTPUT}/figures'

# ============================================================
# 1. PROPENSITY SCORE MATCHING
# ============================================================
print("=" * 60)
print("PHASE 1: PROPENSITY SCORE MATCHING")
print("=" * 60)

channel_df = pd.read_csv(f'{OUTPUT}/channels_with_indices_v2.csv')
print(f"Channels: {channel_df.shape}")

# Treatment: is_family (1 = family/kidfluencer, 0 = adult)
channel_df['is_family'] = (channel_df['category'] == 'family').astype(int)

# Covariates for matching (things that should be similar between groups)
match_covariates = ['log_total_views', 'n_videos', 'span_years']
channel_df['log_total_views'] = np.log10(channel_df['total_views'].clip(lower=1))

# Drop rows with missing values in covariates
psm_df = channel_df.dropna(subset=match_covariates + ['is_family']).copy()
print(f"PSM dataset: {psm_df.shape[0]} channels ({psm_df['is_family'].sum()} family, {(1-psm_df['is_family']).sum()} adult)")

# Step 1: Estimate propensity scores using logistic regression
X_psm = StandardScaler().fit_transform(psm_df[match_covariates])
y_psm = psm_df['is_family'].values

lr = LogisticRegression(random_state=42)
lr.fit(X_psm, y_psm)
psm_df['propensity_score'] = lr.predict_proba(X_psm)[:, 1]

print(f"\nPropensity Score Distribution:")
print(f"  Family: mean={psm_df[psm_df['is_family']==1]['propensity_score'].mean():.3f}, "
      f"std={psm_df[psm_df['is_family']==1]['propensity_score'].std():.3f}")
print(f"  Adult:  mean={psm_df[psm_df['is_family']==0]['propensity_score'].mean():.3f}, "
      f"std={psm_df[psm_df['is_family']==0]['propensity_score'].std():.3f}")

# Step 2: Nearest-neighbor matching (1:1 without replacement)
treated = psm_df[psm_df['is_family'] == 1].copy()
control = psm_df[psm_df['is_family'] == 0].copy()

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity_score']].values)
distances, indices = nn.kneighbors(treated[['propensity_score']].values)

# Get matched pairs
matched_control_idx = control.iloc[indices.flatten()].index
matched_df = pd.concat([treated, psm_df.loc[matched_control_idx]])
print(f"\nMatched sample: {len(matched_df)} channels ({len(treated)} pairs)")

# Step 3: Check covariate balance after matching
print("\n--- Covariate Balance After Matching ---")
print(f"{'Covariate':<20} {'Family Mean':>12} {'Adult Mean':>12} {'SMD':>8}")
print("-" * 55)
for cov in match_covariates:
    f_mean = matched_df[matched_df['is_family']==1][cov].mean()
    a_mean = matched_df[matched_df['is_family']==0][cov].mean()
    pooled_std = np.sqrt((matched_df[matched_df['is_family']==1][cov].std()**2 + 
                          matched_df[matched_df['is_family']==0][cov].std()**2) / 2)
    smd = (f_mean - a_mean) / pooled_std if pooled_std > 0 else 0
    print(f"{cov:<20} {f_mean:>12.3f} {a_mean:>12.3f} {smd:>8.3f}")

# Step 4: Estimate ATE on labor/content outcomes
print("\n--- Average Treatment Effect (Family vs Matched Adult) ---")
outcome_vars = ['videos_per_week', 'mean_duration_min', 'weekly_production_hours_est',
                'mean_exploit_v4', 'sponsor_rate', 'n_child_brands', 'lii_pca', 'ci_pca']

ate_results = []
print(f"{'Outcome':<30} {'ATE':>8} {'95% CI':>20} {'p-value':>10}")
print("-" * 70)

for var in outcome_vars:
    treated_vals = matched_df[matched_df['is_family']==1][var].dropna()
    control_vals = matched_df[matched_df['is_family']==0][var].dropna()
    
    if len(treated_vals) > 0 and len(control_vals) > 0:
        ate = treated_vals.mean() - control_vals.mean()
        # Bootstrap CI
        n_boot = 1000
        boot_ates = []
        for _ in range(n_boot):
            t_sample = treated_vals.sample(n=len(treated_vals), replace=True)
            c_sample = control_vals.sample(n=len(control_vals), replace=True)
            boot_ates.append(t_sample.mean() - c_sample.mean())
        ci_low = np.percentile(boot_ates, 2.5)
        ci_high = np.percentile(boot_ates, 97.5)
        
        # Wilcoxon/Mann-Whitney
        u, p = stats.mannwhitneyu(treated_vals, control_vals, alternative='two-sided')
        
        sig = '*' if p < 0.05 else ''
        print(f"{var:<30} {ate:>8.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {p:>10.4f}{sig}")
        
        ate_results.append({
            'outcome': var, 'ate': ate, 'ci_low': ci_low, 'ci_high': ci_high, 'p': p
        })

# ============================================================
# 2. EMBEDDING-BASED CONTENT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: SENTENCE EMBEDDING ANALYSIS")
print("=" * 60)

# Load video titles from V4
v4 = pd.read_csv(f'{BASE}/data/results_v4/full_results_v4.csv',
                  usecols=['id', 'title', 'channel_short_name', 'channel_category',
                           'viewCount', 'exploit_score_v4'])
v4 = v4[v4['channel_category'].isin(['adult', 'family'])].copy()
v4 = v4.dropna(subset=['title'])

# Sample for embedding (full dataset too large for sentence-transformers)
np.random.seed(42)
n_sample = 5000
family_sample = v4[v4['channel_category'] == 'family'].sample(
    n=min(2500, len(v4[v4['channel_category'] == 'family'])), random_state=42)
adult_sample = v4[v4['channel_category'] == 'adult'].sample(
    n=min(2500, len(v4[v4['channel_category'] == 'adult'])), random_state=42)
sample_df = pd.concat([family_sample, adult_sample]).reset_index(drop=True)
print(f"Sample for embedding: {len(sample_df)} videos")

# Generate embeddings using sentence-transformers
from sentence_transformers import SentenceTransformer

print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings...")
embeddings = model.encode(sample_df['title'].tolist(), show_progress_bar=True, batch_size=64)
print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save(f'{OUTPUT}/title_embeddings.npy', embeddings)
sample_df.to_csv(f'{OUTPUT}/embedding_sample.csv', index=False)

# ============================================================
# 3. CLUSTERING ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: CONTENT CLUSTERING")
print("=" * 60)

# K-Means clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)
sample_df['cluster'] = cluster_labels

# Analyze clusters
print(f"\nCluster composition (Family vs Adult):")
cluster_stats = sample_df.groupby('cluster').agg(
    n_videos=('id', 'count'),
    pct_family=('channel_category', lambda x: (x == 'family').mean()),
    mean_views=('viewCount', 'mean'),
    mean_exploit=('exploit_score_v4', 'mean')
).round(3)
print(cluster_stats.to_string())

# Find representative titles for each cluster
print("\nRepresentative titles per cluster:")
for c in range(n_clusters):
    cluster_titles = sample_df[sample_df['cluster'] == c]['title'].head(5).tolist()
    pct_fam = cluster_stats.loc[c, 'pct_family']
    print(f"\n  Cluster {c} ({pct_fam:.0%} family):")
    for t in cluster_titles[:3]:
        print(f"    - {t[:60]}")

# ============================================================
# 4. DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: t-SNE VISUALIZATION")
print("=" * 60)

# PCA first to reduce to 50 dims, then t-SNE
pca_50 = PCA(n_components=50, random_state=42)
embeddings_pca = pca_50.fit_transform(embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_pca)

sample_df['tsne_x'] = embeddings_2d[:, 0]
sample_df['tsne_y'] = embeddings_2d[:, 1]

# Plot t-SNE colored by category
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Colored by category
ax = axes[0]
family_mask = sample_df['channel_category'] == 'family'
ax.scatter(embeddings_2d[~family_mask, 0], embeddings_2d[~family_mask, 1],
           c='#2196F3', alpha=0.3, s=10, label='Adult')
ax.scatter(embeddings_2d[family_mask, 0], embeddings_2d[family_mask, 1],
           c='#F44336', alpha=0.3, s=10, label='Family')
ax.set_title('(A) Content Space by Channel Category', fontsize=13)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=11, markerscale=3)

# Panel B: Colored by cluster
ax = axes[1]
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=cluster_labels, cmap='tab10', alpha=0.4, s=10)
ax.set_title('(B) Content Clusters (K=8)', fontsize=13)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.suptitle('Title Embedding Space: Family vs Adult Content Strategies', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/tsne_content_space.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: tsne_content_space.png")

# ============================================================
# 5. CLUSTER ENRICHMENT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: CLUSTER ENRICHMENT")
print("=" * 60)

# Chi-square test: are family channels over-represented in certain clusters?
contingency = pd.crosstab(sample_df['cluster'], sample_df['channel_category'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square test for cluster × category association:")
print(f"  χ² = {chi2:.2f}, df = {dof}, p = {p_chi:.6f}")

# Enrichment ratio per cluster
print(f"\n{'Cluster':<10} {'Family %':>10} {'Expected %':>12} {'Enrichment':>12} {'Interpretation':<30}")
print("-" * 75)
overall_family_pct = family_mask.mean()
for c in range(n_clusters):
    obs_pct = cluster_stats.loc[c, 'pct_family']
    enrichment = obs_pct / overall_family_pct
    if enrichment > 1.3:
        interp = "FAMILY-DOMINATED"
    elif enrichment < 0.7:
        interp = "ADULT-DOMINATED"
    else:
        interp = "Mixed"
    print(f"{c:<10} {obs_pct:>10.1%} {overall_family_pct:>12.1%} {enrichment:>12.2f}x {interp:<30}")

# ============================================================
# 6. EXPLOITATION SCORE BY CLUSTER
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: EXPLOITATION PATTERNS BY CLUSTER")
print("=" * 60)

# Which clusters have highest exploitation scores?
exploit_by_cluster = sample_df.groupby(['cluster', 'channel_category'])['exploit_score_v4'].agg(['mean', 'std', 'count'])
print(exploit_by_cluster.to_string())

# Plot exploitation by cluster and category
fig, ax = plt.subplots(figsize=(10, 6))
cluster_exploit = sample_df.groupby(['cluster', 'channel_category'])['exploit_score_v4'].mean().unstack()
cluster_exploit.plot(kind='bar', ax=ax, color=['#2196F3', '#F44336'], alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Content Cluster', fontsize=12)
ax.set_ylabel('Mean Exploitation Score', fontsize=12)
ax.set_title('Content Exploitation by Cluster and Channel Category', fontsize=13)
ax.legend(['Adult', 'Family'], fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/exploit_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: exploit_by_cluster.png")

# ============================================================
# 7. PSM VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: PSM VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Propensity score distribution
ax = axes[0]
ax.hist(psm_df[psm_df['is_family']==0]['propensity_score'], bins=15, alpha=0.6, 
        label='Adult', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.hist(psm_df[psm_df['is_family']==1]['propensity_score'], bins=15, alpha=0.6,
        label='Family', color='#F44336', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Propensity Score')
ax.set_ylabel('Count')
ax.set_title('(A) Propensity Score Distribution')
ax.legend()

# Panel B: ATE Forest Plot
ax = axes[1]
ate_df = pd.DataFrame(ate_results)
ate_df = ate_df.sort_values('ate')
y_pos = range(len(ate_df))
colors_ate = ['#F44336' if p < 0.05 else '#9E9E9E' for p in ate_df['p']]
ax.barh(y_pos, ate_df['ate'], xerr=[ate_df['ate']-ate_df['ci_low'], ate_df['ci_high']-ate_df['ate']],
        color=colors_ate, alpha=0.7, edgecolor='black', linewidth=0.5, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(ate_df['outcome'], fontsize=9)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Average Treatment Effect (Family - Adult)')
ax.set_title('(B) PSM: Treatment Effects')

# Panel C: Matched vs Unmatched comparison
ax = axes[2]
# Show key outcome before and after matching
key_var = 'videos_per_week'
unmatched_diff = (channel_df[channel_df['category']=='family'][key_var].mean() - 
                  channel_df[channel_df['category']=='adult'][key_var].mean())
matched_diff = (matched_df[matched_df['is_family']==1][key_var].mean() - 
                matched_df[matched_df['is_family']==0][key_var].mean())

bars = ax.bar(['Unmatched', 'PSM Matched'], [unmatched_diff, matched_diff],
              color=['#FF9800', '#4CAF50'], alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Difference (Family - Adult)')
ax.set_title(f'(C) Upload Frequency Gap')
for bar, val in zip(bars, [unmatched_diff, matched_diff]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Propensity Score Matching: Family vs Adult Channels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/psm_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: psm_results.png")

# ============================================================
# 8. SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

psm_results = {
    'n_treated': int(len(treated)),
    'n_control': int(len(control)),
    'n_matched_pairs': int(len(treated)),
    'chi2_cluster_category': {'chi2': float(chi2), 'p': float(p_chi), 'dof': int(dof)},
    'ate_results': ate_results,
    'cluster_stats': cluster_stats.to_dict(),
    'n_clusters': n_clusters,
    'embedding_dim': int(embeddings.shape[1]),
    'sample_size': int(len(sample_df)),
}

with open(f'{OUTPUT}/psm_clustering_results.json', 'w') as f:
    json.dump(psm_results, f, indent=2, default=str)
print(f"Saved: psm_clustering_results.json")

# Save cluster assignments
sample_df[['id', 'title', 'channel_short_name', 'channel_category', 'cluster', 
           'tsne_x', 'tsne_y', 'viewCount', 'exploit_score_v4']].to_csv(
    f'{OUTPUT}/cluster_assignments.csv', index=False)
print(f"Saved: cluster_assignments.csv")

print("\n" + "=" * 60)
print("PSM + CLUSTERING COMPLETE")
print("=" * 60)
