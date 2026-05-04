"""
Cluster kidfluencer videos using RESIDUAL Sentence-BERT embeddings.
Method: For each video, subtract the channel's mean embedding to remove
channel-specific titling style, then cluster the residuals.

This preserves semantic understanding ("Who stole my phone" ≈ "Someone took my iPad")
while removing channel confound (Cocomelon's "Song | Nursery Rhymes" pattern).
"""
import pandas as pd
import numpy as np
import json, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats

print("Loading data...")
# Load the full results to get video IDs and map to embeddings
df_full = pd.read_csv('data/results_v4/full_results_v4.csv')
embeddings = np.load('data/results_v4/embeddings_v4.npy')
print(f"  Full dataset: {len(df_full)} videos, embeddings: {embeddings.shape}")

# Load our labeled subset
df_labeled = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
print(f"  Labeled dataset: {len(df_labeled)} videos")

# Map labeled videos to their embeddings via video ID
id_to_idx = {vid: idx for idx, vid in enumerate(df_full['id'].values)}
labeled_indices = []
valid_mask = []
for i, vid in enumerate(df_labeled['id'].values):
    if vid in id_to_idx:
        labeled_indices.append(id_to_idx[vid])
        valid_mask.append(True)
    else:
        valid_mask.append(False)

valid_mask = np.array(valid_mask)
df = df_labeled[valid_mask].reset_index(drop=True)
X = embeddings[labeled_indices]
print(f"  Matched: {len(df)} videos with embeddings")

# ============================================================
# Compute RESIDUAL embeddings (subtract channel mean)
# ============================================================
print("\nComputing residual embeddings (embedding - channel_mean)...")
channels = df['channel_short_name'].values
unique_channels = np.unique(channels)

# Compute channel means
channel_means = {}
for ch in unique_channels:
    ch_mask = channels == ch
    channel_means[ch] = X[ch_mask].mean(axis=0)
    
# Subtract channel mean from each video
X_residual = np.zeros_like(X)
for i in range(len(X)):
    X_residual[i] = X[i] - channel_means[channels[i]]

print(f"  Residual embedding shape: {X_residual.shape}")
print(f"  Original variance: {X.var():.4f}")
print(f"  Residual variance: {X_residual.var():.4f}")
print(f"  Variance explained by channel: {1 - X_residual.var()/X.var():.1%}")

# ============================================================
# PCA reduction
# ============================================================
print("\nApplying PCA (50 components)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_residual)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# ============================================================
# Find optimal K
# ============================================================
print("\nSearching for optimal K...")
results = []
for k in range(5, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=5000, random_state=42)
    
    # Channel diversity
    max_concentration = 0
    min_channels = 25
    for c in range(k):
        mask = labels == c
        ch_counts = pd.Series(channels[mask]).value_counts()
        n_ch = len(ch_counts)
        top1_pct = ch_counts.iloc[0] / mask.sum()
        max_concentration = max(max_concentration, top1_pct)
        min_channels = min(min_channels, n_ch)
    
    results.append({'k': k, 'silhouette': sil, 'max_concentration': max_concentration, 'min_channels': min_channels})
    print(f"  K={k:2d}: silhouette={sil:.4f}, max_top1={max_concentration:.1%}, min_channels={min_channels}")

results_df = pd.DataFrame(results)

# Select K: best silhouette with reasonable diversity (max_conc < 50%)
good = results_df[results_df['max_concentration'] < 0.50]
if len(good) > 0:
    best_k = int(good.loc[good['silhouette'].idxmax(), 'k'])
else:
    # Relax to 60%
    good = results_df[results_df['max_concentration'] < 0.60]
    if len(good) > 0:
        best_k = int(good.loc[good['silhouette'].idxmax(), 'k'])
    else:
        best_k = int(results_df.loc[results_df['silhouette'].idxmax(), 'k'])

print(f"\nSelected K={best_k}")

# ============================================================
# Final clustering
# ============================================================
print(f"\nRunning final K-Means with K={best_k}...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
final_labels = km_final.fit_predict(X_pca)
df['cluster_v3'] = final_labels

# ============================================================
# Cluster analysis
# ============================================================
print(f"\n{'='*110}")
print(f"CLUSTER ANALYSIS (K={best_k}, Residual SBERT Embeddings)")
print(f"{'='*110}")

overall_median = df['viewCount'].median()
cluster_summaries = []

for c in range(best_k):
    mask = df['cluster_v3'] == c
    cl = df[mask]
    
    n_videos = len(cl)
    n_channels = cl['channel_short_name'].nunique()
    median_views = cl['viewCount'].median()
    view_boost = (median_views - overall_median) / overall_median
    
    # Top channels
    top_channels = cl['channel_short_name'].value_counts().head(5)
    top1_pct = top_channels.iloc[0] / n_videos
    
    # Representative titles (from different channels)
    sample_titles = []
    for ch in cl['channel_short_name'].unique()[:6]:
        ch_titles = cl[cl['channel_short_name'] == ch]['title'].head(2).tolist()
        for t in ch_titles[:1]:
            sample_titles.append(f"[{ch}] {t[:70]}")
    
    # Within-channel boost
    wc_boosts = []
    for ch in cl['channel_short_name'].unique():
        ch_all = df[df['channel_short_name'] == ch]
        ch_median = ch_all['viewCount'].median()
        ch_in_cl = cl[cl['channel_short_name'] == ch]['viewCount'].median()
        if ch_median > 0 and len(cl[cl['channel_short_name'] == ch]) >= 5:
            wc_boosts.append((ch_in_cl - ch_median) / ch_median)
    
    wc_median = np.median(wc_boosts) if wc_boosts else 0
    wc_p = stats.ttest_1samp(wc_boosts, 0)[1] if len(wc_boosts) >= 3 else 1.0
    
    summary = {
        'cluster': c,
        'n_videos': n_videos,
        'n_channels': n_channels,
        'top1_channel': top_channels.index[0],
        'top1_pct': float(top1_pct),
        'median_views': float(median_views),
        'view_boost': float(view_boost),
        'within_channel_boost_median': float(wc_median),
        'within_channel_p': float(wc_p),
        'sample_titles': sample_titles[:8]
    }
    cluster_summaries.append(summary)
    
    sig = '***' if wc_p<0.001 else '**' if wc_p<0.01 else '*' if wc_p<0.05 else 'ns'
    print(f"\n{'─'*110}")
    print(f"Cluster {c}: {n_videos} videos ({n_videos/len(df)*100:.1f}%), {n_channels} channels")
    print(f"  View boost: {view_boost*100:+.0f}% (overall) | {wc_median*100:+.1f}% (within-channel, p={wc_p:.3f} {sig})")
    print(f"  Top channels: {', '.join([f'{ch}({cnt}, {cnt/n_videos*100:.0f}%)' for ch, cnt in top_channels.items()])}")
    print(f"  Sample titles:")
    for t in sample_titles[:6]:
        print(f"    {t}")

# ============================================================
# Channel x Cluster distribution
# ============================================================
print(f"\n\n{'='*110}")
print("CHANNEL x CLUSTER DISTRIBUTION (% of each channel's videos)")
print(f"{'='*110}")
cross_tab = pd.crosstab(df['channel_short_name'], df['cluster_v3'], normalize='index') * 100
print(cross_tab.round(1).to_string())

# ============================================================
# Save results
# ============================================================
df.to_csv('analysis_discovery/videos_with_clusters_v3.csv', index=False)
with open('analysis_discovery/cluster_v3_summaries.json', 'w') as f:
    json.dump(cluster_summaries, f, indent=2, default=str)
cross_tab.to_csv('analysis_discovery/channel_cluster_crosstab_v3.csv')
results_df.to_csv('analysis_discovery/k_search_results_v3.csv', index=False)

print(f"\n\nSaved to analysis_discovery/")
print(f"  videos_with_clusters_v3.csv")
print(f"  cluster_v3_summaries.json")
print(f"  channel_cluster_crosstab_v3.csv")
print(f"  k_search_results_v3.csv")
