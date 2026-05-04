"""
Re-cluster with K=7 using de-channelized content features.
K=7 was selected because it has the lowest max_concentration (44.8%)
among all K values tested, meaning best channel diversity.
"""
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading features...")
features_full = pd.read_csv('analysis_discovery/content_features_full.csv')
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')

# Select only the content features (exclude id, channel, viewCount)
feature_cols = [c for c in features_full.columns if c not in ['id', 'channel_short_name', 'viewCount']]
X = features_full[feature_cols].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K=7 clustering
print("Running K-Means with K=7...")
km = KMeans(n_clusters=7, random_state=42, n_init=20, max_iter=500)
labels = km.fit_predict(X_scaled)

features_full['cluster_v2'] = labels
df['cluster_v2'] = labels

# ============================================================
# Detailed cluster analysis
# ============================================================
print(f"\n{'='*110}")
print(f"CLUSTER ANALYSIS (K=7, de-channelized features)")
print(f"{'='*110}")

overall_median = df['viewCount'].median()
cluster_summaries = []

for c in range(7):
    mask = df['cluster_v2'] == c
    cl = df[mask]
    cl_feat = features_full[mask]
    
    n_videos = len(cl)
    n_channels = cl['channel_short_name'].nunique()
    median_views = cl['viewCount'].median()
    view_boost = (median_views - overall_median) / overall_median
    
    # Top channels
    top_channels = cl['channel_short_name'].value_counts().head(5)
    top1_pct = top_channels.iloc[0] / n_videos
    top3_pct = top_channels.head(3).sum() / n_videos
    
    # Enriched features
    binary_features = [c for c in feature_cols if features_full[c].max() <= 1 and features_full[c].dtype in ['int64', 'float64']]
    feat_means = cl_feat[binary_features].mean()
    overall_means = features_full[binary_features].mean()
    enriched = ((feat_means - overall_means) / (overall_means + 0.01)).sort_values(ascending=False)
    top_features = enriched.head(5)
    
    # Sample titles from different channels
    sample_titles = []
    for ch in cl['channel_short_name'].unique()[:4]:
        t = cl[cl['channel_short_name'] == ch]['title'].iloc[0]
        sample_titles.append(f"[{ch}] {t[:65]}")
    
    summary = {
        'cluster': c,
        'n_videos': n_videos,
        'pct_of_total': n_videos / len(df),
        'n_channels': n_channels,
        'top1_channel': top_channels.index[0],
        'top1_pct': float(top1_pct),
        'top3_pct': float(top3_pct),
        'top_channels_list': {ch: int(cnt) for ch, cnt in top_channels.items()},
        'median_views': float(median_views),
        'view_boost': float(view_boost),
        'top_enriched_features': list(top_features.index),
        'enrichment_scores': [float(v) for v in top_features.values],
        'sample_titles': sample_titles[:4]
    }
    cluster_summaries.append(summary)
    
    print(f"\n{'─'*110}")
    print(f"Cluster {c}: {n_videos} videos ({n_videos/len(df)*100:.1f}%), {n_channels} channels, view_boost={view_boost*100:+.0f}%")
    print(f"  Channel diversity: top1={top1_pct:.1%} ({top_channels.index[0]}), top3={top3_pct:.1%}")
    print(f"  Top channels: {', '.join([f'{ch}({cnt})' for ch, cnt in top_channels.items()])}")
    print(f"  Enriched features: {', '.join([f'{f}({v:.1f}x)' for f, v in zip(top_features.index, top_features.values)])}")
    print(f"  Sample titles:")
    for t in sample_titles[:4]:
        print(f"    {t}")

# ============================================================
# Cross-channel validation: for each cluster, show % from each channel
# ============================================================
print(f"\n\n{'='*110}")
print("CHANNEL x CLUSTER DISTRIBUTION (% of each channel's videos in each cluster)")
print(f"{'='*110}")

cross_tab = pd.crosstab(df['channel_short_name'], df['cluster_v2'], normalize='index') * 100
print(cross_tab.round(1).to_string())

# ============================================================
# Statistical tests: view boost significance
# ============================================================
from scipy import stats

print(f"\n\n{'='*110}")
print("VIEW BOOST STATISTICAL SIGNIFICANCE (vs overall median)")
print(f"{'='*110}")

for c in range(7):
    cl_views = df[df['cluster_v2'] == c]['viewCount'].values
    other_views = df[df['cluster_v2'] != c]['viewCount'].values
    stat, p = stats.mannwhitneyu(cl_views, other_views, alternative='two-sided')
    boost = cluster_summaries[c]['view_boost']
    print(f"  C{c}: boost={boost*100:+.0f}%, Mann-Whitney U p={p:.2e} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

# ============================================================
# Save results
# ============================================================
features_full.to_csv('analysis_discovery/videos_with_clusters_v2.csv', index=False)
with open('analysis_discovery/cluster_v2_summaries.json', 'w') as f:
    json.dump(cluster_summaries, f, indent=2, default=str)

# Save cross-tab
cross_tab.to_csv('analysis_discovery/channel_cluster_crosstab_v2.csv')

print(f"\n\nSaved:")
print(f"  analysis_discovery/videos_with_clusters_v2.csv")
print(f"  analysis_discovery/cluster_v2_summaries.json")
print(f"  analysis_discovery/channel_cluster_crosstab_v2.csv")
