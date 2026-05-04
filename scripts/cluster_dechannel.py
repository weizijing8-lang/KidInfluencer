"""
Re-cluster kidfluencer videos using de-channelized content features.
Goal: clusters should reflect CONTENT STRATEGIES that span multiple channels,
not individual channel styles.

Features used: 
- Binary content type indicators (challenge, unboxing, prank, etc.)
- Emotional manipulation signals (clickbait, urgency, mystery, conflict)
- Commercialization signals (brand, money, giveaway)
- Structural features (title length, caps ratio, etc.)
"""
import pandas as pd
import numpy as np
import json, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter

print("Loading features...")
features_full = pd.read_csv('analysis_discovery/content_features_full.csv')
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')

# Select only the content features (exclude id, channel, viewCount)
feature_cols = [c for c in features_full.columns if c not in ['id', 'channel_short_name', 'viewCount']]
X = features_full[feature_cols].values

print(f"Feature matrix: {X.shape} ({len(feature_cols)} features)")
print(f"Features: {feature_cols}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# Find optimal K with channel diversity constraint
# ============================================================
print("\nSearching for optimal K...")
results = []
for k in range(5, 25):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    
    # Calculate channel diversity per cluster
    min_channels = 25
    max_concentration = 0
    for c in range(k):
        mask = labels == c
        channels = features_full.loc[mask, 'channel_short_name']
        n_ch = channels.nunique()
        top1_pct = channels.value_counts().iloc[0] / len(channels) if len(channels) > 0 else 1
        min_channels = min(min_channels, n_ch)
        max_concentration = max(max_concentration, top1_pct)
    
    results.append({
        'k': k, 'silhouette': sil, 
        'min_channels': min_channels, 'max_concentration': max_concentration
    })
    print(f"  K={k:2d}: silhouette={sil:.4f}, min_channels={min_channels}, max_top1={max_concentration:.1%}")

# Choose K: good silhouette + reasonable channel diversity
# Prefer K where max_concentration < 50%
results_df = pd.DataFrame(results)
# Filter for acceptable diversity
good = results_df[results_df['max_concentration'] < 0.6]
if len(good) > 0:
    best_k = good.loc[good['silhouette'].idxmax(), 'k']
else:
    best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']

print(f"\nSelected K={int(best_k)} (best silhouette with max_concentration < 60%)")

# ============================================================
# Final clustering with selected K
# ============================================================
print(f"\nRunning final K-Means with K={int(best_k)}...")
km_final = KMeans(n_clusters=int(best_k), random_state=42, n_init=20, max_iter=500)
final_labels = km_final.fit_predict(X_scaled)

features_full['cluster_v2'] = final_labels
df['cluster_v2'] = final_labels

# ============================================================
# Analyze clusters
# ============================================================
print(f"\n{'='*100}")
print(f"CLUSTER ANALYSIS (K={int(best_k)}, de-channelized features)")
print(f"{'='*100}")

cluster_summaries = []
for c in range(int(best_k)):
    mask = df['cluster_v2'] == c
    cl = df[mask]
    cl_feat = features_full[mask]
    
    n_videos = len(cl)
    n_channels = cl['channel_short_name'].nunique()
    median_views = cl['viewCount'].median()
    overall_median = df['viewCount'].median()
    view_boost = (median_views - overall_median) / overall_median
    
    # Top channels
    top_channels = cl['channel_short_name'].value_counts().head(5)
    top1_pct = top_channels.iloc[0] / n_videos
    
    # Dominant features
    binary_features = [c for c in feature_cols if cl_feat[c].max() <= 1 and cl_feat[c].dtype in ['int64', 'float64']]
    feat_means = cl_feat[binary_features].mean()
    overall_means = features_full[binary_features].mean()
    # Features that are significantly overrepresented in this cluster
    enriched = ((feat_means - overall_means) / (overall_means + 0.01)).sort_values(ascending=False)
    top_features = enriched.head(5)
    
    summary = {
        'cluster': c,
        'n_videos': n_videos,
        'n_channels': n_channels,
        'top1_channel': top_channels.index[0],
        'top1_pct': top1_pct,
        'median_views': median_views,
        'view_boost': view_boost,
        'top_enriched_features': list(top_features.index),
        'enrichment_scores': list(top_features.values)
    }
    cluster_summaries.append(summary)
    
    print(f"\nCluster {c}: {n_videos} videos, {n_channels} channels, boost={view_boost*100:+.0f}%")
    print(f"  Top channels: {', '.join([f'{ch}({cnt})' for ch, cnt in top_channels.head(3).items()])}")
    print(f"  Top-1 concentration: {top1_pct:.1%}")
    print(f"  Enriched features: {', '.join([f'{f}({v:.1f}x)' for f, v in zip(top_features.index, top_features.values)])}")
    
    # Sample titles
    sample_titles = cl['title'].sample(min(3, len(cl)), random_state=42).tolist()
    for t in sample_titles:
        print(f"    \"{t[:70]}\"")

# ============================================================
# Save results
# ============================================================
print(f"\n{'='*100}")
print("SUMMARY TABLE")
print(f"{'='*100}")
print(f"\n{'C':>3} {'N':>6} {'Ch':>4} {'Top1':>6} {'Top1%':>6} {'Boost':>8} {'Key Features':<50}")
print("-" * 90)
for s in sorted(cluster_summaries, key=lambda x: x['view_boost'], reverse=True):
    feats = ', '.join(s['top_enriched_features'][:3])
    print(f"{s['cluster']:>3} {s['n_videos']:>6} {s['n_channels']:>4} {s['top1_channel']:<6} {s['top1_pct']:>5.1%} {s['view_boost']*100:>+7.0f}% {feats:<50}")

# Save
features_full.to_csv('analysis_discovery/videos_with_clusters_v2.csv', index=False)
with open('analysis_discovery/cluster_v2_summaries.json', 'w') as f:
    json.dump(cluster_summaries, f, indent=2, default=str)

# Also save the K search results
results_df.to_csv('analysis_discovery/k_search_results_v2.csv', index=False)

print(f"\nSaved:")
print(f"  analysis_discovery/videos_with_clusters_v2.csv")
print(f"  analysis_discovery/cluster_v2_summaries.json")
print(f"  analysis_discovery/k_search_results_v2.csv")
