"""
Fast version: Skip HDBSCAN, use K-Means on PCA-reduced embeddings.
Embeddings already saved from previous run.
"""
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = 'analysis_discovery'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

# ============ LOAD ============
print("Loading data and pre-computed embeddings...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
embeddings = np.load(f'{OUTPUT_DIR}/title_embeddings.npy')
print(f"Videos: {len(df)}, Embedding shape: {embeddings.shape}")

# ============ PCA REDUCTION ============
print("\nReducing dimensions with PCA (384 → 50)...")
pca = PCA(n_components=50, random_state=42)
emb_reduced = pca.fit_transform(embeddings)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# ============ K-MEANS CLUSTERING ============
# Try multiple k values to find optimal
from sklearn.metrics import silhouette_score

print("\nFinding optimal k...")
scores = {}
for k in [8, 12, 15, 20, 25, 30]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(emb_reduced)
    sil = silhouette_score(emb_reduced, labels, sample_size=5000, random_state=42)
    scores[k] = sil
    print(f"  k={k}: silhouette={sil:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nBest k={best_k} (silhouette={scores[best_k]:.4f})")

# Use k=15 as a good balance between granularity and interpretability
K = 15
print(f"\nUsing k={K} for final clustering...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans.fit_predict(emb_reduced)

# ============ CLUSTER ANALYSIS ============
print(f"\n{'='*60}")
print("CLUSTER ANALYSIS")
print(f"{'='*60}")

overall_median = df['viewCount'].median()
stop_words = {'the','a','an','and','or','is','in','on','to','for','of','my','i','we','it',
              'this','that','with','at','from','by','our','you','your','her','his','he','she',
              'they','them','was','were','are','been','be','have','has','had','do','does','did',
              'will','would','could','should','can','may','might','shall','not','no','but','if',
              'so','as','up','out','about','just','get','got','go','going','went','come','came',
              'back','all','one','two','new','first','last','day','time','make','made','like',
              'very','when','what','how','who','where','why','which','vs','part','episode','ep'}

cluster_info = []
for k in range(K):
    mask = df['cluster'] == k
    cluster_df = df[mask]
    n = len(cluster_df)
    median_views = cluster_df['viewCount'].median()
    boost = median_views / overall_median - 1
    
    # Representative titles (closest to centroid)
    cluster_embs = emb_reduced[mask]
    centroid = kmeans.cluster_centers_[k]
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    closest_idx = np.argsort(dists)[:8]
    rep_titles = cluster_df.iloc[closest_idx]['title'].tolist()
    
    # Top words
    all_words = ' '.join(cluster_df['title'].fillna('')).lower().split()
    word_counts = Counter(w for w in all_words if w not in stop_words and len(w) > 2)
    top_words = [w for w, c in word_counts.most_common(10)]
    
    # Channel distribution
    ch_dist = cluster_df['channel_short_name'].value_counts().head(3)
    top_channels = [f"{ch}({cnt})" for ch, cnt in ch_dist.items()]
    
    info = {
        'cluster': k,
        'n_videos': n,
        'pct': n / len(df),
        'median_views': float(median_views),
        'view_boost': float(boost),
        'top_words': top_words,
        'representative_titles': rep_titles,
        'top_channels': top_channels
    }
    cluster_info.append(info)
    
    print(f"\n--- Cluster {k} (n={n}, {n/len(df):.1%}) | Median views: {median_views:,.0f} | Boost: {boost:+.0%} ---")
    print(f"  Top words: {', '.join(top_words[:7])}")
    print(f"  Top channels: {', '.join(top_channels)}")
    print(f"  Representative titles:")
    for t in rep_titles[:5]:
        print(f"    • {t}")

# ============ SORT BY VIEW BOOST ============
print(f"\n\n{'='*60}")
print("CLUSTERS RANKED BY VIEW BOOST")
print(f"{'='*60}")
print(f"{'Cluster':>8} {'N':>6} {'%':>6} {'Median Views':>14} {'Boost':>8}  Top Words")
print("-" * 80)
for info in sorted(cluster_info, key=lambda x: x['view_boost'], reverse=True):
    print(f"{info['cluster']:>8} {info['n_videos']:>6} {info['pct']:>5.1%} {info['median_views']:>14,.0f} {info['view_boost']:>+7.0%}  {', '.join(info['top_words'][:5])}")

# ============ T-SNE VISUALIZATION ============
print("\n\nCreating t-SNE visualization...")
sample_n = 8000
np.random.seed(42)
sample_idx = np.random.choice(len(df), sample_n, replace=False)
sample_embs = emb_reduced[sample_idx]
sample_clusters = df.iloc[sample_idx]['cluster'].values
sample_views = df.iloc[sample_idx]['viewCount'].values

tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
coords = tsne.fit_transform(sample_embs)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Color by cluster
cmap = plt.cm.get_cmap('tab20', K)
for k in range(K):
    mask = sample_clusters == k
    if mask.sum() > 0:
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[cmap(k)], 
                       alpha=0.4, s=5, label=f'C{k}')
axes[0].set_title('t-SNE: Clusters (K-Means, k=15)', fontsize=13)
axes[0].legend(fontsize=7, ncol=3, loc='upper right')

# Color by log views
log_views = np.log10(sample_views + 1)
sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=log_views, 
                     cmap='RdYlGn', alpha=0.4, s=5)
axes[1].set_title('t-SNE: colored by log10(views)', fontsize=13)
plt.colorbar(sc, ax=axes[1], label='log10(views)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/tsne_clusters_v2.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ BAR CHART OF VIEW BOOST ============
fig, ax = plt.subplots(figsize=(12, 6))
sorted_info = sorted(cluster_info, key=lambda x: x['view_boost'], reverse=True)
clusters = [f"C{i['cluster']}" for i in sorted_info]
boosts = [i['view_boost'] * 100 for i in sorted_info]
colors = ['#d32f2f' if b > 50 else '#f57c00' if b > 0 else '#388e3c' for b in boosts]

bars = ax.bar(clusters, boosts, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('View Boost vs Median (%)', fontsize=11)
ax.set_title('Platform Reward by Content Cluster\n(Discovered via Unsupervised Clustering)', fontsize=13)

# Add top words as labels
for i, (bar, info) in enumerate(zip(bars, sorted_info)):
    label = ', '.join(info['top_words'][:3])
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 5, label, 
            ha='center', va='bottom', fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/view_boost_by_cluster.png', dpi=200, bbox_inches='tight')
plt.close()

# ============ SAVE ============
with open(f'{OUTPUT_DIR}/cluster_info.json', 'w') as f:
    json.dump(cluster_info, f, indent=2, default=str)

df[['title', 'viewCount', 'channel_short_name', 'cluster']].to_csv(
    f'{OUTPUT_DIR}/videos_with_clusters.csv', index=False)

print(f"\n=== COMPLETE ===")
print(f"Saved: {OUTPUT_DIR}/cluster_info.json")
print(f"Saved: {OUTPUT_DIR}/figures/tsne_clusters_v2.png")
print(f"Saved: {OUTPUT_DIR}/figures/view_boost_by_cluster.png")
