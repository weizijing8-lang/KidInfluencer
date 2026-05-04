"""
Data-driven discovery of manipulation patterns in kidfluencer video titles.
Pipeline: Sentence-BERT embedding → HDBSCAN clustering → LLM interpretation → View boost analysis
"""
import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============ CONFIG ============
OUTPUT_DIR = 'analysis_discovery'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

# ============ LOAD DATA ============
print("Loading data...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
print(f"Total videos: {len(df)}")
print(f"Channels: {df['channel_short_name'].nunique()}")

# ============ STEP 1: SENTENCE EMBEDDINGS ============
print("\nStep 1: Generating sentence embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
titles = df['title'].fillna('').tolist()
embeddings = model.encode(titles, show_progress_bar=True, batch_size=256)
print(f"Embedding shape: {embeddings.shape}")

# Save embeddings
np.save(f'{OUTPUT_DIR}/title_embeddings.npy', embeddings)

# ============ STEP 2: CLUSTERING ============
print("\nStep 2: Clustering with HDBSCAN...")

# HDBSCAN for natural cluster discovery
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,  # Minimum 100 videos per cluster
    min_samples=20,
    metric='euclidean',
    cluster_selection_method='eom'
)
cluster_labels = clusterer.fit_predict(embeddings)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_pct = (cluster_labels == -1).mean()
print(f"HDBSCAN found {n_clusters} clusters, {noise_pct:.1%} noise")

# Also try K-Means for comparison (more stable)
print("\nAlso running K-Means (k=20) for finer granularity...")
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(embeddings)

# ============ STEP 3: ANALYZE CLUSTERS ============
print("\nStep 3: Analyzing clusters...")

df['hdbscan_cluster'] = cluster_labels
df['kmeans_cluster'] = kmeans_labels

# For each K-Means cluster, get top titles and stats
print("\n=== K-MEANS CLUSTER ANALYSIS ===")
cluster_info = []
for k in range(20):
    mask = df['kmeans_cluster'] == k
    cluster_df = df[mask]
    n = len(cluster_df)
    median_views = cluster_df['viewCount'].median()
    mean_views = cluster_df['viewCount'].mean()
    
    # Get representative titles (closest to centroid)
    cluster_embs = embeddings[mask]
    centroid = kmeans.cluster_centers_[k]
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    closest_idx = np.argsort(dists)[:10]
    rep_titles = cluster_df.iloc[closest_idx]['title'].tolist()
    
    # Get most common words
    from collections import Counter
    all_words = ' '.join(cluster_df['title'].fillna('')).lower().split()
    stop_words = {'the','a','an','and','or','is','in','on','to','for','of','my','i','we','it','this','that','with','at','from','by','our','you','your','her','his','he','she','they','them','was','were','are','been','be','have','has','had','do','does','did','will','would','could','should','can','may','might','shall','not','no','but','if','so','as','up','out','about','just','get','got','go','going','went','come','came','back','all','one','two','new','first','last','day','time','make','made','like','very','when','what','how','who','where','why','which'}
    word_counts = Counter(w for w in all_words if w not in stop_words and len(w) > 2)
    top_words = word_counts.most_common(10)
    
    info = {
        'cluster': k,
        'n_videos': n,
        'median_views': median_views,
        'mean_views': mean_views,
        'top_words': [w for w,c in top_words],
        'representative_titles': rep_titles[:5]
    }
    cluster_info.append(info)
    
    print(f"\nCluster {k} (n={n}, median_views={median_views:,.0f}):")
    print(f"  Top words: {', '.join(info['top_words'][:7])}")
    print(f"  Examples:")
    for t in rep_titles[:3]:
        print(f"    - {t}")

# ============ STEP 4: VIEW BOOST ANALYSIS ============
print("\n\n=== VIEW BOOST BY CLUSTER ===")
overall_median = df['viewCount'].median()
print(f"Overall median views: {overall_median:,.0f}\n")

cluster_stats = []
for k in range(20):
    mask = df['kmeans_cluster'] == k
    med = df[mask]['viewCount'].median()
    boost = med / overall_median - 1
    n = mask.sum()
    cluster_stats.append({'cluster': k, 'n': n, 'median_views': med, 'boost': boost})
    
cluster_stats_df = pd.DataFrame(cluster_stats).sort_values('boost', ascending=False)
print(cluster_stats_df.to_string(index=False))

# ============ STEP 5: T-SNE VISUALIZATION ============
print("\nStep 5: Creating t-SNE visualization...")
# Sample for speed
sample_n = min(10000, len(df))
np.random.seed(42)
sample_idx = np.random.choice(len(df), sample_n, replace=False)
sample_embs = embeddings[sample_idx]
sample_labels = kmeans_labels[sample_idx]
sample_views = df.iloc[sample_idx]['viewCount'].values

tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
coords = tsne.fit_transform(sample_embs)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Color by cluster
scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=sample_labels, 
                           cmap='tab20', alpha=0.4, s=3)
axes[0].set_title('t-SNE colored by cluster', fontsize=12)
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# Color by log views
log_views = np.log10(sample_views + 1)
scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=log_views, 
                           cmap='RdYlGn', alpha=0.4, s=3)
axes[1].set_title('t-SNE colored by log(views)', fontsize=12)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=axes[1], label='log10(views)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/tsne_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved t-SNE figure")

# ============ STEP 6: SAVE FOR LLM INTERPRETATION ============
# Save cluster info for LLM to interpret
with open(f'{OUTPUT_DIR}/cluster_info.json', 'w') as f:
    json.dump(cluster_info, f, indent=2, default=str)

# Save summary
cluster_stats_df.to_csv(f'{OUTPUT_DIR}/cluster_view_stats.csv', index=False)
df.to_csv(f'{OUTPUT_DIR}/videos_with_clusters.csv', index=False)

print(f"\n=== COMPLETE ===")
print(f"Saved: {OUTPUT_DIR}/cluster_info.json")
print(f"Saved: {OUTPUT_DIR}/cluster_view_stats.csv")
print(f"Saved: {OUTPUT_DIR}/figures/tsne_clusters.png")
