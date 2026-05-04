"""
Download YouTube thumbnails for a stratified sample across all 15 clusters.
YouTube thumbnail URL pattern: https://img.youtube.com/vi/{VIDEO_ID}/hqdefault.jpg
"""
import pandas as pd
import numpy as np
import os, requests, time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load data with cluster assignments
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
clusters = pd.read_csv('analysis_discovery/videos_with_clusters.csv')
df['cluster'] = clusters['cluster'].values

THUMB_DIR = 'data/thumbnails'
os.makedirs(THUMB_DIR, exist_ok=True)

# Check what we already have
existing = set(os.path.splitext(f)[0] for f in os.listdir(THUMB_DIR) if f.endswith('.jpg'))
print(f"Already have {len(existing)} thumbnails")

# Sample strategy: 100 per cluster = 1500 total (but skip already downloaded)
SAMPLE_PER_CLUSTER = 100
np.random.seed(42)

to_download = []
for k in range(15):
    cluster_df = df[df['cluster'] == k]
    # Exclude already downloaded
    available = cluster_df[~cluster_df['id'].isin(existing)]
    n_sample = min(SAMPLE_PER_CLUSTER, len(available))
    if n_sample > 0:
        sampled = available.sample(n=n_sample, random_state=42)
        to_download.extend(sampled['id'].tolist())
    print(f"  Cluster {k}: {len(cluster_df)} total, {len(cluster_df[cluster_df['id'].isin(existing)])} already have, sampling {n_sample} more")

print(f"\nTotal to download: {len(to_download)}")

def download_thumbnail(video_id):
    """Download a YouTube thumbnail."""
    url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    path = os.path.join(THUMB_DIR, f"{video_id}.jpg")
    if os.path.exists(path):
        return video_id, True
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 1000:  # Skip placeholder images
            with open(path, 'wb') as f:
                f.write(resp.content)
            return video_id, True
        return video_id, False
    except Exception as e:
        return video_id, False

# Download with thread pool
success = 0
fail = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(download_thumbnail, vid): vid for vid in to_download}
    for i, future in enumerate(as_completed(futures)):
        vid, ok = future.result()
        if ok:
            success += 1
        else:
            fail += 1
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(to_download)} (success={success}, fail={fail})")

print(f"\nDownload complete: {success} success, {fail} failed")
print(f"Total thumbnails now: {len(os.listdir(THUMB_DIR))}")
