"""
Download thumbnails for all 2306 sample videos.
YouTube thumbnails follow a predictable URL pattern:
https://img.youtube.com/vi/{video_id}/maxresdefault.jpg (high res)
https://img.youtube.com/vi/{video_id}/hqdefault.jpg (medium res fallback)
"""

import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

THUMB_DIR = '/home/ubuntu/KidInfluencer/data/thumbnails_sample'
os.makedirs(THUMB_DIR, exist_ok=True)

# Load sample video IDs
sample = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/classification_5dim_sample.csv')
video_ids = sample['id'].tolist()

# Check which ones we already have
existing = set(f.split('.')[0] for f in os.listdir(THUMB_DIR))
to_download = [vid for vid in video_ids if vid not in existing]

# Also check the old thumbnail directory
old_thumb_dir = '/home/ubuntu/KidInfluencer/data/thumbnails'
if os.path.exists(old_thumb_dir):
    old_existing = set(f.split('.')[0] for f in os.listdir(old_thumb_dir))
    # Copy existing ones
    import shutil
    copied = 0
    for vid in video_ids:
        if vid in old_existing and vid not in existing:
            # Find the file
            for ext in ['.jpg', '.png', '.webp']:
                src = os.path.join(old_thumb_dir, vid + ext)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(THUMB_DIR, vid + ext))
                    copied += 1
                    break
    print(f"Copied {copied} thumbnails from old directory")
    existing = set(f.split('.')[0] for f in os.listdir(THUMB_DIR))
    to_download = [vid for vid in video_ids if vid not in existing]

print(f"Total sample videos: {len(video_ids)}")
print(f"Already have: {len(video_ids) - len(to_download)}")
print(f"Need to download: {len(to_download)}")

def download_thumbnail(video_id):
    """Download thumbnail for a video, trying maxres first then hq"""
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                filepath = os.path.join(THUMB_DIR, f"{video_id}.jpg")
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                return True
        except:
            continue
    return False

# Download in parallel
print(f"\nDownloading {len(to_download)} thumbnails...")
success = 0
failed = 0
start = time.time()

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(download_thumbnail, vid): vid for vid in to_download}
    for i, future in enumerate(as_completed(futures)):
        if future.result():
            success += 1
        else:
            failed += 1
        
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(to_download) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(to_download)} ({success} ok, {failed} fail) - {remaining:.0f}s remaining")

elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f}s")
print(f"  Downloaded: {success}")
print(f"  Failed: {failed}")
print(f"  Total available: {len(os.listdir(THUMB_DIR))}")

# Final check
final_existing = set(f.split('.')[0] for f in os.listdir(THUMB_DIR))
final_overlap = set(video_ids) & final_existing
print(f"  Coverage of sample: {len(final_overlap)}/{len(video_ids)} ({len(final_overlap)/len(video_ids)*100:.1f}%)")
