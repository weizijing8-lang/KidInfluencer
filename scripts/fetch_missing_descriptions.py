"""
Fetch video descriptions for the 58 channels missing from data/descriptions/.
Uses video IDs from data/llm_classifications_v2.csv.

YouTube API quota: 69 requests × 1 unit = 69 units (well within 10,000 daily limit)
"""
import json
import os
import time
import requests
import pandas as pd
from pathlib import Path

API_KEY = "AIzaSyC17NxPT0HPVaXihtyNtmvBhH4Mh6GdowU"
DESC_DIR = Path("/home/ubuntu/KidInfluencer/data/descriptions")
DESC_DIR.mkdir(parents=True, exist_ok=True)

# Load video data
llm = pd.read_csv('/home/ubuntu/KidInfluencer/data/llm_classifications_v2.csv')

# Channels that already have descriptions
existing = set(f.stem.replace('_desc', '') for f in DESC_DIR.glob('*_desc.json'))

# Get missing channels
all_channels = set(llm['channel_short_name'].unique())
missing_channels = sorted(all_channels - existing)

print(f"Total channels: {len(all_channels)}")
print(f"Already have: {len(existing & all_channels)}")
print(f"Missing: {len(missing_channels)}")
print()

def fetch_descriptions(video_ids):
    """Fetch descriptions for a list of video IDs (max 50 per request)."""
    descriptions = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        ids_str = ",".join(batch)
        
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet",
            "id": ids_str,
            "key": API_KEY,
            "maxResults": 50,
        }
        
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    Error {resp.status_code}: {resp.text[:200]}")
            time.sleep(2)
            continue
        
        data = resp.json()
        for item in data.get("items", []):
            vid = item["id"]
            desc = item.get("snippet", {}).get("description", "")
            tags = item.get("snippet", {}).get("tags", [])
            descriptions[vid] = {
                "description": desc,
                "tags": tags,
            }
        
        time.sleep(0.2)  # Rate limiting
    
    return descriptions

# Fetch for each missing channel
total_fetched = 0
quota_used = 0

for idx, channel in enumerate(missing_channels):
    out_file = DESC_DIR / f"{channel}_desc.json"
    
    # Skip if already exists
    if out_file.exists():
        print(f"  [{idx+1}/{len(missing_channels)}] {channel}: already exists, skipping")
        continue
    
    # Get video IDs for this channel
    channel_vids = llm[llm['channel_short_name'] == channel]['id'].tolist()
    
    if not channel_vids:
        print(f"  [{idx+1}/{len(missing_channels)}] {channel}: no videos found")
        continue
    
    print(f"  [{idx+1}/{len(missing_channels)}] {channel}: fetching {len(channel_vids)} descriptions...", end=" ")
    
    descs = fetch_descriptions(channel_vids)
    
    # Save
    with open(out_file, 'w') as fp:
        json.dump(descs, fp, ensure_ascii=False)
    
    n_requests = (len(channel_vids) + 49) // 50
    quota_used += n_requests
    total_fetched += len(descs)
    
    print(f"got {len(descs)} (quota: {quota_used})")

print(f"\n{'='*60}")
print(f"Done!")
print(f"  Total descriptions fetched: {total_fetched}")
print(f"  Total API quota used: {quota_used}")
print(f"  Files saved to: {DESC_DIR}")
print(f"{'='*60}")
