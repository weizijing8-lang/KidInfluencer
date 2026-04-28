"""
Collect video descriptions for all existing videos.
We already have video IDs from the raw data, so we just need to call
videos.list with part=snippet to get descriptions.

Quota: 1 unit per request, 50 videos per request
98,616 videos / 50 = 1,973 requests = 1,973 units (well within 10,000 daily)
"""

import json
import os
import time
import requests
from pathlib import Path

API_KEY = "AIzaSyC17NxPT0HPVaXihtyNtmvBhH4Mh6GdowU"
RAW_DIR = Path("/home/ubuntu/KidInfluencer/data/raw")
DESC_DIR = Path("/home/ubuntu/KidInfluencer/data/descriptions")
DESC_DIR.mkdir(parents=True, exist_ok=True)

def get_all_video_ids():
    """Load all video IDs from raw JSON files."""
    channel_videos = {}
    for f in sorted(RAW_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fp:
            data = json.load(fp)
        if data.get("error") or not data.get("videos"):
            continue
        short_name = data["short_name"]
        video_ids = [v["id"] for v in data["videos"]]
        channel_videos[short_name] = video_ids
    return channel_videos


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
        
        time.sleep(0.1)  # Rate limiting
    
    return descriptions


def main():
    print("=" * 60, flush=True)
    print("COLLECTING VIDEO DESCRIPTIONS", flush=True)
    print("=" * 60, flush=True)
    
    channel_videos = get_all_video_ids()
    total_videos = sum(len(v) for v in channel_videos.values())
    print(f"Total: {total_videos:,} videos across {len(channel_videos)} channels", flush=True)
    
    quota_used = 0
    
    for idx, (channel, video_ids) in enumerate(sorted(channel_videos.items())):
        # Check if already collected
        out_file = DESC_DIR / f"{channel}_desc.json"
        if out_file.exists():
            with open(out_file) as fp:
                existing = json.load(fp)
            if len(existing) >= len(video_ids) * 0.9:  # Allow 10% margin
                print(f"  [{idx+1}/{len(channel_videos)}] {channel}: already collected ({len(existing)} descs)", flush=True)
                continue
        
        print(f"  [{idx+1}/{len(channel_videos)}] {channel}: fetching {len(video_ids)} descriptions...", flush=True)
        
        descs = fetch_descriptions(video_ids)
        
        with open(out_file, 'w') as fp:
            json.dump(descs, fp)
        
        n_requests = (len(video_ids) + 49) // 50
        quota_used += n_requests
        
        print(f"    Got {len(descs)} descriptions (quota used: {quota_used})", flush=True)
    
    print(f"\nDone! Total quota used: {quota_used}", flush=True)
    print(f"Descriptions saved to: {DESC_DIR}", flush=True)


if __name__ == "__main__":
    main()
