"""
Collect video metadata from YouTube family vlog channels via YouTube Data API v3.
Uses search + channel listing to get all video titles, view counts, publish dates.
"""
import requests
import json
import time
import os
from datetime import datetime

API_KEY = "AIzaSyA_your_key_here"  # We'll use a free approach instead

# We'll use the YouTube Data API via the invidious API (public, no key needed)
# Or use yt-dlp to extract channel video lists

def get_channel_videos_ytdlp(channel_url, output_file):
    """Use yt-dlp to get all video metadata from a channel."""
    import subprocess
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-download",
        channel_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if line:
            try:
                data = json.loads(line)
                videos.append({
                    'id': data.get('id', ''),
                    'title': data.get('title', ''),
                    'view_count': data.get('view_count', 0),
                    'duration': data.get('duration', 0),
                    'upload_date': data.get('upload_date', ''),
                    'description': data.get('description', ''),
                    'like_count': data.get('like_count', 0),
                    'comment_count': data.get('comment_count', 0),
                })
            except json.JSONDecodeError:
                continue
    
    with open(output_file, 'w') as f:
        json.dump(videos, f, indent=2)
    
    print(f"Collected {len(videos)} videos from {channel_url}")
    return videos


# Target channels - mix of controversial and normal family channels
CHANNELS = {
    # Controversial / known exploitation cases
    "8passengers": "https://www.youtube.com/@8passengers/videos",  # Ruby Franke - convicted
    "acefamily": "https://www.youtube.com/@TheACEFamily/videos",  # Known for clickbait with kids
    "labrantfam": "https://www.youtube.com/@TheLaBrantFam/videos",  # Known for pranks on kids
    
    # Large family channels
    "ryansworld": "https://www.youtube.com/@RyansWorld/videos",  # Ryan Kaji - biggest kid channel
    "familyfunpack": "https://www.youtube.com/@FamilyFunPack/videos",
    
    # Control group - adult-only vloggers (no kids)
    "caseyneistat": "https://www.youtube.com/@casey/videos",  # Adult vlogger
    "markwiens": "https://www.youtube.com/@MarkWiens/videos",  # Food vlogger, no kids
}

if __name__ == "__main__":
    os.makedirs("/home/ubuntu/pilot/data", exist_ok=True)
    
    for name, url in CHANNELS.items():
        output_file = f"/home/ubuntu/pilot/data/{name}.json"
        if os.path.exists(output_file):
            print(f"Skipping {name}, already collected")
            continue
        
        print(f"\nCollecting: {name} ({url})")
        try:
            videos = get_channel_videos_ytdlp(url, output_file)
            print(f"  -> {len(videos)} videos saved to {output_file}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
        
        time.sleep(2)  # Be polite
