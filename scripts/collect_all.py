"""
Collect video metadata from YouTube family vlog channels using yt-dlp.
Saves title, view_count, upload_date, duration for each video.
"""
import subprocess
import json
import os
import time

CHANNELS = {
    # Family / Kidfluencer channels (treatment group)
    "acefamily": "https://www.youtube.com/@TheACEFamily/videos",
    "labrantfam": "https://www.youtube.com/@TheLaBrantFam/videos",
    "ryansworld": "https://www.youtube.com/@RyansWorld/videos",
    "familyfunpack": "https://www.youtube.com/@FamilyFunPack/videos",
    
    # Control group - adult-only vloggers
    "caseyneistat": "https://www.youtube.com/@casey/videos",
    "markwiens": "https://www.youtube.com/@MarkWiens/videos",
}

def collect_channel(name, url, output_dir):
    output_file = os.path.join(output_dir, f"{name}.json")
    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        print(f"[SKIP] {name}: already have {len(data)} videos")
        return
    
    print(f"[COLLECTING] {name}: {url}")
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-download",
        url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        try:
            d = json.loads(line)
            videos.append({
                'id': d.get('id', ''),
                'title': d.get('title', ''),
                'view_count': d.get('view_count', 0),
                'duration': d.get('duration', 0),
                'upload_date': d.get('upload_date', ''),
                'like_count': d.get('like_count', 0),
                'comment_count': d.get('comment_count', 0),
                'playlist_index': d.get('playlist_index', 0),
                'n_entries': d.get('n_entries', 0),
            })
        except json.JSONDecodeError:
            continue
    
    with open(output_file, 'w') as f:
        json.dump(videos, f, indent=2)
    
    print(f"[DONE] {name}: {len(videos)} videos saved")
    return videos

if __name__ == "__main__":
    output_dir = "/home/ubuntu/pilot/data"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, url in CHANNELS.items():
        try:
            collect_channel(name, url, output_dir)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
        time.sleep(3)
    
    # Summary
    print("\n=== SUMMARY ===")
    for name in CHANNELS:
        fpath = os.path.join(output_dir, f"{name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            print(f"{name}: {len(data)} videos")
        else:
            print(f"{name}: FAILED")
