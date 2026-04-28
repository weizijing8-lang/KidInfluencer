"""
Collect comments for sampled family channel videos.
Strategy: For each family channel, pick top 50 highest exploitation score videos
+ 50 lowest exploitation score videos = 100 per channel.
Get top 20 comments per video.

25 family channels × 100 videos × 1 request = 2,500 units
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

API_KEY = "AIzaSyC17NxPT0HPVaXihtyNtmvBhH4Mh6GdowU"
DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
RESULTS_V2 = DATA_DIR / "results_v2"
COMMENTS_DIR = DATA_DIR / "comments"
COMMENTS_DIR.mkdir(parents=True, exist_ok=True)


def get_sample_videos():
    """Get top 50 + bottom 50 exploitation score videos per family channel."""
    df = pd.read_csv(RESULTS_V2 / "full_results_v2.csv")
    family = df[df['channel_category'] == 'family'].copy()
    
    samples = []
    for channel, group in family.groupby('channel_short_name'):
        group = group.sort_values('exploit_score_v2', ascending=False)
        n = min(50, len(group) // 2)
        top = group.head(n)
        bottom = group.tail(n)
        combined = pd.concat([top, bottom])
        combined['sample_type'] = ['high'] * len(top) + ['low'] * len(bottom)
        samples.append(combined)
    
    result = pd.concat(samples, ignore_index=True)
    print(f"Sampled {len(result)} videos from {result['channel_short_name'].nunique()} family channels", flush=True)
    return result


def fetch_comments(video_id, max_results=20):
    """Fetch top comments for a video."""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": API_KEY,
        "maxResults": max_results,
        "order": "relevance",
        "textFormat": "plainText",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 403:
            # Comments disabled
            return []
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        comments = []
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "publishedAt": snippet.get("publishedAt", ""),
            })
        return comments
    except Exception:
        return []


def main():
    print("=" * 60, flush=True)
    print("COLLECTING COMMENTS (SAMPLED)", flush=True)
    print("=" * 60, flush=True)
    
    sample = get_sample_videos()
    
    all_comments = []
    quota_used = 0
    total = len(sample)
    
    for idx, (_, row) in enumerate(sample.iterrows()):
        video_id = row['id']
        channel = row['channel_short_name']
        
        comments = fetch_comments(video_id)
        quota_used += 1
        
        for c in comments:
            all_comments.append({
                'video_id': video_id,
                'channel': channel,
                'video_title': row['title'],
                'exploit_score_v2': row['exploit_score_v2'],
                'sample_type': row['sample_type'],
                'comment_text': c['text'],
                'comment_likes': c['likes'],
                'comment_date': c['publishedAt'],
            })
        
        done = idx + 1
        if done % 100 == 0 or done == total:
            print(f"  Progress: {done}/{total} videos, {len(all_comments)} comments, quota={quota_used}", flush=True)
        
        time.sleep(0.1)
    
    # Save
    comments_df = pd.DataFrame(all_comments)
    comments_df.to_csv(COMMENTS_DIR / "sampled_comments.csv", index=False)
    
    print(f"\nDone! {len(comments_df)} comments from {total} videos", flush=True)
    print(f"Quota used: {quota_used}", flush=True)
    
    # Quick stats
    high = comments_df[comments_df['sample_type'] == 'high']
    low = comments_df[comments_df['sample_type'] == 'low']
    print(f"\nHigh exploitation videos: {high['video_id'].nunique()} videos, {len(high)} comments", flush=True)
    print(f"Low exploitation videos:  {low['video_id'].nunique()} videos, {len(low)} comments", flush=True)


if __name__ == "__main__":
    main()
