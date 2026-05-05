#!/usr/bin/env python3
"""
Fix View Counts for Expanded Channels
=======================================
The original collection script used vid.get('viewCount') but the API returns
views in vid['stats']['views']. This script re-fetches channel videos and 
extracts the correct view count.
"""
import sys
import csv
import os
import time
from datetime import datetime
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
client = ApiClient()

DATA_DIR = '/home/ubuntu/KidInfluencer/data'
INPUT_CSV = os.path.join(DATA_DIR, 'expanded_channels_videos.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'expanded_channels_videos_fixed.csv')
DELAY = 1.5
MAX_VIDEOS_PER_CHANNEL = 500

def fetch_channel_videos_fixed(channel_id, max_videos=MAX_VIDEOS_PER_CHANNEL):
    """Fetch videos with correct view count extraction."""
    all_videos = []
    cursor = None
    pages = 0
    while len(all_videos) < max_videos:
        params = {'id': channel_id, 'filter': 'videos_latest', 'hl': 'en', 'gl': 'US'}
        if cursor:
            params['cursor'] = cursor
        try:
            resp = client.call_api('Youtube/get_channel_videos', query=params)
            if not resp:
                break
            contents = resp.get('contents', [])
            if not contents:
                break
            for item in contents:
                vid = item.get('video', item)
                video_id = vid.get('videoId', '')
                title = vid.get('title', '')
                published = vid.get('publishedTimeText', '')
                # CORRECT: views are in stats.views
                stats = vid.get('stats', {})
                view_count = stats.get('views', 0)
                if isinstance(view_count, str):
                    view_count = view_count.replace(',', '')
                    try:
                        view_count = int(view_count)
                    except:
                        view_count = 0
                
                if video_id and title:
                    all_videos.append({
                        'id': video_id,
                        'title': title,
                        'publishedAt': published,
                        'channelId': channel_id,
                        'viewCount': view_count,
                    })
            
            pages += 1
            cursor = resp.get('cursorNext', '')
            if not cursor:
                break
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [ERROR] fetch_videos page {pages}: {e}")
            break
    return all_videos[:max_videos]

def main():
    import pandas as pd
    
    # Load existing data to get channel IDs
    df = pd.read_csv(INPUT_CSV)
    channels = df.groupby('channel_short_name').first().reset_index()[['channel_short_name', 'channelId']]
    
    print(f"[{datetime.now()}] Fixing view counts for {len(channels)} channels")
    
    all_fixed = []
    
    for i, row in channels.iterrows():
        ch_name = row['channel_short_name']
        ch_id = row['channelId']
        
        print(f"[{i+1}/{len(channels)}] {ch_name} ({ch_id})...")
        
        videos = fetch_channel_videos_fixed(ch_id)
        
        for v in videos:
            v['channel_short_name'] = ch_name
            v['channel_category'] = 'family'
        
        all_fixed.extend(videos)
        print(f"  Got {len(videos)} videos, sample view: {videos[0]['viewCount'] if videos else 'N/A'}")
        
        # Checkpoint every 5 channels
        if (i + 1) % 5 == 0:
            save_csv(all_fixed)
            print(f"  [CHECKPOINT] Saved {len(all_fixed)} videos")
        
        time.sleep(DELAY)
    
    # Final save
    save_csv(all_fixed)
    
    print(f"\n{'='*60}")
    print(f"FIX COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos with fixed views: {len(all_fixed)}")
    
    # Quick stats
    views = [v['viewCount'] for v in all_fixed if v['viewCount'] > 0]
    print(f"Videos with views > 0: {len(views)} / {len(all_fixed)}")
    if views:
        print(f"Median views: {sorted(views)[len(views)//2]:,}")

def save_csv(all_videos):
    fieldnames = ['id', 'title', 'publishedAt', 'channelId', 'viewCount', 
                  'channel_short_name', 'channel_category']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in all_videos:
            writer.writerow({k: v.get(k, '') for k in fieldnames})

if __name__ == '__main__':
    main()
