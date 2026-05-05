#!/usr/bin/env python3
"""
Fix viewCount for channels that have 0 views due to wrong field parsing.
Re-fetches videos from the API and extracts stats.views correctly.
"""
import sys
import os
import csv
import time
import json
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

import pandas as pd

client = ApiClient()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DELAY = 1.0

# Channel IDs we need to fix (resolved from the original collection)
# We'll re-resolve them from the data we have
CHANNELS_TO_FIX = [
    'aforadley', 'ballingerfamily', 'ethangamer', 'familyfudge',
    'funsquadfamily', 'gavinmagnus', 'inghamfamily', 'itsjudyslife',
    'jesssfam', 'jhousevlogs', 'jillianandaddie', 'johnsonsfam',
    'kidsdianashow', 'likenastya', 'mccluretwins', 'norrisnuts',
    'ohanaadventure', 'onyxfamily', 'royaltyfamily', 'samandnia',
    'shotofyeagers', 'smellybellytv', 'thatyoutub3family',
    'tydustalbott', 'wearethedavises'
]


def main():
    print(f"[{datetime.now()}] Fixing viewCounts for {len(CHANNELS_TO_FIX)} channels")
    
    # Load the combined dataset
    df = pd.read_csv(os.path.join(DATA_DIR, 'combined_family_videos.csv'))
    print(f"Loaded {len(df)} videos total")
    
    # Get channel IDs for the broken channels
    broken = df[df['channel_short_name'].isin(CHANNELS_TO_FIX)]
    print(f"Videos to fix: {len(broken)}")
    
    # Get unique channelIds
    channel_map = broken.groupby('channel_short_name')['channelId'].first().to_dict()
    print(f"Channels to process: {len(channel_map)}")
    
    # For each channel, re-fetch all videos and build a videoId -> views mapping
    all_views = {}  # videoId -> viewCount
    
    for i, (short_name, channel_id) in enumerate(channel_map.items()):
        print(f"  [{i+1}/{len(channel_map)}] {short_name} ({channel_id})...", end=' ', flush=True)
        
        cursor = None
        channel_views = {}
        pages = 0
        
        while True:
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
                    stats = vid.get('stats', {})
                    views = stats.get('views', 0)
                    if video_id:
                        channel_views[video_id] = views
                
                pages += 1
                cursor = resp.get('cursorNext', '')
                if not cursor:
                    break
                time.sleep(DELAY)
                
            except Exception as e:
                print(f"ERROR: {e}")
                break
        
        all_views.update(channel_views)
        print(f"{len(channel_views)} videos, {pages} pages")
    
    print(f"\nTotal video views collected: {len(all_views)}")
    
    # Update the dataframe
    fixed_count = 0
    for idx, row in df.iterrows():
        if row['channel_short_name'] in CHANNELS_TO_FIX:
            vid_id = row['id']
            if vid_id in all_views:
                df.at[idx, 'viewCount'] = all_views[vid_id]
                fixed_count += 1
    
    print(f"Fixed {fixed_count} viewCounts")
    
    # Save
    output_path = os.path.join(DATA_DIR, 'combined_family_videos.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Verify
    df2 = pd.read_csv(output_path)
    df2['viewCount'] = pd.to_numeric(df2['viewCount'], errors='coerce')
    still_zero = df2[df2['channel_short_name'].isin(CHANNELS_TO_FIX)]
    print(f"\nVerification: {(still_zero['viewCount']>0).sum()}/{len(still_zero)} now have views > 0")
    print(f"Still zero: {(still_zero['viewCount']==0).sum()}")


if __name__ == '__main__':
    main()
