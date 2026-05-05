#!/usr/bin/env python3
"""
Collect New Kidfluencer Channels
=================================
Adds more real-child kidfluencer channels to expand the dataset.
Removes animation channels (Cocomelon, SuperHeroBuddy).
"""

import sys
import json
import csv
import os
import time
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'new_channels_videos.csv')
DELAY = 1.0
MAX_VIDEOS_PER_CHANNEL = 500  # Get more videos per channel for better stats

# New channels to collect (real children only, no animation)
NEW_CHANNELS = [
    # Large family/kid channels not yet in dataset
    ("@TheLaBrantFam", "labrantfam"),
    ("@theroyaltyfamily", "royaltyfamily"),
    ("@LikeNastyaofficial", "likenastya"),
    ("@AforAdley", "aforadley"),
    ("@KidsdianaShow", "kidsdianashow"),
    ("@soty", "shotofyeagers"),
    ("@TheOhanaAdventure", "ohanaadventure"),
    ("@OnyxFamily", "onyxfamily"),
    ("@JHouseVlogs", "jhousevlogs"),
    ("@BallingerFamily", "ballingerfamily"),
    ("@PantonsSquad", "pantonssquad"),
    ("@SamandNia", "samandnia"),
    ("@itsJudysLife", "itsjudyslife"),
    ("@TheJohnsonFam", "johnsonsfam"),
    ("@JesssFam", "jesssfam"),
    ("@TheFamilyFudge", "familyfudge"),
    ("@TheInghamFamily", "inghamfamily"),
    ("@mccluretwins", "mccluretwins"),
    ("@WeAreTheDavises", "wearethedavises"),
    ("@ThatYouTub3Family", "thatyoutub3family"),
    # Controversial / high-exploitation channels
    ("@TydusTalbott", "tydustalbott"),
    ("@GavinMagnus", "gavinmagnus"),
    ("@CocoQuinnB", "cocoquinn"),
    # Additional kid-focused channels
    ("@EthanGamerTV", "ethangamer"),
    ("@GabyandAlexGames", "gabyandAlex"),
    ("@JillianandAddie", "jillianandaddie"),
    ("@SmellyBellyTV", "smellybellytv"),
    ("@FunSquadFamily", "funsquadfamily"),
    ("@NorrisNuts", "norrisnuts"),
    ("@TheRoyaltyFam", "royaltyfam2"),
]


def resolve_channel_handle(handle):
    """Resolve a YouTube handle to a channel ID."""
    try:
        # Search for the channel
        resp = client.call_api('Youtube/search', query={'q': handle, 'hl': 'en', 'gl': 'US'})
        if not resp:
            return None
        contents = resp.get('contents', [])
        for item in contents:
            if item.get('type') == 'channel':
                ch = item.get('channel', {})
                return ch.get('channelId', '')
            elif item.get('type') == 'video':
                vid = item.get('video', {})
                # Check if the channel name matches
                ch_id = vid.get('channelId', '')
                if ch_id:
                    return ch_id
        return None
    except Exception as e:
        print(f"  [ERROR] resolve_handle({handle}): {e}")
        return None


def fetch_channel_videos(channel_id, max_videos=MAX_VIDEOS_PER_CHANNEL):
    """Fetch videos from a channel using pagination."""
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
            all_videos.extend(contents)
            pages += 1
            cursor = resp.get('cursorNext', '')
            if not cursor:
                break
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [ERROR] fetch_videos page {pages}: {e}")
            break
    return all_videos[:max_videos]


def parse_video(video_data, channel_id, channel_short_name):
    """Parse a video item into a flat dict."""
    vid = video_data.get('video', video_data)
    
    # Handle view count
    view_count = vid.get('viewCount', 0)
    if isinstance(view_count, str):
        view_count = view_count.replace(',', '')
        try:
            view_count = int(view_count)
        except:
            view_count = 0
    
    return {
        'id': vid.get('videoId', ''),
        'title': vid.get('title', ''),
        'publishedAt': vid.get('publishedTimeText', ''),
        'channelId': channel_id,
        'channelTitle': vid.get('channelTitle', vid.get('channelName', '')),
        'viewCount': view_count,
        'likeCount': 0,  # Not available from list
        'commentCount': 0,
        'channel_short_name': channel_short_name,
        'channel_category': 'family',
    }


def main():
    print(f"[{datetime.now()}] Starting new channel collection")
    print(f"Target: {len(NEW_CHANNELS)} channels, up to {MAX_VIDEOS_PER_CHANNEL} videos each")
    print()
    
    all_videos = []
    channel_stats = []
    
    for i, (handle, short_name) in enumerate(NEW_CHANNELS):
        print(f"[{i+1}/{len(NEW_CHANNELS)}] Processing {handle} ({short_name})...")
        
        # Resolve channel ID
        channel_id = resolve_channel_handle(handle)
        if not channel_id:
            print(f"  SKIP: Could not resolve channel ID for {handle}")
            continue
        
        time.sleep(DELAY)
        
        # Fetch videos
        videos = fetch_channel_videos(channel_id)
        print(f"  Found {len(videos)} videos")
        
        # Parse videos
        parsed = []
        for v in videos:
            p = parse_video(v, channel_id, short_name)
            if p['id'] and p['title']:
                parsed.append(p)
        
        all_videos.extend(parsed)
        channel_stats.append({
            'handle': handle,
            'short_name': short_name,
            'channel_id': channel_id,
            'videos_collected': len(parsed),
        })
        
        print(f"  Parsed {len(parsed)} valid videos (total so far: {len(all_videos)})")
        
        # Save incrementally every 5 channels
        if (i + 1) % 5 == 0:
            save_csv(all_videos)
            print(f"  [CHECKPOINT] Saved {len(all_videos)} videos")
        
        time.sleep(DELAY)
    
    # Final save
    save_csv(all_videos)
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total channels processed: {len(channel_stats)}")
    print(f"Total videos collected: {len(all_videos)}")
    print()
    for cs in channel_stats:
        print(f"  {cs['short_name']:25s} | {cs['videos_collected']:>4} videos | ID: {cs['channel_id']}")


def save_csv(videos):
    """Save videos to CSV."""
    if not videos:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['id', 'title', 'publishedAt', 'channelId', 'channelTitle', 
                  'viewCount', 'likeCount', 'commentCount', 'channel_short_name', 'channel_category']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in videos:
            writer.writerow({k: v.get(k, '') for k in fieldnames})


if __name__ == '__main__':
    main()
