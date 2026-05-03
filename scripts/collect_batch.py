#!/usr/bin/env python3
"""
Batch Data Collection for All Discovered Channels
==================================================
Reads discovered_channels.csv and collects channel details + video metadata.
Saves incrementally to avoid data loss on interruption.
"""

import sys
import json
import csv
import os
import time
from datetime import datetime

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DELAY = 1.0
MAX_VIDEOS = 50  # 50 per channel to keep total manageable


def fetch_channel_details(channel_id):
    try:
        resp = client.call_api('Youtube/get_channel_details', query={'id': channel_id, 'hl': 'en'})
        return resp if resp else None
    except Exception as e:
        print(f"  [ERROR] channel_details({channel_id}): {e}")
        return None


def fetch_channel_videos(channel_id, max_videos=MAX_VIDEOS):
    all_videos = []
    cursor = None
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
            cursor = resp.get('cursorNext', '')
            if not cursor:
                break
            time.sleep(DELAY)
        except Exception as e:
            print(f"  [ERROR] channel_videos: {e}")
            break
    return all_videos[:max_videos]


def parse_channel(raw):
    stats = raw.get('stats', {})
    links = raw.get('links', [])
    link_parts = []
    for link in links:
        if isinstance(link, dict):
            link_parts.extend([link.get('url', '') or '', link.get('targetUrl', '') or '', link.get('title', '') or ''])
        elif isinstance(link, str):
            link_parts.append(link)
    lt = ' '.join(link_parts).lower()
    has_ig = 'instagram' in lt
    has_tt = 'tiktok' in lt
    has_tw = 'twitter' in lt or 'x.com' in lt
    has_fb = 'facebook' in lt
    return {
        'channel_id': raw.get('channelId', ''),
        'title': raw.get('title', ''),
        'description': (raw.get('description', '') or '')[:500],
        'custom_url': raw.get('customUrl', ''),
        'handle': raw.get('handle', ''),
        'country': raw.get('country', ''),
        'joined_date': raw.get('joinedDate', ''),
        'subscribers': stats.get('subscribers') or 0,
        'total_videos': stats.get('videos') or 0,
        'total_views': stats.get('views') or 0,
        'keywords': '|'.join(raw.get('keywords', [])),
        'has_instagram': has_ig,
        'has_tiktok': has_tt,
        'has_twitter': has_tw,
        'has_facebook': has_fb,
        'cross_platform_count': sum([has_ig, has_tt, has_tw, has_fb]),
        'badges': '|'.join(str(b.get('type', '')) if isinstance(b, dict) else str(b) for b in raw.get('badges', [])),
    }


def parse_video(raw_item, channel_id):
    if raw_item.get('type') != 'video':
        return None
    video = raw_item.get('video', {})
    stats = video.get('stats', {})
    title = video.get('title', '')
    desc = video.get('descriptionSnippet', '') or ''
    tl = title.lower()
    dl = desc.lower()
    commercial_kw = ['#ad', 'sponsor', 'sponsored', 'paid partnership', 'affiliate', 'discount code', 'use code', 'promo', 'brand deal', 'collab']
    emotional_kw = ['crying', 'cried', 'tears', 'punishment', 'punished', 'grounded', 'hospital', 'emergency', 'secret', 'prank', 'surprise', 'shocking', 'heartbreaking', 'emotional', 'broke down', 'meltdown', 'tantrum', 'screaming', 'angry', 'fight', 'kicked out', 'ran away', 'called the cops', 'destroyed']
    return {
        'channel_id': channel_id,
        'video_id': video.get('videoId', ''),
        'title': title,
        'published_text': video.get('publishedTimeText', ''),
        'length_seconds': video.get('lengthSeconds', 0),
        'views': stats.get('views', 0),
        'is_live': video.get('isLiveNow', False),
        'description_snippet': desc[:300],
        'is_commercial': any(kw in tl or kw in dl for kw in commercial_kw),
        'has_emotional_title': any(kw in tl for kw in emotional_kw),
    }


def main():
    # Load discovered channels
    input_path = os.path.join(DATA_DIR, 'discovered_channels.csv')
    with open(input_path, 'r') as f:
        seeds = list(csv.DictReader(f))
    print(f"Loaded {len(seeds)} seed channels")

    all_channels = []
    all_videos = []
    failed = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, seed in enumerate(seeds, 1):
        cid = seed['channel_id']
        print(f"\n[{i}/{len(seeds)}] {seed.get('title', '')} ({cid})")

        # Fetch channel details
        raw = fetch_channel_details(cid)
        if not raw or not raw.get('channelId'):
            print(f"  [SKIP] No data returned")
            failed.append(cid)
            time.sleep(DELAY)
            continue

        ch = parse_channel(raw)
        resolved_id = raw.get('channelId', cid)
        subs = ch['subscribers'] or 0
        print(f"  → {ch['title']} | {subs:,} subs | {ch['total_videos']} vids")
        all_channels.append(ch)
        time.sleep(DELAY)

        # Fetch videos
        raw_vids = fetch_channel_videos(resolved_id)
        vid_count = 0
        for item in raw_vids:
            v = parse_video(item, resolved_id)
            if v:
                all_videos.append(v)
                vid_count += 1
        print(f"  → {vid_count} videos collected")
        time.sleep(DELAY)

    # Final save
    ch_path = os.path.join(DATA_DIR, f'all_channels_{timestamp}.csv')
    vid_path = os.path.join(DATA_DIR, f'all_videos_{timestamp}.csv')

    if all_channels:
        with open(ch_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=all_channels[0].keys())
            w.writeheader()
            w.writerows(all_channels)
        print(f"\nSaved {len(all_channels)} channels → {ch_path}")

    if all_videos:
        with open(vid_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=all_videos[0].keys())
            w.writeheader()
            w.writerows(all_videos)
        print(f"Saved {len(all_videos)} videos → {vid_path}")

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"  Channels: {len(all_channels)}")
    print(f"  Videos:   {len(all_videos)}")
    print(f"  Failed:   {len(failed)}")
    if all_videos:
        print(f"  Commercial: {sum(1 for v in all_videos if v['is_commercial'])}")
        print(f"  Emotional:  {sum(1 for v in all_videos if v['has_emotional_title'])}")
    if failed:
        print(f"  Failed channels: {failed[:10]}...")


if __name__ == '__main__':
    main()
