#!/usr/bin/env python3
"""
Kidfluencer YouTube Data Collection Script
==========================================
Collects channel-level details and video-level metadata for a list of
kidfluencer/family vlog YouTube channels.

Usage:
    python3 collect_data.py                    # Run with default test channels
    python3 collect_data.py --input seeds.csv  # Run with a CSV of channel IDs
"""

import sys
import json
import csv
import os
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DELAY_BETWEEN_CALLS = 1.0  # seconds, to respect rate limits
MAX_VIDEOS_PER_CHANNEL = 100  # fetch up to this many videos per channel

# ── API Client ──────────────────────────────────────────────────────────────
client = ApiClient()


def fetch_channel_details(channel_id: str) -> Optional[Dict[str, Any]]:
    """Fetch channel-level metadata."""
    try:
        resp = client.call_api('Youtube/get_channel_details', query={
            'id': channel_id,
            'hl': 'en'
        })
        return resp if resp else None
    except Exception as e:
        print(f"  [ERROR] get_channel_details({channel_id}): {e}")
        return None


def fetch_channel_videos(channel_id: str, filter_type: str = "videos_latest",
                         max_videos: int = MAX_VIDEOS_PER_CHANNEL) -> List[Dict]:
    """Fetch video list with pagination, up to max_videos."""
    all_videos = []
    cursor = None
    page = 0

    while len(all_videos) < max_videos:
        page += 1
        try:
            params = {
                'id': channel_id,
                'filter': filter_type,
                'hl': 'en',
                'gl': 'US'
            }
            if cursor:
                params['cursor'] = cursor

            resp = client.call_api('Youtube/get_channel_videos', query=params)
            if not resp:
                break

            contents = resp.get('contents', [])
            if not contents:
                break

            all_videos.extend(contents)
            print(f"    Page {page}: got {len(contents)} items (total: {len(all_videos)})")

            cursor = resp.get('cursorNext', '')
            if not cursor:
                break

            time.sleep(DELAY_BETWEEN_CALLS)

        except Exception as e:
            print(f"  [ERROR] get_channel_videos page {page}: {e}")
            break

    return all_videos[:max_videos]


def parse_channel(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured fields from raw channel API response."""
    stats = raw.get('stats', {})
    links = raw.get('links', [])

    # Detect cross-platform presence (check both URLs and titles)
    link_parts = []
    for link in links:
        if isinstance(link, dict):
            link_parts.append(link.get('url', '') or '')
            link_parts.append(link.get('targetUrl', '') or '')
            link_parts.append(link.get('title', '') or '')
        elif isinstance(link, str):
            link_parts.append(link)
    link_text = ' '.join(link_parts).lower()

    has_instagram = 'instagram' in link_text
    has_tiktok = 'tiktok' in link_text
    has_twitter = 'twitter' in link_text or 'x.com' in link_text
    has_facebook = 'facebook' in link_text
    cross_platform_count = sum([has_instagram, has_tiktok, has_twitter, has_facebook])

    return {
        'channel_id': raw.get('channelId', ''),
        'title': raw.get('title', ''),
        'description': (raw.get('description', '') or '')[:500],
        'custom_url': raw.get('customUrl', ''),
        'handle': raw.get('handle', ''),
        'country': raw.get('country', ''),
        'joined_date': raw.get('joinedDate', ''),
        'subscribers': stats.get('subscribers', 0),
        'total_videos': stats.get('videos', 0),
        'total_views': stats.get('views', 0),
        'keywords': '|'.join(raw.get('keywords', [])),
        'has_instagram': has_instagram,
        'has_tiktok': has_tiktok,
        'has_twitter': has_twitter,
        'has_facebook': has_facebook,
        'cross_platform_count': cross_platform_count,
        'badges': '|'.join(str(b.get('type','')) if isinstance(b, dict) else str(b) for b in raw.get('badges', [])),
        'links_raw': json.dumps(links),
    }


def parse_video(raw_item: Dict[str, Any], channel_id: str) -> Optional[Dict[str, Any]]:
    """Extract structured fields from a single video item."""
    if raw_item.get('type') != 'video':
        return None

    video = raw_item.get('video', {})
    stats = video.get('stats', {})
    title = video.get('title', '')
    description = video.get('descriptionSnippet', '') or ''

    # ── Commercial indicators ──
    commercial_keywords = ['#ad', 'sponsor', 'sponsored', 'paid partnership',
                           'affiliate', 'discount code', 'use code', 'promo']
    title_lower = title.lower()
    desc_lower = description.lower()
    is_commercial = any(kw in title_lower or kw in desc_lower for kw in commercial_keywords)

    # ── Emotional manipulation indicators ──
    emotional_keywords = ['crying', 'cried', 'tears', 'punishment', 'punished',
                          'grounded', 'hospital', 'emergency', 'secret',
                          'prank', 'surprise', 'shocking', 'heartbreaking',
                          'emotional', 'broke down', 'meltdown', 'tantrum']
    has_emotional_title = any(kw in title_lower for kw in emotional_keywords)

    return {
        'channel_id': channel_id,
        'video_id': video.get('videoId', ''),
        'title': title,
        'published_text': video.get('publishedTimeText', ''),
        'length_seconds': video.get('lengthSeconds', 0),
        'views': stats.get('views', 0),
        'is_live': video.get('isLiveNow', False),
        'badges': '|'.join(str(b.get('type','')) if isinstance(b, dict) else str(b) for b in video.get('badges', [])),
        'description_snippet': description[:300],
        'is_commercial': is_commercial,
        'has_emotional_title': has_emotional_title,
    }


def save_csv(data: List[Dict], filepath: str):
    """Save list of dicts to CSV."""
    if not data:
        print(f"  No data to save for {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = data[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved {len(data)} rows → {filepath}")


def collect_one_channel(channel_id: str) -> tuple:
    """Collect all data for a single channel. Returns (channel_row, video_rows)."""
    print(f"\n{'='*60}")
    print(f"Collecting: {channel_id}")
    print(f"{'='*60}")

    # 1) Channel details
    print("  Fetching channel details...")
    raw_channel = fetch_channel_details(channel_id)
    if not raw_channel:
        print(f"  [SKIP] Could not fetch channel details for {channel_id}")
        return None, []
    channel_row = parse_channel(raw_channel)
    print(f"  → {channel_row['title']} | {channel_row['subscribers']} subs | {channel_row['total_videos']} videos")
    time.sleep(DELAY_BETWEEN_CALLS)

    # 2) Video list — MUST use resolved channelId, not URL
    resolved_id = raw_channel.get('channelId', channel_id)
    print(f"  Fetching video list (resolved ID: {resolved_id})...")
    raw_videos = fetch_channel_videos(resolved_id)
    video_rows = []
    for item in raw_videos:
        parsed = parse_video(item, resolved_id)
        if parsed:
            video_rows.append(parsed)
    print(f"  → Parsed {len(video_rows)} videos")

    # Quick stats
    if video_rows:
        commercial_count = sum(1 for v in video_rows if v['is_commercial'])
        emotional_count = sum(1 for v in video_rows if v['has_emotional_title'])
        avg_duration = sum(v['length_seconds'] for v in video_rows) / len(video_rows)
        avg_views = sum(v['views'] for v in video_rows) / len(video_rows)
        print(f"  → Avg duration: {avg_duration:.0f}s | Avg views: {avg_views:,.0f}")
        print(f"  → Commercial videos: {commercial_count}/{len(video_rows)} | Emotional titles: {emotional_count}/{len(video_rows)}")

    return channel_row, video_rows


def main():
    parser = argparse.ArgumentParser(description='Kidfluencer YouTube Data Collector')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to CSV with channel_id column')
    parser.add_argument('--max-videos', type=int, default=MAX_VIDEOS_PER_CHANNEL,
                        help='Max videos to fetch per channel')
    args = parser.parse_args()

    # ── Determine channel list ──
    if args.input and os.path.exists(args.input):
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            channel_ids = [row['channel_id'] for row in reader if row.get('channel_id')]
        print(f"Loaded {len(channel_ids)} channels from {args.input}")
    else:
        # Default test set: well-known kidfluencer / family vlog channels
        channel_ids = [
            "https://www.youtube.com/@KidsdianaShow",      # Kids Diana Show
            "https://www.youtube.com/@VladandNiki",         # Vlad and Niki
            "https://www.youtube.com/@LikeNastyaofficial",  # Like Nastya
            "https://www.youtube.com/@ACEFamily",           # ACE Family
            "https://www.youtube.com/@TheLabrantFam",       # The LaBrant Fam
        ]
        print(f"Using default test set: {len(channel_ids)} channels")

    # ── Collect ──
    all_channels = []
    all_videos = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, cid in enumerate(channel_ids, 1):
        print(f"\n[{i}/{len(channel_ids)}]")
        channel_row, video_rows = collect_one_channel(cid)
        if channel_row:
            all_channels.append(channel_row)
            all_videos.extend(video_rows)
        time.sleep(DELAY_BETWEEN_CALLS)

    # ── Save ──
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    save_csv(all_channels, os.path.join(OUTPUT_DIR, f'channels_{timestamp}.csv'))
    save_csv(all_videos, os.path.join(OUTPUT_DIR, f'videos_{timestamp}.csv'))

    # Also save raw JSON for debugging
    raw_path = os.path.join(OUTPUT_DIR, f'raw_{timestamp}.json')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, 'w') as f:
        json.dump({'channels': all_channels, 'videos': all_videos}, f, indent=2, default=str)
    print(f"  Saved raw JSON → {raw_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Channels collected: {len(all_channels)}")
    print(f"  Videos collected:   {len(all_videos)}")
    if all_videos:
        print(f"  Commercial videos:  {sum(1 for v in all_videos if v['is_commercial'])}")
        print(f"  Emotional titles:   {sum(1 for v in all_videos if v['has_emotional_title'])}")


if __name__ == '__main__':
    main()
