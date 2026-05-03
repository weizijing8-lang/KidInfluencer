#!/usr/bin/env python3
"""
Kidfluencer Channel Discovery Script
=====================================
Uses YouTube Search API to find kidfluencer/family vlog channels
via multiple keyword queries, deduplicates, and outputs a seed list.
"""

import sys
import json
import csv
import os
import time
from typing import Dict, Any, List, Optional

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DELAY = 1.0

# Search queries designed to surface kidfluencer / family vlog channels
SEARCH_QUERIES = [
    "family vlog",
    "family vlog channel",
    "kid YouTuber",
    "kidfluencer",
    "family daily vlog",
    "mom vlog kids",
    "day in our life family",
    "kids routine morning",
    "family challenge video",
    "our family channel",
    "family fun",
    "kids play pretend",
    "family adventure vlog",
    "toddler vlog",
    "siblings fun play",
    "family prank kids",
    "unboxing toys kids",
    "kid influencer YouTube",
    "family of five vlog",
    "mommy vlog",
]

# Known kidfluencer channels to include directly (channel IDs or URLs)
KNOWN_CHANNELS = [
    # Top kidfluencers
    "https://www.youtube.com/@KidsdianaShow",
    "https://www.youtube.com/@VladandNiki",
    "https://www.youtube.com/@LikeNastyaofficial",
    "https://www.youtube.com/@RyansPlanet",        # Ryan's World / Ryan's Planet
    "https://www.youtube.com/@Toys4KidsTV",
    # Family vlogs
    "https://www.youtube.com/@TheLaBrantFam",
    "https://www.youtube.com/@SACCONEJOLYs",
    "https://www.youtube.com/@FamilyFunPack",
    "https://www.youtube.com/@itsJudysLife",
    "https://www.youtube.com/@EhBeeFamily",
    "https://www.youtube.com/@TheBramfam",
    "https://www.youtube.com/@SmellyBellyTV",
    "https://www.youtube.com/@DailyBumps",
    "https://www.youtube.com/@OKbaby",
    "https://www.youtube.com/@TheOhSoFamousFamily",
    # Known exploitation / controversy cases (ground truth)
    "https://www.youtube.com/@PiperRockelle",
    "https://www.youtube.com/@JordanMatter",
    # Mid-tier family channels
    "https://www.youtube.com/@TheBucketListFamily",
    "https://www.youtube.com/@BraveTVFamily",
    "https://www.youtube.com/@TiffyQuake",
    "https://www.youtube.com/@JesssFam",
    "https://www.youtube.com/@TheStauffersOfficial",
]


def search_youtube(query: str, max_pages: int = 2) -> List[Dict]:
    """Search YouTube and return channel info from results."""
    channels_found = []
    cursor = None

    for page in range(max_pages):
        params = {'q': query, 'hl': 'en', 'gl': 'US'}
        if cursor:
            params['cursor'] = cursor

        try:
            resp = client.call_api('Youtube/search', query=params)
            if not resp:
                break

            contents = resp.get('contents', [])
            for item in contents:
                item_type = item.get('type', '')

                if item_type == 'channel':
                    ch = item.get('channel', {})
                    channels_found.append({
                        'channel_id': ch.get('channelId', ''),
                        'title': ch.get('title', ''),
                        'subscribers_text': ch.get('subscriberCountText', ''),
                        'video_count_text': ch.get('videoCountText', ''),
                        'source': f'search:{query}',
                    })

                elif item_type == 'video':
                    vid = item.get('video', {})
                    ch_id = vid.get('channelId', '')
                    ch_title = vid.get('channelTitle', '')
                    if ch_id:
                        channels_found.append({
                            'channel_id': ch_id,
                            'title': ch_title,
                            'subscribers_text': '',
                            'video_count_text': '',
                            'source': f'search:{query}',
                        })

            cursor = resp.get('cursorNext', '')
            if not cursor:
                break
            time.sleep(DELAY)

        except Exception as e:
            print(f"  [ERROR] search '{query}' page {page}: {e}")
            break

    return channels_found


def main():
    all_channels = {}  # channel_id -> info dict (dedup)

    # 1) Add known channels
    print("=== Adding known channels ===")
    for url in KNOWN_CHANNELS:
        # Use URL as temp key; will be resolved later during data collection
        all_channels[url] = {
            'channel_id': url,
            'title': '',
            'subscribers_text': '',
            'video_count_text': '',
            'source': 'known_list',
        }
    print(f"  Added {len(KNOWN_CHANNELS)} known channels")

    # 2) Search-based discovery
    print(f"\n=== Searching with {len(SEARCH_QUERIES)} queries ===")
    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"  [{i}/{len(SEARCH_QUERIES)}] Searching: '{query}'")
        results = search_youtube(query, max_pages=2)
        new_count = 0
        for ch in results:
            cid = ch['channel_id']
            if cid and cid not in all_channels:
                all_channels[cid] = ch
                new_count += 1
        print(f"    Found {len(results)} results, {new_count} new unique channels")
        time.sleep(DELAY)

    # 3) Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'discovered_channels.csv')

    rows = list(all_channels.values())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channel_id', 'title', 'subscribers_text', 'video_count_text', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== DISCOVERY COMPLETE ===")
    print(f"  Total unique channels: {len(rows)}")
    print(f"  From known list: {sum(1 for r in rows if r['source'] == 'known_list')}")
    print(f"  From search: {sum(1 for r in rows if r['source'].startswith('search:'))}")
    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    main()
