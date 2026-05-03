#!/usr/bin/env python3
"""
Kidfluencer Channel Discovery V2
==================================
Expanded search with more queries + related channel crawling.
"""

import sys
import json
import csv
import os
import time
from typing import Dict, List

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DELAY = 1.0

# Expanded search queries
SEARCH_QUERIES = [
    # Family vlogs
    "family vlog 2024",
    "family vlog daily routine",
    "family of 4 vlog",
    "family of 6 vlog",
    "big family vlog",
    "young family vlog",
    "new family vlog channel",
    "family vloggers",
    # Kid-focused
    "kid YouTuber channel",
    "kids YouTube channel",
    "children YouTube star",
    "child actor YouTube",
    "young YouTuber",
    "teen YouTuber family",
    # Specific content types
    "family challenge",
    "kids challenge video",
    "family prank video",
    "kids surprise video",
    "family road trip vlog",
    "homeschool family vlog",
    "adoption family vlog",
    "military family vlog",
    "interracial family vlog",
    # Mom/Dad vlogs with kids
    "mommy vlogger",
    "stay at home mom vlog",
    "dad vlog with kids",
    "parenting vlog",
    "mom of 3 vlog",
    "mom of 4 vlog",
    # Kid activities
    "kids toy review",
    "toy unboxing channel",
    "kids cooking channel",
    "kids science experiments",
    "baby shark dance",
    "nursery rhymes channel",
    "kids educational channel",
    # Family lifestyle
    "family lifestyle channel",
    "family travel vlog",
    "family house tour",
    "day in the life family",
    "morning routine family",
    "night routine kids",
    "back to school vlog family",
    # Regional
    "UK family vlog",
    "Australian family vlog",
    "Canadian family vlog",
    "Filipino family vlog",
    "Indian family vlog",
]

# Additional known channels to include
ADDITIONAL_KNOWN = [
    "https://www.youtube.com/@RyansWorld",
    "https://www.youtube.com/@CKNToys",
    "https://www.youtube.com/@FGTeeV",
    "https://www.youtube.com/@GamerGirl",
    "https://www.youtube.com/@EvanTubeHD",
    "https://www.youtube.com/@JillianTubeHD",
    "https://www.youtube.com/@ToyLabTV",
    "https://www.youtube.com/@BabyPlayful",
    "https://www.youtube.com/@CocomelonenEspanol",
    "https://www.youtube.com/@BLIPPIofficial",
    "https://www.youtube.com/@itsyeboi",
    "https://www.youtube.com/@SIS_vs_BRO",
    "https://www.youtube.com/@TwinsFromRussia",
    "https://www.youtube.com/@NikoandGabi",
    "https://www.youtube.com/@BrooklynAndBailey",
    "https://www.youtube.com/@FamilyFunPack",
    "https://www.youtube.com/@8Passengers",
    "https://www.youtube.com/@FantasticAdventures",
    "https://www.youtube.com/@MYHouseIsFULL",
    "https://www.youtube.com/@NotEnoughNelsons",
    "https://www.youtube.com/@FamilyFizz",
    "https://www.youtube.com/@Ohana",
    "https://www.youtube.com/@TheFisherFamily",
    "https://www.youtube.com/@JacyandKacy",
    "https://www.youtube.com/@TheCroutonCrackerjacks",
    "https://www.youtube.com/@KidsFunTV",
    "https://www.youtube.com/@ToyMonster",
    "https://www.youtube.com/@TheEngineeringFamily",
    "https://www.youtube.com/@TheSuperHeroKid",
]


def search_youtube(query: str, max_pages: int = 3) -> List[Dict]:
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
    # Load existing discovered channels to avoid duplicates
    existing_path = os.path.join(DATA_DIR, 'discovered_channels.csv')
    existing_ids = set()
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            for row in csv.DictReader(f):
                existing_ids.add(row['channel_id'])
    print(f"Already have {len(existing_ids)} channels from previous discovery")

    all_channels = {}

    # 1) Add additional known channels
    print("\n=== Adding additional known channels ===")
    for url in ADDITIONAL_KNOWN:
        if url not in existing_ids:
            all_channels[url] = {
                'channel_id': url,
                'title': '',
                'subscribers_text': '',
                'video_count_text': '',
                'source': 'known_list_v2',
            }
    print(f"  Added {len(all_channels)} new known channels")

    # 2) Search-based discovery
    print(f"\n=== Searching with {len(SEARCH_QUERIES)} queries ===")
    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"  [{i}/{len(SEARCH_QUERIES)}] Searching: '{query}'")
        results = search_youtube(query, max_pages=2)
        new_count = 0
        for ch in results:
            cid = ch['channel_id']
            if cid and cid not in all_channels and cid not in existing_ids:
                all_channels[cid] = ch
                new_count += 1
        print(f"    Found {len(results)} results, {new_count} new unique channels")
        time.sleep(DELAY)

    # 3) Save new channels
    output_path = os.path.join(DATA_DIR, 'discovered_channels_v2.csv')
    rows = list(all_channels.values())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channel_id', 'title', 'subscribers_text', 'video_count_text', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    # 4) Also create a combined file
    combined_path = os.path.join(DATA_DIR, 'all_discovered_channels_combined.csv')
    all_rows = rows.copy()
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
    with open(combined_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channel_id', 'title', 'subscribers_text', 'video_count_text', 'source'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n=== DISCOVERY V2 COMPLETE ===")
    print(f"  New channels found: {len(rows)}")
    print(f"  Combined total: {len(all_rows)}")
    print(f"  Saved new → {output_path}")
    print(f"  Saved combined → {combined_path}")


if __name__ == '__main__':
    main()
