#!/usr/bin/env python3
"""
Collect Expanded Kidfluencer Channels
======================================
Adds ~50 new real-child kidfluencer channels to expand dataset from 48 to ~100.
Criteria: real children featured, English language, various sizes (large + medium).
Excludes: pure animation channels.
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
OUTPUT_CSV = os.path.join(DATA_DIR, 'expanded_channels_videos.csv')
DELAY = 1.5
MAX_VIDEOS_PER_CHANNEL = 500

# New channels NOT already in our 48-channel dataset
NEW_CHANNELS = [
    # === LARGE (1M+ subs) - Real children featured ===
    ("@NinjaKidzTV", "ninjakidztv"),           # ~40M, martial arts/action challenges
    ("@TheLaBrantFam", "labrantfam"),           # ~13M, family vlog
    ("@FGTeeV", "fgteev"),                     # ~22M, family gaming
    ("@SISvssBRO", "sisvsbro"),                # ~18M, sibling challenges
    ("@TrinityandBeyond", "trinityandbeyond"), # ~8M, kid challenges
    ("@SalishMatter", "salishmatter"),         # ~5M, gymnastics/challenges teen
    ("@JacyandKacy", "jacyandkacy"),           # ~5M, twin challenges
    ("@TicTacToy", "tictactoy"),              # ~5M, toy play/challenges
    ("@KidCityFamily", "kidcity"),             # ~5M, family adventures
    ("@FamousTubeFamily", "famoustubefamily"), # ~4M, family skits
    ("@TheSkory", "theskorys"),               # ~4M, family challenges
    ("@HayleyLeBlanc", "hayleyleblanc"),       # ~4M, teen vlogger (Bratayley)
    ("@Txunamy", "txunamy"),                   # ~4M, teen influencer
    ("@CocoQuinnB", "cocoquinn"),              # ~2M, teen dancer
    ("@TheHoldernessFam", "holderness"),       # ~1M, family comedy
    ("@OurFamilyNest", "ourfamilynest"),       # ~800K, family vlog
    ("@FamilyFunEveryDay", "familyfuneveryday"), # ~2M, family challenges
    ("@TheBeeFam", "thebeefamily"),            # ~5M, family challenges
    ("@KarinaKurzawa", "karinakurzawa"),       # ~11M, gaming/challenges teen
    ("@RonaldOMG", "ronaldomg"),               # ~10M, gaming/challenges kid
    
    # === MEDIUM (100K - 1M subs) - Important for reducing brand bias ===
    ("@TheGemSisters", "gemsisters"),          # ~400K, sister challenges
    ("@LifeWithBrothers", "lifewithbrothers"), # ~400K, brothers content
    ("@BrockandBoston", "brockandboston"),     # ~500K, brothers
    ("@YawiVlogs", "yawivlogs"),              # ~800K, family vlog
    ("@TheAdventurers", "theadventurers"),     # ~600K, family challenges
    ("@OhanaBoys", "ohanaboys"),              # ~500K, boys challenges
    ("@TheCrosbyFamily", "thecrosbys"),        # ~300K, family vlog
    ("@MeetTheMillers", "meetthemillers"),     # ~100K, family vlog
    ("@TheDashleys", "thedashleys"),           # ~150K, family vlog
    ("@ConfusedMillers", "confusedmillers"),   # ~200K, family vlog
    ("@CrazyGorilla", "crazygorilla"),         # ~200K, family challenges
    ("@FamilyFiveVlogs", "family5vlogs"),      # ~300K, family vlog
    ("@TheBramfam", "thebramfam"),            # ~2M, family vlog
    ("@itsRucka", "itsrucka"),                # ~200K, kid content
    ("@PantonsSquad", "pantonssquad"),         # ~6M, family challenges
    
    # === CHILD-AUDIENCE (to balance the moderation analysis) ===
    ("@KidsBabyBus", "babybus"),              # ~40M, but has live-action too
    ("@Blippi", "blippi"),                     # ~20M, real person for kids
    ("@DianaRoma", "dianaandroma"),            # ~120M, real kids for kid audience
    ("@JasonVlogs", "jasonvlogs"),            # ~5M, kid content for kids
    ("@SmileFamilyOfficial", "smilefamily"),   # ~10M, real kids for kid audience
    ("@KidsSongs", "kidssongs"),              # real kids performing
    ("@VladNikiEN", "vladnikien"),            # English version, kid audience
    
    # === Additional teen/young influencer channels ===
    ("@SawyerSharbino", "sawyersharbino"),    # ~1M, teen actor
    ("@PiperRockelleSquad", "piperrockellesquad"), # squad members
    ("@JordynJones", "jordynjones"),          # ~2M, teen dancer/influencer
    ("@AnnieRose", "annierose"),              # ~8M, Jules LeBlanc
    ("@RhythmicGymnast", "rhythmicgymnast"), # kid gymnast channels
]

def resolve_channel_handle(handle):
    """Resolve a YouTube handle to a channel ID."""
    try:
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
        'likeCount': 0,
        'commentCount': 0,
        'channel_short_name': channel_short_name,
        'channel_category': 'family',
    }

def save_csv(all_videos):
    """Save videos to CSV."""
    if not all_videos:
        return
    fieldnames = ['id', 'title', 'publishedAt', 'channelId', 'channelTitle', 
                  'viewCount', 'likeCount', 'commentCount', 'channel_short_name', 'channel_category']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in all_videos:
            writer.writerow(v)

def main():
    print(f"[{datetime.now()}] Starting expanded channel collection")
    print(f"Target: {len(NEW_CHANNELS)} new channels, up to {MAX_VIDEOS_PER_CHANNEL} videos each")
    print()
    
    all_videos = []
    channel_stats = []
    failed = []
    
    for i, (handle, short_name) in enumerate(NEW_CHANNELS):
        print(f"[{i+1}/{len(NEW_CHANNELS)}] Processing {handle} ({short_name})...")
        
        channel_id = resolve_channel_handle(handle)
        if not channel_id:
            print(f"  SKIP: Could not resolve channel ID for {handle}")
            failed.append(handle)
            continue
        
        time.sleep(DELAY)
        
        videos = fetch_channel_videos(channel_id)
        print(f"  Found {len(videos)} videos")
        
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
    print(f"Failed channels: {len(failed)}")
    if failed:
        print(f"  Failed: {failed}")
    print()
    
    # Print per-channel summary
    print(f"{'Channel':<25} {'Videos':>8}")
    print("-" * 35)
    for cs in channel_stats:
        print(f"{cs['short_name']:<25} {cs['videos_collected']:>8}")

if __name__ == '__main__':
    main()
