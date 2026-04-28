"""
TikTok Data Collection for Kidfluencer Study
=============================================
1. Map YouTube family channels to TikTok accounts
2. Collect user info (followers, video count)
3. Collect popular posts (views, likes, comments)
"""

import sys
import json
import os
import time

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

BASE_DIR = '/home/ubuntu/KidInfluencer'
TIKTOK_DIR = os.path.join(BASE_DIR, 'data', 'tiktok')
os.makedirs(TIKTOK_DIR, exist_ok=True)

client = ApiClient()

# ============================================================
# TikTok usernames to try for our YouTube family channels
# Many use the same or similar handles across platforms
# ============================================================

FAMILY_TIKTOK_MAP = {
    # YouTube short_name -> list of possible TikTok usernames to try
    'acefamily': ['theacefamily', 'acefamily'],
    'labrantfam': ['labrantfam', 'thelabrantfam'],
    'ryansworld': ['ryansworld', 'rfryan'],
    'bratayley': ['bratayley', 'annieleblanc'],
    'familyfunpack': ['familyfunpack'],
    'cocomelon': ['cocomelon', 'cocomelonofficial'],
    'dailybumps': ['dailybumps'],
    'piperrockelle': ['piperrockelle'],
    'jordanmatter': ['jordanmatter'],
    'rebeccazamolo': ['rebeccazamolo'],
    'brentrivera': ['brentrivera'],
    'piersonwodzynski': ['piaborr', 'pierson'],
    'vladandniki': ['vladandniki', 'vladandnikiy'],
    'ehbee': ['ehbeefamily'],
    'familyfizz': ['familyfizz'],
    'everleighrose': ['everleighrose'],
    'kkandbabyj': ['kkandbabyj'],
    'bonniehoellein': ['bonniehoellein'],
    'thesacconejolys': ['sacconejolys'],
    'theleray': ['theleroys'],
    'theweisslife': ['theweisslife'],
    'tannerites': ['tannerites'],
    'itsyeboi': ['itsyeboi'],
    'andrewdavila': ['andrewdavila'],
    'jakepaul': ['jakepaul'],
}

# Adult control channels
ADULT_TIKTOK_MAP = {
    'mrbeast': ['mrbeast'],
    'markwiens': ['markwiens'],
    'caseyneistat': ['caseyneistat'],
    'emmachamberlain': ['emmachamberlain'],
    'daviddobrik': ['daviddobrik'],
    'mkbhd': ['mkbhd'],
    'grahamstephan': ['grahamstephan'],
    'aliabdaal': ['aliabdaal'],
    'pewdiepie': ['pewdiepie'],
    'jamescharles': ['jamescharles'],
    'jeffreestar': ['jeffreestar'],
    'nikkietutorials': ['nikkietutorials'],
    'mattdavella': ['mattdavella'],
}


def get_user_info(unique_id):
    """Get TikTok user info by username."""
    try:
        response = client.call_api('Tiktok/get_user_info', query={'uniqueId': unique_id})
        return response
    except Exception as e:
        print(f"    Error: {e}")
        return None


def get_user_posts(sec_uid, count=35, cursor="0"):
    """Get user's popular posts."""
    try:
        response = client.call_api('Tiktok/get_user_popular_posts', query={
            'secUid': sec_uid,
            'count': str(count),
            'cursor': cursor,
        })
        return response
    except Exception as e:
        print(f"    Error getting posts: {e}")
        return None


def collect_channel(yt_name, tiktok_usernames, category):
    """Try to find and collect data for a channel on TikTok."""
    
    for username in tiktok_usernames:
        print(f"  Trying @{username}...", end=" ")
        
        result = get_user_info(username)
        time.sleep(0.5)  # Rate limiting
        
        if not result:
            print("API error")
            continue
        
        user_info = result.get('userInfo', {})
        user = user_info.get('user', {})
        stats = user_info.get('stats', {})
        
        if not user.get('id'):
            print("not found")
            continue
        
        sec_uid = user.get('secUid', '')
        followers = stats.get('followerCount', 0)
        videos = stats.get('videoCount', 0)
        hearts = stats.get('heartCount', 0)
        
        print(f"FOUND! {followers:,} followers, {videos} videos, {hearts:,} hearts")
        
        # Collect popular posts
        all_posts = []
        cursor = "0"
        page = 0
        
        while page < 10:  # Max 10 pages
            posts_result = get_user_posts(sec_uid, count=35, cursor=cursor)
            time.sleep(0.5)
            
            if not posts_result:
                break
            
            data = posts_result.get('data', {})
            if not data:
                # Try alternate structure
                items = posts_result.get('itemList', [])
                if items:
                    data = {'itemList': items, 'hasMore': False}
                else:
                    break
            
            items = data.get('itemList', [])
            if not items:
                break
            
            for item in items:
                post_stats = item.get('stats', {})
                post_data = {
                    'id': item.get('id', ''),
                    'desc': item.get('desc', ''),
                    'createTime': item.get('createTime', 0),
                    'playCount': post_stats.get('playCount', 0),
                    'diggCount': post_stats.get('diggCount', 0),  # likes
                    'commentCount': post_stats.get('commentCount', 0),
                    'shareCount': post_stats.get('shareCount', 0),
                    'duration': item.get('video', {}).get('duration', 0),
                }
                all_posts.append(post_data)
            
            has_more = data.get('hasMore', False)
            cursor = str(data.get('cursor', '0'))
            page += 1
            
            if not has_more:
                break
        
        print(f"    Collected {len(all_posts)} posts")
        
        # Save
        channel_data = {
            'youtube_channel': yt_name,
            'tiktok_username': username,
            'category': category,
            'sec_uid': sec_uid,
            'user_id': user.get('id', ''),
            'nickname': user.get('nickname', ''),
            'verified': user.get('verified', False),
            'signature': user.get('signature', ''),
            'followers': followers,
            'following': stats.get('followingCount', 0),
            'video_count': videos,
            'heart_count': hearts,
            'posts': all_posts,
        }
        
        output_file = os.path.join(TIKTOK_DIR, f'{yt_name}_tiktok.json')
        with open(output_file, 'w') as f:
            json.dump(channel_data, f, indent=2)
        
        return channel_data
    
    print(f"  ❌ No TikTok account found for {yt_name}")
    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("TIKTOK DATA COLLECTION FOR KIDFLUENCER STUDY")
    print("="*60)
    
    results = {'family': [], 'adult': []}
    
    # Family channels
    print("\n--- FAMILY CHANNELS ---")
    for yt_name, tiktok_names in FAMILY_TIKTOK_MAP.items():
        print(f"\n[{yt_name}]")
        data = collect_channel(yt_name, tiktok_names, 'family')
        if data:
            results['family'].append({
                'youtube': yt_name,
                'tiktok': data['tiktok_username'],
                'followers': data['followers'],
                'videos': data['video_count'],
                'hearts': data['heart_count'],
                'posts_collected': len(data['posts']),
            })
    
    # Adult control channels
    print("\n--- ADULT CONTROL CHANNELS ---")
    for yt_name, tiktok_names in ADULT_TIKTOK_MAP.items():
        print(f"\n[{yt_name}]")
        data = collect_channel(yt_name, tiktok_names, 'adult')
        if data:
            results['adult'].append({
                'youtube': yt_name,
                'tiktok': data['tiktok_username'],
                'followers': data['followers'],
                'videos': data['video_count'],
                'hearts': data['heart_count'],
                'posts_collected': len(data['posts']),
            })
    
    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"\nFamily channels found on TikTok: {len(results['family'])}/{len(FAMILY_TIKTOK_MAP)}")
    print(f"Adult channels found on TikTok: {len(results['adult'])}/{len(ADULT_TIKTOK_MAP)}")
    
    total_posts = sum(r['posts_collected'] for r in results['family'] + results['adult'])
    print(f"Total posts collected: {total_posts}")
    
    if results['family']:
        print("\n--- FAMILY CHANNELS ON TIKTOK ---")
        for r in sorted(results['family'], key=lambda x: -x['followers']):
            print(f"  @{r['tiktok']:25s}: {r['followers']:>12,} followers, {r['videos']:>5} videos, {r['posts_collected']} posts collected")
    
    if results['adult']:
        print("\n--- ADULT CHANNELS ON TIKTOK ---")
        for r in sorted(results['adult'], key=lambda x: -x['followers']):
            print(f"  @{r['tiktok']:25s}: {r['followers']:>12,} followers, {r['videos']:>5} videos, {r['posts_collected']} posts collected")
    
    # Save summary
    with open(os.path.join(TIKTOK_DIR, 'collection_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll data saved to {TIKTOK_DIR}")
