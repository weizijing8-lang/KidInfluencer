"""
TikTok Data Collection V2 - Using Search API
=============================================
Since get_user_popular_posts returns empty, we use search API
to find videos by each channel, then filter by author.
Also combines with user_info data already collected.
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

# Channels with significant TikTok presence (>100K followers or >100 videos)
ACTIVE_CHANNELS = {
    'family': {
        'brentrivera': {'tiktok': 'brentrivera', 'followers': 50_600_000},
        'piperrockelle': {'tiktok': 'piperrockelle', 'followers': 20_100_000},
        'jakepaul': {'tiktok': 'jakepaul', 'followers': 19_500_000},
        'rebeccazamolo': {'tiktok': 'rebeccazamolo', 'followers': 18_500_000},
        'piersonwodzynski': {'tiktok': 'pierson', 'followers': 17_300_000},
        'ehbee': {'tiktok': 'ehbeefamily', 'followers': 12_200_000},
        'cocomelon': {'tiktok': 'cocomelon', 'followers': 8_500_000},
        'theacefamily': {'tiktok': 'theacefamily', 'followers': 6_600_000},
        'vladandniki': {'tiktok': 'vladandniki', 'followers': 0},  # will search
        'theweisslife': {'tiktok': 'theweisslife', 'followers': 1_500_000},
        'familyfizz': {'tiktok': 'familyfizz', 'followers': 805_700},
        'jordanmatter': {'tiktok': 'jordanmatter', 'followers': 0},  # will search
    },
    'adult': {
        'mrbeast': {'tiktok': 'mrbeast', 'followers': 126_400_000},
        'jamescharles': {'tiktok': 'jamescharles', 'followers': 40_800_000},
        'daviddobrik': {'tiktok': 'daviddobrik', 'followers': 25_000_000},
        'nikkietutorials': {'tiktok': 'nikkietutorials', 'followers': 9_100_000},
        'jeffreestar': {'tiktok': 'jeffreestar', 'followers': 8_300_000},
        'mkbhd': {'tiktok': 'mkbhd', 'followers': 2_300_000},
        'markwiens': {'tiktok': 'markwiens', 'followers': 2_100_000},
        'pewdiepie': {'tiktok': 'pewdiepie', 'followers': 1_000_000},
        'grahamstephan': {'tiktok': 'grahamstephan', 'followers': 937_500},
        'caseyneistat': {'tiktok': 'caseyneistat', 'followers': 861_100},
    }
}


def search_channel_videos(channel_name, tiktok_username, max_pages=5):
    """Search for a channel's videos using the search API."""
    all_videos = []
    cursor = 0
    search_id = None
    
    for page in range(max_pages):
        query = {'keyword': tiktok_username}
        if cursor > 0:
            query['cursor'] = cursor
        if search_id:
            query['search_id'] = search_id
        
        result = client.call_api('Tiktok/search_tiktok_video_general', query=query)
        time.sleep(1)  # Rate limit
        
        if not result:
            break
        
        items = result.get('item_list', [])
        if not items:
            break
        
        for item in items:
            author = item.get('author', {})
            author_uid = author.get('unique_id', '') or author.get('uniqueId', '')
            author_nick = author.get('nickname', '')
            
            stats = item.get('stats', {})
            video_data = {
                'id': item.get('id', ''),
                'desc': item.get('desc', ''),
                'createTime': item.get('createTime', 0),
                'author_uid': author_uid,
                'author_nickname': author_nick,
                'playCount': stats.get('playCount', 0),
                'diggCount': stats.get('diggCount', 0),
                'commentCount': stats.get('commentCount', 0),
                'shareCount': stats.get('shareCount', 0),
                'collectCount': stats.get('collectCount', 0),
                'duration': item.get('video', {}).get('duration', 0),
                'is_ad': item.get('isAd', False),
            }
            all_videos.append(video_data)
        
        has_more = result.get('has_more', 0)
        cursor = result.get('cursor', 0)
        
        # Try to get search_id for pagination
        log_pb = result.get('log_pb', {})
        if log_pb:
            search_id = log_pb.get('impr_id', search_id)
        
        print(f"    Page {page+1}: {len(items)} results (total: {len(all_videos)})")
        
        if not has_more:
            break
    
    # Filter to only videos by this author
    own_videos = [v for v in all_videos 
                  if tiktok_username.lower() in (v['author_uid'] or '').lower()
                  or tiktok_username.lower() in (v['author_nickname'] or '').lower()]
    
    other_videos = [v for v in all_videos if v not in own_videos]
    
    return own_videos, other_videos, all_videos


def main():
    print("=" * 60)
    print("TIKTOK VIDEO COLLECTION V2 (Search API)")
    print("=" * 60)
    
    results = []
    
    for category, channels in ACTIVE_CHANNELS.items():
        print(f"\n--- {category.upper()} CHANNELS ---")
        
        for yt_name, info in channels.items():
            tiktok_name = info['tiktok']
            print(f"\n[{yt_name}] Searching @{tiktok_name}...")
            
            own, other, all_vids = search_channel_videos(yt_name, tiktok_name)
            
            print(f"    Own videos: {len(own)}, Other mentions: {len(other)}, Total: {len(all_vids)}")
            
            if own:
                plays = [v['playCount'] for v in own]
                likes = [v['diggCount'] for v in own]
                comments = [v['commentCount'] for v in own]
                durations = [v['duration'] for v in own if v['duration'] > 0]
                
                print(f"    Avg plays: {sum(plays)/len(plays):,.0f}")
                print(f"    Avg likes: {sum(likes)/len(likes):,.0f}")
                print(f"    Avg comments: {sum(comments)/len(comments):,.0f}")
                if durations:
                    print(f"    Avg duration: {sum(durations)/len(durations):.1f}s")
            
            # Save
            channel_data = {
                'youtube_channel': yt_name,
                'tiktok_username': tiktok_name,
                'category': category,
                'followers': info['followers'],
                'own_videos': own,
                'related_videos': other,
                'total_search_results': len(all_vids),
            }
            
            output_file = os.path.join(TIKTOK_DIR, f'{yt_name}_tiktok_v2.json')
            with open(output_file, 'w') as f:
                json.dump(channel_data, f, indent=2)
            
            results.append({
                'youtube': yt_name,
                'tiktok': tiktok_name,
                'category': category,
                'followers': info['followers'],
                'own_videos_found': len(own),
                'avg_plays': sum(v['playCount'] for v in own) / max(len(own), 1),
                'avg_likes': sum(v['diggCount'] for v in own) / max(len(own), 1),
                'avg_comments': sum(v['commentCount'] for v in own) / max(len(own), 1),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(TIKTOK_DIR, 'tiktok_summary.csv'), index=False)
    
    for cat in ['family', 'adult']:
        sub = df[df['category'] == cat]
        print(f"\n{cat.upper()} ({len(sub)} channels):")
        print(f"  Avg followers: {sub['followers'].mean():,.0f}")
        print(f"  Avg own videos found: {sub['own_videos_found'].mean():.1f}")
        print(f"  Avg plays per video: {sub['avg_plays'].mean():,.0f}")
        print(f"  Avg likes per video: {sub['avg_likes'].mean():,.0f}")
    
    print(f"\nAll data saved to {TIKTOK_DIR}")


if __name__ == '__main__':
    main()
