#!/usr/bin/env python3
"""
Test YouTube API - Round 2 with corrected channel URLs and more channels.
Also test search functionality to find kidfluencer channels.
"""

import sys
import json

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()

# Search for kidfluencer / family vlog channels
print("="*60)
print("PART 1: Search YouTube for kidfluencer content")
print("="*60)

search_queries = [
    "family vlog kids",
    "kidfluencer",
    "kids unboxing toys",
]

search_results = {}
for query in search_queries:
    print(f"\nSearching: '{query}'")
    try:
        result = client.call_api('Youtube/search', query={
            'q': query,
            'hl': 'en',
            'gl': 'US'
        })
        if result:
            contents = result.get('contents', [])
            channels_found = []
            for item in contents:
                if item.get('type') == 'video':
                    video = item.get('video', {})
                    channel_title = video.get('channelTitle', 'N/A')
                    channel_id = video.get('channelId', 'N/A')
                    title = video.get('title', 'N/A')
                    views = video.get('viewCountText', 'N/A')
                    print(f"  Video: {title[:50]}... | Channel: {channel_title} | Views: {views}")
                    channels_found.append({
                        'channel': channel_title,
                        'channel_id': channel_id,
                        'video_title': title,
                        'views': views
                    })
                elif item.get('type') == 'channel':
                    channel = item.get('channel', {})
                    print(f"  Channel: {channel.get('title', 'N/A')} | Subs: {channel.get('subscriberCountText', 'N/A')}")
            search_results[query] = channels_found
    except Exception as e:
        print(f"  Error: {e}")

# Now try corrected channel URLs
print(f"\n\n{'='*60}")
print("PART 2: Fetch specific channels with corrected URLs")
print("="*60)

channels_v2 = [
    "https://www.youtube.com/@RyansWorld",           # Ryan Kaji - biggest kidfluencer
    "https://www.youtube.com/@TheLaBrantFam",        # LaBrant Family
    "https://www.youtube.com/@8passengers",          # 8 Passengers (Ruby Franke - convicted)
    "https://www.youtube.com/@ColeAndSav",           # Cole and Sav LaBrant
    "https://www.youtube.com/@DaddyOFive",           # DaddyOFive (notorious case)
    "https://www.youtube.com/@ItsJoJoSiwa",          # JoJo Siwa
    "https://www.youtube.com/@EvanTubeHD",           # EvanTubeHD
    "https://www.youtube.com/@LikNastya",            # Like Nastya
]

channel_data = {}
for url in channels_v2:
    print(f"\nFetching: {url}")
    try:
        result = client.call_api('Youtube/get_channel_details', query={
            'id': url,
            'hl': 'en'
        })
        if result and result.get('title'):
            name = result.get('title', 'Unknown')
            stats = result.get('stats', {})
            channel_id = result.get('channelId', '')
            
            info = {
                'channel_id': channel_id,
                'url': url,
                'subscribers': stats.get('subscribersText', 'N/A'),
                'total_videos': stats.get('videos', 0),
                'total_views': stats.get('views', 0),
                'country': result.get('country', 'N/A'),
                'joined': result.get('joinedDate', 'N/A'),
                'description': result.get('description', 'N/A')[:300],
            }
            channel_data[name] = info
            print(f"  ✓ {name} | {info['subscribers']} | {info['total_videos']} videos | Views: {info['total_views']:,}")
            
            # Get 5 latest videos with full metadata
            if channel_id:
                vids = client.call_api('Youtube/get_channel_videos', query={
                    'id': channel_id,
                    'filter': 'videos_latest',
                    'hl': 'en',
                    'gl': 'US'
                })
                if vids:
                    video_list = []
                    for item in vids.get('contents', [])[:3]:
                        if item.get('type') == 'video':
                            v = item.get('video', {})
                            vs = v.get('stats', {})
                            video_list.append({
                                'title': v.get('title', ''),
                                'views': vs.get('views', 0),
                                'duration': v.get('lengthSeconds', 0),
                                'published': v.get('publishedTimeText', ''),
                            })
                            print(f"    - {v.get('title', '')[:50]}... | {vs.get('views', 0):,} views")
                    info['recent_videos'] = video_list
        else:
            print(f"  ✗ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Save results
with open('/home/ubuntu/youtube_api_test_v2.json', 'w') as f:
    json.dump({
        'search_results': search_results,
        'channel_data': channel_data
    }, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print("SUMMARY OF AVAILABLE DATA")
print("="*60)
print(f"Channels successfully fetched: {len(channel_data)}")
for name, data in channel_data.items():
    print(f"  {name}: {data['subscribers']} | {data['total_videos']} videos | Joined: {data['joined']}")

print(f"\nKey fields available per channel:")
print(f"  - Channel metadata: name, description, subscribers, total views, country, join date")
print(f"  - Per video: title, views, duration (seconds), publish time")
print(f"\nResults saved to /home/ubuntu/youtube_api_test_v2.json")
