#!/usr/bin/env python3
"""
Test YouTube API to see what data we can collect for family vlog / kidfluencer channels.
"""

import sys
import json

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()

# Known family vlog / kidfluencer channels
test_channels = [
    "https://www.youtube.com/@RyansToyReview",      # Ryan's World - one of the biggest kidfluencers
    "https://www.youtube.com/@TheACEFamily",          # The ACE Family - controversial family vlog
    "https://www.youtube.com/@FamilyFunPack",         # Family Fun Pack
    "https://www.youtube.com/@TheLaBrantFam",         # The LaBrant Fam - known for clickbait with kids
    "https://www.youtube.com/@JesssFam",              # Jess Fam - family vlog
]

all_results = {}

for channel_url in test_channels:
    print(f"\n{'='*60}")
    print(f"Fetching: {channel_url}")
    print('='*60)
    
    try:
        # Get channel details
        result = client.call_api('Youtube/get_channel_details', query={
            'id': channel_url,
            'hl': 'en'
        })
        
        if result:
            channel_name = result.get('title', 'Unknown')
            stats = result.get('stats', {})
            
            print(f"Channel: {channel_name}")
            print(f"Subscribers: {stats.get('subscribersText', 'N/A')}")
            print(f"Total Videos: {stats.get('videos', 'N/A')}")
            print(f"Total Views: {stats.get('views', 'N/A')}")
            print(f"Country: {result.get('country', 'N/A')}")
            print(f"Joined: {result.get('joinedDate', 'N/A')}")
            print(f"Description: {result.get('description', 'N/A')[:200]}")
            
            # Get channel ID for video fetching
            channel_id = result.get('channelId', '')
            
            all_results[channel_name] = {
                'channel_id': channel_id,
                'channel_url': channel_url,
                'subscribers': stats.get('subscribers', 0),
                'subscribers_text': stats.get('subscribersText', 'N/A'),
                'total_videos': stats.get('videos', 0),
                'total_views': stats.get('views', 0),
                'country': result.get('country', 'N/A'),
                'joined': result.get('joinedDate', 'N/A'),
                'description': result.get('description', 'N/A')[:500],
            }
            
            # Now get latest videos
            if channel_id:
                print(f"\nFetching latest videos for {channel_name}...")
                videos_result = client.call_api('Youtube/get_channel_videos', query={
                    'id': channel_id,
                    'filter': 'videos_latest',
                    'hl': 'en',
                    'gl': 'US'
                })
                
                if videos_result:
                    contents = videos_result.get('contents', [])
                    print(f"Got {len(contents)} videos")
                    
                    video_list = []
                    for i, item in enumerate(contents[:5]):
                        if item.get('type') == 'video':
                            video = item.get('video', {})
                            v_stats = video.get('stats', {})
                            video_info = {
                                'title': video.get('title', 'N/A'),
                                'video_id': video.get('videoId', 'N/A'),
                                'published': video.get('publishedTimeText', 'N/A'),
                                'duration_seconds': video.get('lengthSeconds', 0),
                                'views': v_stats.get('views', 0),
                            }
                            video_list.append(video_info)
                            print(f"  [{i+1}] {video_info['title'][:60]}... | Views: {video_info['views']:,} | Duration: {video_info['duration_seconds']}s | Published: {video_info['published']}")
                    
                    all_results[channel_name]['sample_videos'] = video_list
                    all_results[channel_name]['total_videos_fetched'] = len(contents)
        else:
            print(f"No data returned for {channel_url}")
            
    except Exception as e:
        print(f"Error: {e}")

# Save all results
with open('/home/ubuntu/youtube_api_test_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Successfully fetched data for {len(all_results)} channels")
for name, data in all_results.items():
    print(f"  - {name}: {data.get('subscribers_text', 'N/A')} subscribers, {data.get('total_videos', 'N/A')} videos")
print(f"\nFull results saved to /home/ubuntu/youtube_api_test_results.json")
