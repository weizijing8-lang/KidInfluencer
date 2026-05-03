#!/usr/bin/env python3
"""Debug: inspect raw video API response."""
import sys, json
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

client = ApiClient()

# First get the channel ID from channel details
print("=== Step 1: Get channel details to find channelId ===")
resp = client.call_api('Youtube/get_channel_details', query={
    'id': 'https://www.youtube.com/@KidsdianaShow',
    'hl': 'en'
})
channel_id = resp.get('channelId', '')
print(f"Channel ID: {channel_id}")
print(f"Title: {resp.get('title', '')}")

# Try video API with the actual channel ID
print("\n=== Step 2: Try get_channel_videos with channelId ===")
resp2 = client.call_api('Youtube/get_channel_videos', query={
    'id': channel_id,
    'filter': 'videos_latest',
    'hl': 'en',
    'gl': 'US'
})
print(f"Response keys: {list(resp2.keys()) if resp2 else 'None'}")
print(f"Contents count: {len(resp2.get('contents', []))}")
if resp2:
    # Save full response for inspection
    with open('/home/ubuntu/KidInfluencer/data/debug_videos_response.json', 'w') as f:
        json.dump(resp2, f, indent=2, default=str)
    print("Full response saved to debug_videos_response.json")
    # Show first item if exists
    contents = resp2.get('contents', [])
    if contents:
        print(f"\nFirst item type: {contents[0].get('type', 'unknown')}")
        print(json.dumps(contents[0], indent=2, default=str)[:500])

# Also try with URL directly
print("\n=== Step 3: Try get_channel_videos with URL ===")
resp3 = client.call_api('Youtube/get_channel_videos', query={
    'id': 'https://www.youtube.com/@KidsdianaShow',
    'filter': 'videos_latest',
    'hl': 'en',
    'gl': 'US'
})
print(f"Response keys: {list(resp3.keys()) if resp3 else 'None'}")
print(f"Contents count: {len(resp3.get('contents', []))}")
if resp3 and resp3.get('contents'):
    print(f"First item: {json.dumps(resp3['contents'][0], indent=2, default=str)[:500]}")
