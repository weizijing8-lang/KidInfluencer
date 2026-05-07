"""
Fetch video descriptions for the 23 user-annotated videos via YouTube Data API v3.
"""
import json
import requests

API_KEY = "AIzaSyC17NxPT0HPVaXihtyNtmvBhH4Mh6GdowU"

video_ids = [
    "1BcbDYtORH0", "mEcLEyDi3bw", "-qKBVTFhKHw", "ijeCnHEx8W4", "0SmLtajf--U",
    "xpmU8dCteIw", "QhuaVZJpW6o", "AdFxzIdSZek", "aN--3LU_cb4", "o4yYkq0k12Y",
    "RElUn9WRxbY", "8AnIUAdvITA", "Jv8bR-R7OIY", "Af8OIPU_0f8", "TEa1UHCYRBs",
    "XwuxFjVl3qQ", "uDKMJKCvUCg", "5NCSLIRxO7o", "_3HmV2xtgiU", "-emarHZ0GBY",
    "oUBEYfw62qI", "XhZpzNlN2uA", "xlQgJSC6H6s"
]

print(f"Fetching descriptions for {len(video_ids)} videos...")

# YouTube API allows up to 50 IDs per request
ids_str = ",".join(video_ids)

url = "https://www.googleapis.com/youtube/v3/videos"
params = {
    "part": "snippet",
    "id": ids_str,
    "key": API_KEY,
    "maxResults": 50,
}

resp = requests.get(url, params=params, timeout=30)
print(f"API Response Status: {resp.status_code}")

if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
    exit(1)

data = resp.json()
items = data.get("items", [])
print(f"Got {len(items)} video results")

# Store results
results = {}
for item in items:
    vid = item["id"]
    snippet = item.get("snippet", {})
    results[vid] = {
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
        "channelTitle": snippet.get("channelTitle", ""),
        "publishedAt": snippet.get("publishedAt", ""),
    }

# Save to file
output_path = "/home/ubuntu/KidInfluencer/data/descriptions/annotated_23_descriptions.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {output_path}")
print(f"\nVideos found: {len(results)}/{len(video_ids)}")

# Show which videos were NOT found
missing = [vid for vid in video_ids if vid not in results]
if missing:
    print(f"\nMissing videos ({len(missing)}): {missing}")

# Print descriptions
print("\n" + "=" * 80)
print("VIDEO DESCRIPTIONS")
print("=" * 80)

for vid in video_ids:
    if vid in results:
        r = results[vid]
        print(f"\n{'─' * 70}")
        print(f"📹 [{vid}] {r['title']}")
        print(f"   Channel: {r['channelTitle']}")
        print(f"   Tags: {r['tags'][:8]}")
        print(f"   Description (first 400 chars):")
        desc = r['description'][:400]
        for line in desc.split('\n')[:10]:
            print(f"     {line}")
    else:
        print(f"\n{'─' * 70}")
        print(f"❌ [{vid}] NOT FOUND")
