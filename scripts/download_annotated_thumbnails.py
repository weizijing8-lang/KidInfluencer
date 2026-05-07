"""Download thumbnails for all 23 annotated videos."""
import os
import requests

video_ids = [
    "1BcbDYtORH0", "mEcLEyDi3bw", "-qKBVTFhKHw", "ijeCnHEx8W4", "0SmLtajf--U",
    "xpmU8dCteIw", "QhuaVZJpW6o", "AdFxzIdSZek", "aN--3LU_cb4", "o4yYkq0k12Y",
    "RElUn9WRxbY", "8AnIUAdvITA", "Jv8bR-R7OIY", "Af8OIPU_0f8", "TEa1UHCYRBs",
    "XwuxFjVl3qQ", "uDKMJKCvUCg", "5NCSLIRxO7o", "_3HmV2xtgiU", "-emarHZ0GBY",
    "oUBEYfw62qI", "XhZpzNlN2uA", "xlQgJSC6H6s"
]

out_dir = "/home/ubuntu/KidInfluencer/thumbnails_test"
os.makedirs(out_dir, exist_ok=True)

success = 0
for vid in video_ids:
    out_path = os.path.join(out_dir, f"{vid}.jpg")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  ✅ {vid} (already exists)")
        success += 1
        continue
    
    # Try maxresdefault first, then hqdefault
    for quality in ['maxresdefault', 'hqdefault', 'mqdefault']:
        url = f"https://img.youtube.com/vi/{vid}/{quality}.jpg"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 2000:
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                success += 1
                print(f"  ✅ {vid} ({quality})")
                break
        except:
            continue
    else:
        print(f"  ❌ {vid} - failed all qualities")

print(f"\nDownloaded: {success}/{len(video_ids)}")
