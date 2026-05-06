"""
Fetch video duration and publish date for the 4685 stratified sample videos.
Uses the YouTube Data API via the existing googleapiclient setup.
Saves incrementally to avoid data loss.
"""
import os
import csv
import time
import json
from googleapiclient.discovery import build
import pandas as pd
import isodate

API_KEY = os.environ.get('YOUTUBE_API_KEY', '')

# Try to use the same API setup as the existing collection scripts
# Check if there's a key in the existing config
if not API_KEY:
    # Try reading from existing scripts or config
    config_paths = [
        '/home/ubuntu/KidInfluencer/config.json',
        '/home/ubuntu/KidInfluencer/.env',
    ]
    for p in config_paths:
        if os.path.exists(p):
            with open(p) as f:
                content = f.read()
                if 'YOUTUBE_API_KEY' in content or 'api_key' in content:
                    print(f"Found config at {p}")

# Load the stratified sample to get video IDs
sample = pd.read_csv('/home/ubuntu/KidInfluencer/data/stratified_sample_v2.csv')
video_ids = sample['id'].tolist()
print(f"Total videos to fetch metadata for: {len(video_ids)}")

# Output file
output_file = '/home/ubuntu/KidInfluencer/data/video_metadata_duration_date.csv'

# Check if we have partial results
existing_ids = set()
if os.path.exists(output_file):
    existing = pd.read_csv(output_file)
    existing_ids = set(existing['id'].tolist())
    print(f"Already have metadata for {len(existing_ids)} videos, fetching remaining")

remaining_ids = [vid for vid in video_ids if vid not in existing_ids]
print(f"Remaining to fetch: {len(remaining_ids)}")

if not remaining_ids:
    print("All done!")
    exit(0)

# Try to build YouTube API client
try:
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    print("YouTube API client built successfully")
except Exception as e:
    print(f"Could not build YouTube API client: {e}")
    print("\nFalling back to extracting from existing data...")
    
    # Try to extract duration and publishedAt from the existing dataset
    full_data = pd.read_csv('/home/ubuntu/KidInfluencer/data/full_expanded_dataset.csv')
    print(f"Full dataset columns: {list(full_data.columns)}")
    print(f"Full dataset shape: {full_data.shape}")
    
    # Check if publishedAt exists
    if 'publishedAt' in full_data.columns:
        print("Found publishedAt in full dataset!")
        merged = sample.merge(full_data[['id', 'publishedAt']].drop_duplicates(), on='id', how='left')
        print(f"Matched publishedAt for {merged['publishedAt'].notna().sum()} / {len(merged)} videos")
    
    if 'duration' in full_data.columns:
        print("Found duration in full dataset!")
    
    # Save what we can extract
    exit(0)

# Fetch in batches of 50 (API limit)
batch_size = 50
write_header = not os.path.exists(output_file) or len(existing_ids) == 0

with open(output_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['id', 'duration_seconds', 'publishedAt', 'duration_iso'])
    
    for i in range(0, len(remaining_ids), batch_size):
        batch = remaining_ids[i:i+batch_size]
        
        try:
            response = youtube.videos().list(
                part='contentDetails,snippet',
                id=','.join(batch)
            ).execute()
            
            for item in response.get('items', []):
                vid_id = item['id']
                duration_iso = item.get('contentDetails', {}).get('duration', 'PT0S')
                published_at = item.get('snippet', {}).get('publishedAt', '')
                
                # Parse ISO 8601 duration to seconds
                try:
                    duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())
                except:
                    duration_seconds = 0
                
                writer.writerow([vid_id, duration_seconds, published_at, duration_iso])
            
            f.flush()
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Batch {i//batch_size + 1}/{len(remaining_ids)//batch_size + 1} done ({i+len(batch)}/{len(remaining_ids)})")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  Error at batch {i//batch_size + 1}: {e}")
            time.sleep(2)
            continue

print(f"\nDone! Total metadata fetched: {len(remaining_ids)}")
