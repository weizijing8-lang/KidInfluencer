"""
Validate collected YouTube data and generate summary statistics.
Produces a comprehensive report of the dataset for the paper.
"""
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data/raw")
OUTPUT_DIR = Path("/home/ubuntu/KidInfluencer/data")

def load_all_data():
    """Load all channel JSON files into a unified DataFrame."""
    all_videos = []
    channel_meta = []
    
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        
        with open(f) as fp:
            data = json.load(fp)
        
        if data.get("error") or not data.get("videos"):
            channel_meta.append({
                "short_name": data.get("short_name", f.stem),
                "category": data.get("category", "unknown"),
                "channel_title": data.get("channel_title", "N/A"),
                "total_videos": 0,
                "status": "failed" if data.get("error") else "empty",
            })
            continue
        
        channel_meta.append({
            "short_name": data["short_name"],
            "category": data["category"],
            "channel_title": data.get("channel_title", "N/A"),
            "channel_id": data.get("channel_id", ""),
            "total_videos": data["total_videos"],
            "status": "success",
        })
        
        for v in data["videos"]:
            v["channel_short_name"] = data["short_name"]
            v["channel_category"] = data["category"]
            all_videos.append(v)
    
    df = pd.DataFrame(all_videos)
    channels_df = pd.DataFrame(channel_meta)
    return df, channels_df


def validate_and_summarize():
    """Run validation checks and generate summary statistics."""
    print("Loading data...")
    df, channels_df = load_all_data()
    
    print(f"\n{'='*60}")
    print("DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    
    # Channel-level summary
    print(f"\n--- CHANNEL SUMMARY ---")
    print(f"Total channels attempted: {len(channels_df)}")
    print(f"  Successful: {(channels_df['status'] == 'success').sum()}")
    print(f"  Failed: {(channels_df['status'] == 'failed').sum()}")
    print(f"  Empty: {(channels_df['status'] == 'empty').sum()}")
    
    successful = channels_df[channels_df['status'] == 'success']
    print(f"\n  Family channels (treatment): {(successful['category'] == 'family').sum()}")
    print(f"  Adult channels (control): {(successful['category'] == 'adult').sum()}")
    
    # Video-level summary
    print(f"\n--- VIDEO SUMMARY ---")
    print(f"Total videos: {len(df):,}")
    print(f"  Family channel videos: {(df['channel_category'] == 'family').sum():,}")
    print(f"  Adult channel videos: {(df['channel_category'] == 'adult').sum():,}")
    
    # View count validation
    print(f"\n--- VIEW COUNT VALIDATION ---")
    non_zero_views = (df['viewCount'] > 0).sum()
    print(f"Videos with view count > 0: {non_zero_views:,} / {len(df):,} ({100*non_zero_views/len(df):.1f}%)")
    print(f"Mean view count: {df['viewCount'].mean():,.0f}")
    print(f"Median view count: {df['viewCount'].median():,.0f}")
    print(f"Max view count: {df['viewCount'].max():,.0f}")
    
    # Like count validation
    non_zero_likes = (df['likeCount'] > 0).sum()
    print(f"\nVideos with like count > 0: {non_zero_likes:,} / {len(df):,} ({100*non_zero_likes/len(df):.1f}%)")
    
    # Comment count validation
    non_zero_comments = (df['commentCount'] > 0).sum()
    print(f"Videos with comment count > 0: {non_zero_comments:,} / {len(df):,} ({100*non_zero_comments/len(df):.1f}%)")
    
    # Date validation
    print(f"\n--- DATE RANGE ---")
    df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    valid_dates = df['publishedAt_dt'].notna().sum()
    print(f"Videos with valid dates: {valid_dates:,} / {len(df):,}")
    if valid_dates > 0:
        print(f"Date range: {df['publishedAt_dt'].min()} to {df['publishedAt_dt'].max()}")
    
    # Per-channel statistics
    print(f"\n--- PER-CHANNEL STATISTICS ---")
    channel_stats = df.groupby(['channel_short_name', 'channel_category']).agg(
        n_videos=('id', 'count'),
        total_views=('viewCount', 'sum'),
        mean_views=('viewCount', 'mean'),
        median_views=('viewCount', 'median'),
        max_views=('viewCount', 'max'),
        total_likes=('likeCount', 'sum'),
        total_comments=('commentCount', 'sum'),
    ).reset_index()
    
    channel_stats = channel_stats.sort_values('total_views', ascending=False)
    
    print(f"\nTop 10 channels by total views:")
    for _, row in channel_stats.head(10).iterrows():
        print(f"  {row['channel_short_name']:20s} ({row['channel_category']:6s}): "
              f"{row['n_videos']:5d} videos, {row['total_views']:>15,.0f} views")
    
    # Family vs Adult comparison
    print(f"\n--- FAMILY vs ADULT COMPARISON ---")
    family_df = df[df['channel_category'] == 'family']
    adult_df = df[df['channel_category'] == 'adult']
    
    print(f"{'Metric':<25s} {'Family':>15s} {'Adult':>15s}")
    print(f"{'-'*55}")
    print(f"{'Total videos':<25s} {len(family_df):>15,d} {len(adult_df):>15,d}")
    print(f"{'Mean views/video':<25s} {family_df['viewCount'].mean():>15,.0f} {adult_df['viewCount'].mean():>15,.0f}")
    print(f"{'Median views/video':<25s} {family_df['viewCount'].median():>15,.0f} {adult_df['viewCount'].median():>15,.0f}")
    print(f"{'Mean likes/video':<25s} {family_df['likeCount'].mean():>15,.0f} {adult_df['likeCount'].mean():>15,.0f}")
    print(f"{'Mean comments/video':<25s} {family_df['commentCount'].mean():>15,.0f} {adult_df['commentCount'].mean():>15,.0f}")
    
    # Save processed data
    print(f"\n--- SAVING PROCESSED DATA ---")
    
    # Save channel stats
    channel_stats_file = OUTPUT_DIR / "channel_statistics.csv"
    channel_stats.to_csv(channel_stats_file, index=False)
    print(f"Channel statistics: {channel_stats_file}")
    
    # Save full video dataframe (without description to save space)
    video_df_file = OUTPUT_DIR / "all_videos.csv"
    df_save = df.drop(columns=['description', 'tags'], errors='ignore')
    df_save.to_csv(video_df_file, index=False)
    print(f"All videos CSV: {video_df_file} ({len(df_save):,} rows)")
    
    # Save failed channels list
    failed = channels_df[channels_df['status'] != 'success']
    if len(failed) > 0:
        print(f"\n--- FAILED CHANNELS ({len(failed)}) ---")
        for _, row in failed.iterrows():
            print(f"  {row['short_name']:20s} ({row['category']:6s}): {row['status']}")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    
    return df, channels_df, channel_stats


if __name__ == "__main__":
    df, channels_df, channel_stats = validate_and_summarize()
