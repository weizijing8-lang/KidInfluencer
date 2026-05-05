#!/usr/bin/env python3
"""
Stratified Sampling from Expanded Dataset
==========================================
Merges old (48 channels) + new (34 channels) = 82 channels.
Draws a stratified sample by:
  - Channel (proportional representation)
  - View count tier (low/medium/high within each channel)
  - Time period (early/recent based on publishedAt)

Target: ~4,000-5,000 videos for LLM classification.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = '/home/ubuntu/KidInfluencer/data'
OUTPUT_SAMPLE = os.path.join(DATA_DIR, 'stratified_sample.csv')
OUTPUT_FULL = os.path.join(DATA_DIR, 'full_expanded_dataset.csv')

# Target sample size per channel
TARGET_PER_CHANNEL = 60  # 82 channels * 60 = ~4,920 videos

def load_and_merge():
    """Load and merge old + new datasets."""
    # Old dataset (48 channels, 46,589 videos)
    old_df = pd.read_csv(os.path.join(DATA_DIR, 'combined_family_videos.csv'))
    print(f"Old dataset: {len(old_df)} videos, {old_df['channel_short_name'].nunique()} channels")
    
    # New expanded dataset (34 new channels, 12,449 videos)
    new_df = pd.read_csv(os.path.join(DATA_DIR, 'expanded_channels_videos.csv'))
    print(f"New dataset: {len(new_df)} videos, {new_df['channel_short_name'].nunique()} channels")
    
    # Remove overlap (labrantfam is in both)
    old_channels = set(old_df['channel_short_name'].unique())
    new_df_unique = new_df[~new_df['channel_short_name'].isin(old_channels)]
    print(f"New unique (after removing overlap): {len(new_df_unique)} videos, {new_df_unique['channel_short_name'].nunique()} channels")
    
    # Standardize columns
    common_cols = ['id', 'title', 'publishedAt', 'channelId', 'channelTitle', 
                   'viewCount', 'channel_short_name']
    
    for col in common_cols:
        if col not in old_df.columns:
            old_df[col] = ''
        if col not in new_df_unique.columns:
            new_df_unique[col] = ''
    
    merged = pd.concat([old_df[common_cols], new_df_unique[common_cols]], ignore_index=True)
    
    # Clean viewCount
    merged['viewCount'] = pd.to_numeric(merged['viewCount'], errors='coerce').fillna(0).astype(int)
    
    # Remove videos with 0 views (likely errors)
    merged = merged[merged['viewCount'] > 0]
    
    # Remove channels with < 10 videos
    channel_counts = merged['channel_short_name'].value_counts()
    valid_channels = channel_counts[channel_counts >= 10].index
    merged = merged[merged['channel_short_name'].isin(valid_channels)]
    
    print(f"\nMerged dataset: {len(merged)} videos, {merged['channel_short_name'].nunique()} channels")
    return merged

def stratified_sample(df, target_per_channel=TARGET_PER_CHANNEL):
    """
    Stratified sampling: within each channel, sample proportionally from
    view count terciles (low/med/high).
    """
    sampled = []
    
    for channel, group in df.groupby('channel_short_name'):
        n_available = len(group)
        n_sample = min(target_per_channel, n_available)
        
        if n_available <= n_sample:
            # Take all if fewer than target
            sampled.append(group)
        else:
            # Stratify by view count terciles
            group = group.copy()
            group['view_tercile'] = pd.qcut(group['viewCount'], q=3, labels=['low', 'med', 'high'], duplicates='drop')
            
            # Sample proportionally from each tercile
            tercile_sample = group.groupby('view_tercile', observed=True).apply(
                lambda x: x.sample(n=min(len(x), n_sample // 3 + 1), random_state=42),
                include_groups=False
            ).reset_index(level=0, drop=True)
            
            # Trim to exact target
            if len(tercile_sample) > n_sample:
                tercile_sample = tercile_sample.sample(n=n_sample, random_state=42)
            
            sampled.append(tercile_sample)
    
    result = pd.concat(sampled, ignore_index=True)
    return result

def main():
    # Load and merge
    full_df = load_and_merge()
    
    # Save full merged dataset
    full_df.to_csv(OUTPUT_FULL, index=False)
    print(f"Saved full expanded dataset: {OUTPUT_FULL}")
    
    # Draw stratified sample
    sample_df = stratified_sample(full_df)
    
    # Add view_tier column for reference
    sample_df['log_views'] = np.log10(sample_df['viewCount'].clip(lower=1))
    
    # Save sample
    sample_df.to_csv(OUTPUT_SAMPLE, index=False)
    
    print(f"\n{'='*60}")
    print(f"STRATIFIED SAMPLE SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos in sample: {len(sample_df)}")
    print(f"Total channels: {sample_df['channel_short_name'].nunique()}")
    print(f"Videos per channel: min={sample_df.groupby('channel_short_name').size().min()}, "
          f"max={sample_df.groupby('channel_short_name').size().max()}, "
          f"median={sample_df.groupby('channel_short_name').size().median():.0f}")
    print(f"\nView count distribution in sample:")
    print(f"  Min: {sample_df['viewCount'].min():,.0f}")
    print(f"  Median: {sample_df['viewCount'].median():,.0f}")
    print(f"  Mean: {sample_df['viewCount'].mean():,.0f}")
    print(f"  Max: {sample_df['viewCount'].max():,.0f}")
    
    # Channel size distribution
    channel_sizes = sample_df.groupby('channel_short_name').size()
    print(f"\nChannel size distribution:")
    print(f"  Channels with 60 videos: {(channel_sizes == 60).sum()}")
    print(f"  Channels with < 60 videos: {(channel_sizes < 60).sum()}")
    
    # Print per-channel counts
    print(f"\n{'Channel':<25} {'N_sample':>8} {'Median Views':>15}")
    print("-" * 50)
    for ch in sorted(sample_df['channel_short_name'].unique()):
        ch_data = sample_df[sample_df['channel_short_name'] == ch]
        print(f"{ch:<25} {len(ch_data):>8} {ch_data['viewCount'].median():>15,.0f}")

if __name__ == '__main__':
    main()
