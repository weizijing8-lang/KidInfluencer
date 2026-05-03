#!/usr/bin/env python3
"""
Kidfluencer Dataset Descriptive Analysis
=========================================
Produces summary statistics and visualizations for the collected data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ── Config ──
DATA_DIR = '/home/ubuntu/KidInfluencer/data'
OUT_DIR = '/home/ubuntu/KidInfluencer/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Load Data ──
# Find the latest all_channels and all_videos files
import glob
ch_files = sorted(glob.glob(os.path.join(DATA_DIR, 'all_channels_*.csv')))
vid_files = sorted(glob.glob(os.path.join(DATA_DIR, 'all_videos_*.csv')))

if not ch_files or not vid_files:
    print("ERROR: No data files found!")
    exit(1)

channels = pd.read_csv(ch_files[-1])
videos = pd.read_csv(vid_files[-1])

print(f"Loaded {len(channels)} channels, {len(videos)} videos")
print(f"Channel file: {ch_files[-1]}")
print(f"Video file: {vid_files[-1]}")

# ── Clean Data ──
# Filter out tiny/fake channels (< 1000 subscribers)
channels_clean = channels[channels['subscribers'] >= 1000].copy()
print(f"\nAfter filtering (>= 1000 subs): {len(channels_clean)} channels")

# Merge videos with channel info
videos_merged = videos.merge(
    channels_clean[['channel_id', 'title', 'subscribers', 'total_videos', 'total_views']],
    on='channel_id', how='inner', suffixes=('', '_channel')
)
print(f"Videos with valid channels: {len(videos_merged)}")

# ═══════════════════════════════════════════════════════════════
# 1. CHANNEL-LEVEL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. CHANNEL-LEVEL SUMMARY")
print("="*60)

summary = channels_clean[['subscribers', 'total_videos', 'total_views']].describe()
print(summary)

# Save summary
summary.to_csv(os.path.join(OUT_DIR, 'channel_summary_stats.csv'))

# ── Fig 1: Subscriber Distribution ──
fig, ax = plt.subplots()
subs_millions = channels_clean['subscribers'] / 1e6
ax.hist(subs_millions, bins=20, color='#2196F3', edgecolor='white', alpha=0.8)
ax.set_xlabel('Subscribers (millions)')
ax.set_ylabel('Number of Channels')
ax.set_title('Distribution of Subscriber Counts Across Kidfluencer Channels')
ax.axvline(subs_millions.median(), color='red', linestyle='--', label=f'Median: {subs_millions.median():.1f}M')
ax.legend()
fig.savefig(os.path.join(OUT_DIR, 'fig1_subscriber_distribution.png'))
plt.close()
print("Saved fig1_subscriber_distribution.png")

# ── Fig 2: Top 15 Channels by Subscribers ──
fig, ax = plt.subplots(figsize=(12, 7))
top15 = channels_clean.nlargest(15, 'subscribers')
bars = ax.barh(range(len(top15)), top15['subscribers'] / 1e6, color='#FF9800', edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['title'].values)
ax.set_xlabel('Subscribers (millions)')
ax.set_title('Top 15 Kidfluencer Channels by Subscriber Count')
ax.invert_yaxis()
for bar, val in zip(bars, top15['subscribers'] / 1e6):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}M', va='center', fontsize=9)
fig.savefig(os.path.join(OUT_DIR, 'fig2_top15_channels.png'))
plt.close()
print("Saved fig2_top15_channels.png")

# ── Fig 3: Cross-Platform Presence ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3a: Platform breakdown
platforms = ['has_instagram', 'has_tiktok', 'has_twitter', 'has_facebook']
platform_labels = ['Instagram', 'TikTok', 'Twitter/X', 'Facebook']
platform_counts = [channels_clean[p].sum() for p in platforms]
axes[0].bar(platform_labels, platform_counts, color=['#E1306C', '#000000', '#1DA1F2', '#4267B2'])
axes[0].set_ylabel('Number of Channels')
axes[0].set_title('Cross-Platform Presence')
for i, v in enumerate(platform_counts):
    axes[0].text(i, v + 0.3, f'{v} ({v/len(channels_clean)*100:.0f}%)', ha='center', fontsize=9)

# 3b: Cross-platform count distribution
cp_counts = channels_clean['cross_platform_count'].value_counts().sort_index()
axes[1].bar(cp_counts.index, cp_counts.values, color='#4CAF50', edgecolor='white')
axes[1].set_xlabel('Number of Other Platforms Linked')
axes[1].set_ylabel('Number of Channels')
axes[1].set_title('Cross-Platform Synchronization Level')

fig.suptitle('Social Media Ecosystem of Kidfluencer Channels', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_cross_platform.png'))
plt.close()
print("Saved fig3_cross_platform.png")

# ═══════════════════════════════════════════════════════════════
# 2. VIDEO-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. VIDEO-LEVEL ANALYSIS")
print("="*60)

# Duration stats
videos_merged['duration_minutes'] = videos_merged['length_seconds'] / 60
print(f"\nVideo duration (minutes):")
print(videos_merged['duration_minutes'].describe())

# ── Fig 4: Video Duration Distribution ──
fig, ax = plt.subplots()
dur = videos_merged['duration_minutes']
dur_clipped = dur[dur <= 60]  # clip at 60 min for visualization
ax.hist(dur_clipped, bins=40, color='#9C27B0', edgecolor='white', alpha=0.8)
ax.set_xlabel('Video Duration (minutes)')
ax.set_ylabel('Number of Videos')
ax.set_title('Distribution of Video Durations')
ax.axvline(dur.median(), color='red', linestyle='--', label=f'Median: {dur.median():.1f} min')
ax.legend()
fig.savefig(os.path.join(OUT_DIR, 'fig4_duration_distribution.png'))
plt.close()
print("Saved fig4_duration_distribution.png")

# ── Fig 5: Views Distribution (log scale) ──
fig, ax = plt.subplots()
views_nonzero = videos_merged['views'][videos_merged['views'] > 0]
ax.hist(np.log10(views_nonzero), bins=40, color='#F44336', edgecolor='white', alpha=0.8)
ax.set_xlabel('Log10(Views)')
ax.set_ylabel('Number of Videos')
ax.set_title('Distribution of Video Views (Log Scale)')
ax.axvline(np.log10(views_nonzero.median()), color='blue', linestyle='--',
           label=f'Median: {views_nonzero.median():,.0f} views')
ax.legend()
fig.savefig(os.path.join(OUT_DIR, 'fig5_views_distribution.png'))
plt.close()
print("Saved fig5_views_distribution.png")

# ═══════════════════════════════════════════════════════════════
# 3. RISK INDICATORS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. RISK INDICATORS")
print("="*60)

# Emotional title analysis
emotional_by_channel = videos_merged.groupby('channel_id').agg(
    channel_title=('title_channel', 'first'),
    total_vids=('video_id', 'count'),
    emotional_count=('has_emotional_title', 'sum'),
    avg_views=('views', 'mean'),
    subscribers=('subscribers', 'first'),
).reset_index()
emotional_by_channel['emotional_rate'] = emotional_by_channel['emotional_count'] / emotional_by_channel['total_vids']

print("\nEmotional Title Rate by Channel (top 10):")
top_emotional = emotional_by_channel.nlargest(10, 'emotional_rate')
for _, row in top_emotional.iterrows():
    print(f"  {row['channel_title'][:35]:35s} | {row['emotional_count']:2d}/{row['total_vids']:2d} ({row['emotional_rate']:.0%})")

# ── Fig 6: Emotional Title Rate vs Subscribers ──
fig, ax = plt.subplots()
sc = ax.scatter(
    emotional_by_channel['subscribers'] / 1e6,
    emotional_by_channel['emotional_rate'] * 100,
    s=emotional_by_channel['total_vids'] * 3,
    alpha=0.6, c='#FF5722', edgecolors='white'
)
ax.set_xlabel('Subscribers (millions)')
ax.set_ylabel('Emotional Title Rate (%)')
ax.set_title('Emotional Manipulation in Titles vs Channel Size')
# Label top outliers
for _, row in emotional_by_channel.nlargest(5, 'emotional_rate').iterrows():
    ax.annotate(row['channel_title'][:20], (row['subscribers']/1e6, row['emotional_rate']*100),
                fontsize=7, alpha=0.8)
fig.savefig(os.path.join(OUT_DIR, 'fig6_emotional_vs_size.png'))
plt.close()
print("Saved fig6_emotional_vs_size.png")

# ── Fig 7: Do emotional titles get more views? ──
fig, ax = plt.subplots()
emotional_vids = videos_merged[videos_merged['has_emotional_title'] == True]
normal_vids = videos_merged[videos_merged['has_emotional_title'] == False]

categories = ['Normal Titles', 'Emotional Titles']
medians = [normal_vids['views'].median(), emotional_vids['views'].median()]
means = [normal_vids['views'].mean(), emotional_vids['views'].mean()]

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, [m/1e6 for m in medians], width, label='Median', color='#2196F3')
bars2 = ax.bar(x + width/2, [m/1e6 for m in means], width, label='Mean', color='#FF9800')
ax.set_ylabel('Views (millions)')
ax.set_title('Do Emotional Titles Get More Views?')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}M',
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}M',
            ha='center', va='bottom', fontsize=9)
fig.savefig(os.path.join(OUT_DIR, 'fig7_emotional_views_comparison.png'))
plt.close()
print("Saved fig7_emotional_views_comparison.png")

# ═══════════════════════════════════════════════════════════════
# 4. CONTENT PRODUCTION INTENSITY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. CONTENT PRODUCTION INTENSITY")
print("="*60)

# Videos per channel stats
vids_per_channel = videos_merged.groupby('channel_id').agg(
    channel_title=('title_channel', 'first'),
    subscribers=('subscribers', 'first'),
    total_videos_on_channel=('total_videos', 'first'),
    avg_duration_min=('duration_minutes', 'mean'),
    total_duration_min=('duration_minutes', 'sum'),
    avg_views=('views', 'mean'),
).reset_index()

# ── Fig 8: Total Videos vs Subscribers (production intensity) ──
fig, ax = plt.subplots()
ax.scatter(
    vids_per_channel['subscribers'] / 1e6,
    vids_per_channel['total_videos_on_channel'],
    s=60, alpha=0.6, c='#673AB7', edgecolors='white'
)
ax.set_xlabel('Subscribers (millions)')
ax.set_ylabel('Total Videos on Channel')
ax.set_title('Content Production Volume vs Channel Size')
# Label outliers
for _, row in vids_per_channel.nlargest(5, 'total_videos_on_channel').iterrows():
    ax.annotate(row['channel_title'][:20],
                (row['subscribers']/1e6, row['total_videos_on_channel']),
                fontsize=7, alpha=0.8)
fig.savefig(os.path.join(OUT_DIR, 'fig8_production_vs_size.png'))
plt.close()
print("Saved fig8_production_vs_size.png")

# ═══════════════════════════════════════════════════════════════
# 5. SAVE FULL ANALYSIS REPORT
# ═══════════════════════════════════════════════════════════════
report = {
    'dataset_overview': {
        'total_channels': len(channels_clean),
        'total_videos_sampled': len(videos_merged),
        'channels_with_1M_plus_subs': int((channels_clean['subscribers'] >= 1e6).sum()),
        'channels_with_10M_plus_subs': int((channels_clean['subscribers'] >= 1e7).sum()),
    },
    'subscriber_stats': {
        'median': float(channels_clean['subscribers'].median()),
        'mean': float(channels_clean['subscribers'].mean()),
        'max': float(channels_clean['subscribers'].max()),
        'min': float(channels_clean['subscribers'].min()),
    },
    'video_stats': {
        'median_duration_min': float(videos_merged['duration_minutes'].median()),
        'mean_duration_min': float(videos_merged['duration_minutes'].mean()),
        'median_views': float(videos_merged['views'].median()),
        'mean_views': float(videos_merged['views'].mean()),
    },
    'risk_indicators': {
        'total_emotional_titles': int(videos_merged['has_emotional_title'].sum()),
        'emotional_title_rate': float(videos_merged['has_emotional_title'].mean()),
        'total_commercial_videos': int(videos_merged['is_commercial'].sum()),
        'commercial_rate': float(videos_merged['is_commercial'].mean()),
    },
    'cross_platform': {
        'channels_with_instagram': int(channels_clean['has_instagram'].sum()),
        'channels_with_tiktok': int(channels_clean['has_tiktok'].sum()),
        'channels_with_twitter': int(channels_clean['has_twitter'].sum()),
        'channels_with_facebook': int(channels_clean['has_facebook'].sum()),
        'avg_cross_platform_count': float(channels_clean['cross_platform_count'].mean()),
    },
}

with open(os.path.join(OUT_DIR, 'analysis_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(json.dumps(report, indent=2))
print(f"\nAll outputs saved to: {OUT_DIR}")
