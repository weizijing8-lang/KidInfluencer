"""
Cross-Platform Analysis: YouTube vs TikTok
===========================================
Compare family vs adult channels across both platforms.
Key questions:
1. Do family channels have more TikTok presence than adult channels?
2. Is the engagement pattern different on TikTok?
3. Do high-exploitation YouTube channels also have high TikTok activity?
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

BASE_DIR = '/home/ubuntu/KidInfluencer'
TIKTOK_DIR = os.path.join(BASE_DIR, 'data', 'tiktok')
FIG_DIR = os.path.join(BASE_DIR, 'figures_tiktok')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. Load TikTok user info data (from V1 collection)
# ============================================================
print("Loading TikTok user info data...")

tiktok_users = []
for f in glob.glob(os.path.join(TIKTOK_DIR, '*_tiktok.json')):
    with open(f) as fh:
        data = json.load(fh)
    tiktok_users.append({
        'youtube_channel': data['youtube_channel'],
        'tiktok_username': data['tiktok_username'],
        'category': data['category'],
        'tiktok_followers': data['followers'],
        'tiktok_videos': data['video_count'],
        'tiktok_hearts': data['heart_count'],
        'tiktok_verified': data.get('verified', False),
    })

df_users = pd.DataFrame(tiktok_users)
print(f"Loaded {len(df_users)} TikTok user profiles")

# ============================================================
# 2. Load TikTok search data (from V2 collection)
# ============================================================
print("Loading TikTok search/video data...")

tiktok_videos = []
for f in glob.glob(os.path.join(TIKTOK_DIR, '*_tiktok_v2.json')):
    with open(f) as fh:
        data = json.load(fh)
    
    own_vids = data.get('own_videos', [])
    if own_vids:
        avg_plays = np.mean([v['playCount'] for v in own_vids])
        avg_likes = np.mean([v['diggCount'] for v in own_vids])
        avg_comments = np.mean([v['commentCount'] for v in own_vids])
        avg_shares = np.mean([v['shareCount'] for v in own_vids])
        durations = [v['duration'] for v in own_vids if 0 < v['duration'] < 600]  # filter outliers
        avg_duration = np.mean(durations) if durations else 0
    else:
        avg_plays = avg_likes = avg_comments = avg_shares = avg_duration = 0
    
    tiktok_videos.append({
        'youtube_channel': data['youtube_channel'],
        'tiktok_username': data['tiktok_username'],
        'category': data['category'],
        'own_videos_found': len(own_vids),
        'tiktok_avg_plays': avg_plays,
        'tiktok_avg_likes': avg_likes,
        'tiktok_avg_comments': avg_comments,
        'tiktok_avg_shares': avg_shares,
        'tiktok_avg_duration': avg_duration,
    })

df_videos = pd.DataFrame(tiktok_videos)
print(f"Loaded video data for {len(df_videos)} channels")

# ============================================================
# 3. Load YouTube data for comparison
# ============================================================
print("Loading YouTube data...")

yt_freq = pd.read_csv(os.path.join(BASE_DIR, 'data', 'results_v4', 'upload_frequency_metrics.csv'))
yt_exploit = pd.read_csv(os.path.join(BASE_DIR, 'data', 'results_v4', 'composite_exploitation_index.csv'))

# ============================================================
# 4. Merge all data
# ============================================================
print("Merging datasets...")

# Merge TikTok user + video data
df_tiktok = df_users.merge(df_videos, on=['youtube_channel', 'category'], how='outer', suffixes=('', '_v2'))

# Merge with YouTube data
df_merged = df_tiktok.merge(
    yt_freq[['channel', 'category', 'videos_per_week', 'n_videos']].rename(columns={'channel': 'youtube_channel'}),
    on=['youtube_channel', 'category'],
    how='left'
)

if 'channel' in yt_exploit.columns:
    df_merged = df_merged.merge(
        yt_exploit[['channel', 'composite_index']].rename(columns={'channel': 'youtube_channel'}),
        on='youtube_channel',
        how='left'
    )

# Filter to channels with meaningful TikTok presence (>1000 followers)
df_active = df_merged[df_merged['tiktok_followers'] > 1000].copy()

print(f"\nActive TikTok channels: {len(df_active)}")
print(f"  Family: {len(df_active[df_active['category']=='family'])}")
print(f"  Adult: {len(df_active[df_active['category']=='adult'])}")

# ============================================================
# 5. Cross-Platform Analysis
# ============================================================
print("\n" + "="*60)
print("CROSS-PLATFORM ANALYSIS")
print("="*60)

# 5a. TikTok presence comparison
print("\n--- TikTok Presence ---")
for cat in ['family', 'adult']:
    sub = df_merged[df_merged['category'] == cat]
    active = sub[sub['tiktok_followers'] > 1000]
    total = len(sub)
    n_active = len(active)
    print(f"{cat}: {n_active}/{total} have active TikTok ({n_active/total*100:.0f}%)")
    if n_active > 0:
        print(f"  Avg followers: {active['tiktok_followers'].mean():,.0f}")
        print(f"  Avg videos: {active['tiktok_videos'].mean():.0f}")
        print(f"  Avg hearts: {active['tiktok_hearts'].mean():,.0f}")

# 5b. Engagement comparison
print("\n--- TikTok Engagement (active channels only) ---")
for cat in ['family', 'adult']:
    sub = df_active[df_active['category'] == cat]
    if len(sub) > 0:
        print(f"\n{cat.upper()} ({len(sub)} channels):")
        print(f"  Avg plays/video: {sub['tiktok_avg_plays'].mean():,.0f}")
        print(f"  Avg likes/video: {sub['tiktok_avg_likes'].mean():,.0f}")
        print(f"  Avg comments/video: {sub['tiktok_avg_comments'].mean():,.0f}")

# 5c. Cross-platform correlation
if 'composite_index' in df_active.columns:
    valid = df_active.dropna(subset=['composite_index', 'tiktok_followers'])
    if len(valid) > 3:
        from scipy import stats
        r, p = stats.pearsonr(valid['composite_index'], np.log1p(valid['tiktok_followers']))
        print(f"\n--- Exploitation Index vs TikTok Followers ---")
        print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
        print(f"  {'SIGNIFICANT' if p < 0.05 else 'Not significant'}")

# ============================================================
# 6. Visualizations
# ============================================================
print("\nGenerating visualizations...")

# Fig 1: Cross-platform presence overview
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Cross-Platform Presence: YouTube vs TikTok', fontsize=16, fontweight='bold')

# 1a: YouTube vs TikTok followers (scatter)
ax = axes[0]
for cat, color, marker in [('family', '#e74c3c', 'o'), ('adult', '#3498db', 's')]:
    sub = df_active[df_active['category'] == cat]
    if len(sub) > 0 and 'n_videos' in sub.columns:
        ax.scatter(sub['n_videos'], sub['tiktok_followers']/1e6,
                  c=color, marker=marker, s=80, alpha=0.7, label=cat.title(), edgecolors='white')
        for _, row in sub.iterrows():
            if row['tiktok_followers'] > 5e6:
                ax.annotate(row['youtube_channel'], 
                          (row['n_videos'], row['tiktok_followers']/1e6),
                          fontsize=7, ha='left', va='bottom')

ax.set_xlabel('YouTube Videos (total)')
ax.set_ylabel('TikTok Followers (millions)')
ax.set_title('A. YouTube Activity vs TikTok Following')
ax.legend()

# 1b: Family vs Adult TikTok followers distribution
ax = axes[1]
family_followers = df_active[df_active['category']=='family']['tiktok_followers'] / 1e6
adult_followers = df_active[df_active['category']=='adult']['tiktok_followers'] / 1e6

positions = [1, 2]
bp = ax.boxplot([family_followers.dropna(), adult_followers.dropna()],
               positions=positions, widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('#3498db')
bp['boxes'][1].set_alpha(0.5)
ax.set_xticks(positions)
ax.set_xticklabels(['Family', 'Adult'])
ax.set_ylabel('TikTok Followers (millions)')
ax.set_title('B. TikTok Following by Category')

# 1c: Top channels on TikTok
ax = axes[2]
top_tiktok = df_active.nlargest(15, 'tiktok_followers')
colors = ['#e74c3c' if c == 'family' else '#3498db' for c in top_tiktok['category']]
bars = ax.barh(range(len(top_tiktok)), top_tiktok['tiktok_followers']/1e6, color=colors, alpha=0.8)
ax.set_yticks(range(len(top_tiktok)))
ax.set_yticklabels([f"@{r['tiktok_username']}" for _, r in top_tiktok.iterrows()], fontsize=8)
ax.set_xlabel('TikTok Followers (millions)')
ax.set_title('C. Top 15 Channels on TikTok')
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#e74c3c', alpha=0.8, label='Family'),
                   Patch(facecolor='#3498db', alpha=0.8, label='Adult')],
         loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_cross_platform_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_cross_platform_overview.png")

# Fig 2: TikTok engagement comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('TikTok Engagement: Family vs Adult Channels', fontsize=16, fontweight='bold')

metrics = [
    ('tiktok_avg_plays', 'Avg Plays per Video', 'A. Video Plays'),
    ('tiktok_avg_likes', 'Avg Likes per Video', 'B. Video Likes'),
    ('tiktok_avg_comments', 'Avg Comments per Video', 'C. Video Comments'),
]

for i, (col, ylabel, title) in enumerate(metrics):
    ax = axes[i]
    family_data = df_active[df_active['category']=='family'][col].dropna()
    adult_data = df_active[df_active['category']=='adult'][col].dropna()
    
    # Only plot if we have data
    data_to_plot = []
    labels = []
    if len(family_data) > 0:
        data_to_plot.append(family_data)
        labels.append('Family')
    if len(adult_data) > 0:
        data_to_plot.append(adult_data)
        labels.append('Adult')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        colors_box = ['#e74c3c', '#3498db'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Format y-axis with K/M
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'
    ))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_tiktok_engagement.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_tiktok_engagement.png")

# Fig 3: Exploitation Index vs TikTok Metrics
if 'composite_index' in df_active.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('YouTube Exploitation Index vs TikTok Presence', fontsize=16, fontweight='bold')
    
    tiktok_metrics = [
        ('tiktok_followers', 'TikTok Followers', 'A. Exploitation vs TikTok Following'),
        ('tiktok_videos', 'TikTok Video Count', 'B. Exploitation vs TikTok Activity'),
        ('tiktok_avg_plays', 'TikTok Avg Plays/Video', 'C. Exploitation vs TikTok Engagement'),
    ]
    
    for i, (col, ylabel, title) in enumerate(tiktok_metrics):
        ax = axes[i]
        valid = df_active.dropna(subset=['composite_index', col])
        family = valid[valid['category'] == 'family']
        adult = valid[valid['category'] == 'adult']
        
        if len(family) > 0:
            ax.scatter(family['composite_index'], family[col]/1e6 if family[col].max() > 1e6 else family[col],
                      c='#e74c3c', s=80, alpha=0.7, label='Family', edgecolors='white')
            for _, row in family.iterrows():
                val = row[col]/1e6 if valid[col].max() > 1e6 else row[col]
                ax.annotate(row['youtube_channel'], (row['composite_index'], val),
                          fontsize=6, ha='left', va='bottom')
        
        if len(adult) > 0:
            ax.scatter(adult['composite_index'], adult[col]/1e6 if adult[col].max() > 1e6 else adult[col],
                      c='#3498db', s=80, alpha=0.7, label='Adult', edgecolors='white')
        
        ax.set_xlabel('Composite Exploitation Index')
        suffix = ' (millions)' if valid[col].max() > 1e6 else ''
        ax.set_ylabel(f'{ylabel}{suffix}')
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_exploit_vs_tiktok.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_exploit_vs_tiktok.png")

# Fig 4: YouTube vs TikTok upload frequency comparison
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Cross-Platform Activity: YouTube Upload Frequency vs TikTok Videos', 
             fontsize=14, fontweight='bold')

for cat, color, marker in [('family', '#e74c3c', 'o'), ('adult', '#3498db', 's')]:
    sub = df_active[(df_active['category'] == cat) & (df_active['tiktok_videos'] > 0)]
    if len(sub) > 0 and 'videos_per_week' in sub.columns:
        ax.scatter(sub['videos_per_week'], sub['tiktok_videos'],
                  c=color, marker=marker, s=100, alpha=0.7, label=cat.title(), edgecolors='white')
        for _, row in sub.iterrows():
            ax.annotate(row['youtube_channel'], 
                      (row['videos_per_week'], row['tiktok_videos']),
                      fontsize=7, ha='left', va='bottom')

ax.set_xlabel('YouTube Upload Frequency (videos/week)')
ax.set_ylabel('TikTok Total Videos')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_cross_platform_activity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_cross_platform_activity.png")

# ============================================================
# 7. Save summary
# ============================================================
df_active.to_csv(os.path.join(TIKTOK_DIR, 'cross_platform_merged.csv'), index=False)

# Print key channel-level comparison
print("\n" + "="*60)
print("KEY CHANNEL COMPARISON (YouTube vs TikTok)")
print("="*60)

for _, row in df_active.sort_values('tiktok_followers', ascending=False).head(15).iterrows():
    yt_vids = row.get('n_videos', 0)
    tk_vids = row.get('tiktok_videos', 0)
    tk_followers = row.get('tiktok_followers', 0)
    cat = row['category']
    exploit = row.get('composite_index', 0)
    
    print(f"\n{'🔴' if cat=='family' else '🔵'} {row['youtube_channel']} (@{row.get('tiktok_username', '')})")
    print(f"  YouTube: {yt_vids:.0f} videos" if yt_vids else "  YouTube: N/A")
    print(f"  TikTok: {tk_followers:,.0f} followers, {tk_vids:.0f} videos, {row.get('tiktok_hearts',0):,.0f} hearts")
    if exploit:
        print(f"  Exploitation Index: {exploit:.3f}")

print("\nDone! All figures saved to", FIG_DIR)
