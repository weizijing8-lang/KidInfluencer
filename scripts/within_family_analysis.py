"""
Within-Family Exploitation Analysis
=====================================
NO MORE family vs adult comparisons.
This script focuses ENTIRELY on the 25 family channels:
- How do they differ from each other?
- What exploitation patterns exist?
- What clusters emerge?
- What drives the differences?

Adult channels are used ONLY as a silent baseline reference line.
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

plt.style.use('seaborn-v0_8-whitegrid')

BASE_DIR = '/home/ubuntu/KidInfluencer'
FIG_DIR = os.path.join(BASE_DIR, 'figures_within_family')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. Load and merge ALL data sources for family channels
# ============================================================
print("="*60)
print("WITHIN-FAMILY EXPLOITATION PROFILING")
print("="*60)

# Composite index (has content_exploit, labor, commercial, network)
df_comp = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/composite_exploitation_index.csv'))
df_fam = df_comp[df_comp['category'] == 'family'].copy()

# Upload frequency (has detailed labor metrics)
df_freq = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/upload_frequency_metrics.csv'))
df_freq_fam = df_freq[df_freq['category'] == 'family'][
    ['channel', 'n_videos', 'span_years', 'videos_per_week', 'mean_duration_min',
     'total_content_hours', 'monthly_content_hours', 'weekly_production_hours_est',
     'freq_change_pct']
]

# Merge
df = df_fam.merge(df_freq_fam, on='channel', how='left', suffixes=('', '_freq'))

# Collaboration network
df_collab = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/collaboration_network.csv'))
df_network = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/network_centrality.csv'))
df_network_fam = df_network[df_network['channel'].isin(df['channel'])]
df = df.merge(df_network_fam[['channel', 'degree', 'family_partners', 'adult_partners']],
              on='channel', how='left', suffixes=('', '_net'))

# Sponsorship
df_sponsor = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/sponsorship_by_channel.csv'))
df_sponsor_fam = df_sponsor[df_sponsor['channel'].isin(df['channel'])]
if len(df_sponsor_fam) > 0:
    df = df.merge(df_sponsor_fam[['channel', 'sponsor_rate', 'n_child_brands']].rename(
        columns={'sponsor_rate': 'sponsor_rate_detail', 'n_child_brands': 'n_child_brands_detail'}),
        on='channel', how='left')

# TikTok data
tiktok_data = []
for f in glob.glob(os.path.join(BASE_DIR, 'data/tiktok/*_tiktok.json')):
    with open(f) as fh:
        d = json.load(fh)
    if d['category'] == 'family':
        tiktok_data.append({
            'channel': d['youtube_channel'],
            'tiktok_followers': d['followers'],
            'tiktok_videos': d['video_count'],
            'tiktok_hearts': d['heart_count'],
        })
if tiktok_data:
    df_tiktok = pd.DataFrame(tiktok_data)
    df = df.merge(df_tiktok, on='channel', how='left')
    df['tiktok_followers'] = df['tiktok_followers'].fillna(0)
    df['tiktok_videos'] = df['tiktok_videos'].fillna(0)
    df['total_videos_cross_platform'] = df['n_videos'].fillna(0) + df['tiktok_videos']
else:
    df['tiktok_followers'] = 0
    df['tiktok_videos'] = 0
    df['total_videos_cross_platform'] = df['n_videos'].fillna(0)

# Comment analysis (partial)
try:
    df_comments = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/comment_classifications_partial.csv'))
    # Aggregate by channel - get inappropriate ratio
    if 'channel' in df_comments.columns:
        comment_agg = df_comments.groupby('channel').agg(
            n_comments=('classification', 'count'),
            n_inappropriate=('classification', lambda x: (x == 'inappropriate').sum()),
            n_timestamp=('classification', lambda x: (x == 'timestamp').sum()),
        ).reset_index()
        comment_agg['inappropriate_ratio'] = comment_agg['n_inappropriate'] / comment_agg['n_comments']
        comment_agg['timestamp_ratio'] = comment_agg['n_timestamp'] / comment_agg['n_comments']
        df = df.merge(comment_agg[['channel', 'inappropriate_ratio', 'timestamp_ratio']], 
                     on='channel', how='left')
except Exception as e:
    print(f"  Comment data not available: {e}")

print(f"\nFamily channels with full profiles: {len(df)}")
print(f"Columns available: {list(df.columns)}")

# ============================================================
# 2. Define exploitation dimensions (normalized 0-1)
# ============================================================
print("\n--- Building Exploitation Dimensions ---")

scaler = MinMaxScaler()

# Define dimensions with clear names
dimensions = {}

# D1: Content Exploitation (from embedding-based score)
if 'content_exploit' in df.columns:
    dimensions['Content\nExploitation'] = df['content_exploit'].fillna(0).values

# D2: Labor Intensity (upload frequency)
if 'videos_per_week' in df.columns:
    dimensions['Labor\nIntensity'] = df['videos_per_week'].fillna(0).values

# D3: Commercial Exploitation (sponsor rate + child brand targeting)
if 'sponsor_rate' in df.columns and 'n_child_brands' in df.columns:
    # Combine sponsor rate and child brand count
    sr = df['sponsor_rate'].fillna(0).values
    cb = df['n_child_brands'].fillna(0).values
    cb_norm = cb / (cb.max() + 1e-10)
    dimensions['Commercial\nExploitation'] = (sr + cb_norm) / 2

# D4: Network Exploitation (collaboration with other family channels)
if 'family_partners' in df.columns:
    dimensions['Network\nExploitation'] = df['family_partners'].fillna(0).values
elif 'family_partners_net' in df.columns:
    dimensions['Network\nExploitation'] = df['family_partners_net'].fillna(0).values

# D5: Cross-Platform Burden (total videos across YouTube + TikTok)
if 'total_videos_cross_platform' in df.columns:
    dimensions['Cross-Platform\nBurden'] = df['total_videos_cross_platform'].fillna(0).values

# D6: Frequency Escalation (are they producing MORE over time?)
if 'freq_change_pct' in df.columns:
    # Positive = increasing frequency = escalating
    dimensions['Frequency\nEscalation'] = df['freq_change_pct'].fillna(0).clip(lower=-100, upper=200).values

# Normalize all dimensions to 0-1
dim_names = list(dimensions.keys())
dim_matrix = np.column_stack([dimensions[d] for d in dim_names])
dim_normalized = scaler.fit_transform(dim_matrix)

df_dims = pd.DataFrame(dim_normalized, columns=dim_names, index=df['channel'].values)

print(f"Dimensions: {dim_names}")
print(f"Shape: {df_dims.shape}")

# Overall exploitation score (mean of all dimensions)
df['overall_exploit'] = dim_normalized.mean(axis=1)
df_dims['Overall'] = df['overall_exploit'].values

# ============================================================
# 3. Hierarchical Clustering
# ============================================================
print("\n--- Clustering Family Channels ---")

dist_matrix = pdist(dim_normalized, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')
df['cluster'] = clusters
df_dims['cluster'] = clusters

# Name clusters based on their characteristics
cluster_profiles = {}
for c in sorted(df['cluster'].unique()):
    mask = df['cluster'] == c
    channels = df[mask]['channel'].tolist()
    mean_dims = dim_normalized[mask].mean(axis=0)
    top_dim_idx = np.argmax(mean_dims)
    cluster_profiles[c] = {
        'channels': channels,
        'n': len(channels),
        'mean_dims': mean_dims,
        'dominant_dim': dim_names[top_dim_idx],
        'overall_mean': df[mask]['overall_exploit'].mean(),
    }
    print(f"\nCluster {c} ({len(channels)} channels):")
    print(f"  Dominant dimension: {dim_names[top_dim_idx]}")
    print(f"  Overall exploitation: {df[mask]['overall_exploit'].mean():.3f}")
    print(f"  Channels: {', '.join(channels)}")

# Assign cluster labels
cluster_labels = {}
# Sort clusters by overall exploitation
sorted_clusters = sorted(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]['overall_mean'])
label_names = ['Low-Risk Family Channels', 'Moderate Exploitation', 'High Exploitation']
for i, c in enumerate(sorted_clusters):
    cluster_labels[c] = label_names[min(i, 2)]

df['cluster_label'] = df['cluster'].map(cluster_labels)

# ============================================================
# 4. Print detailed profiles
# ============================================================
print("\n" + "="*60)
print("FAMILY CHANNEL PROFILES (ranked by overall exploitation)")
print("="*60)

df_sorted = df.sort_values('overall_exploit', ascending=False)

for _, row in df_sorted.iterrows():
    ch = row['channel']
    print(f"\n{'='*40}")
    print(f"  {ch.upper()}")
    print(f"  Cluster: {row['cluster_label']}")
    print(f"  Overall Exploitation Score: {row['overall_exploit']:.3f}")
    print(f"{'='*40}")
    
    # Dimension breakdown
    for dim in dim_names:
        val = df_dims.loc[ch, dim] if ch in df_dims.index else 0
        bar = '█' * int(val * 20) + '░' * (20 - int(val * 20))
        print(f"  {dim:25s} [{bar}] {val:.2f}")
    
    # Key facts
    n_vids = row.get('n_videos', 0)
    vpw = row.get('videos_per_week', 0)
    tk_vids = row.get('tiktok_videos', 0)
    tk_followers = row.get('tiktok_followers', 0)
    sr = row.get('sponsor_rate', 0)
    fp = row.get('family_partners', row.get('family_connections', 0))
    
    print(f"\n  YouTube: {n_vids:.0f} videos, {vpw:.1f}/week")
    if tk_vids > 0:
        print(f"  TikTok: {tk_vids:.0f} videos, {tk_followers:,.0f} followers")
        print(f"  Total cross-platform: {row.get('total_videos_cross_platform', n_vids):.0f} videos")
    print(f"  Sponsor rate: {sr*100:.1f}%")
    print(f"  Family channel partners: {fp:.0f}")

# Save profiles
df_sorted.to_csv(os.path.join(BASE_DIR, 'data/results_v4/within_family_profiles.csv'), index=False)
df_dims.to_csv(os.path.join(BASE_DIR, 'data/results_v4/within_family_dimensions.csv'))

# ============================================================
# 5. VISUALIZATIONS - ALL WITHIN-FAMILY
# ============================================================
print("\n\nGenerating within-family visualizations...")

# Color palette for clusters
cluster_colors = {}
for c in sorted(df['cluster'].unique()):
    label = cluster_labels[c]
    if 'High' in label:
        cluster_colors[c] = '#c0392b'
    elif 'Moderate' in label:
        cluster_colors[c] = '#e67e22'
    else:
        cluster_colors[c] = '#27ae60'

# ---- FIG 1: Channel Profiles Radar/Bar Chart ----
fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle('Family Channel Exploitation Profiles', fontsize=18, fontweight='bold', y=0.98)
ax.set_title('Each bar shows the overall exploitation score; color indicates cluster membership',
            fontsize=11, color='gray', style='italic', pad=10)

# Sort by overall score
order = df_sorted['channel'].values
y_pos = np.arange(len(order))
colors = [cluster_colors[df[df['channel']==ch]['cluster'].values[0]] for ch in order]

bars = ax.barh(y_pos, df_sorted['overall_exploit'].values, color=colors, alpha=0.85, height=0.7,
              edgecolor='white', linewidth=0.5)

# Add dimension breakdown as stacked mini-bars
for i, ch in enumerate(order):
    if ch in df_dims.index:
        vals = df_dims.loc[ch, dim_names].values
        # Small stacked indicator on the right
        x_start = df_sorted[df_sorted['channel']==ch]['overall_exploit'].values[0] + 0.01
        dim_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
        for j, (v, dc) in enumerate(zip(vals, dim_colors[:len(vals)])):
            ax.barh(i, v * 0.03, left=x_start + j * 0.035, color=dc, alpha=0.8, height=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(order, fontsize=10)
ax.set_xlabel('Overall Exploitation Score (0-1)', fontsize=12)
ax.invert_yaxis()

# Add cluster legend
legend_patches = [mpatches.Patch(color=cluster_colors[c], label=cluster_labels[c], alpha=0.85) 
                 for c in sorted(cluster_colors.keys())]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10, framealpha=0.9)

# Add dimension legend (small)
dim_colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
dim_legend = [mpatches.Patch(color=dim_colors_list[i], label=dim_names[i].replace('\n', ' '), alpha=0.8)
             for i in range(len(dim_names))]
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.legend(handles=dim_legend, loc='upper right', fontsize=7, title='Dimension Indicators',
          title_fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_channel_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_channel_profiles.png")

# ---- FIG 2: Multi-Dimension Heatmap (ONLY family channels) ----
fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle('Exploitation Dimensions Across Family Channels', fontsize=16, fontweight='bold')
ax.set_title('Normalized scores (0=lowest, 1=highest within family channels)', 
            fontsize=10, color='gray', style='italic')

# Sort by overall score
sorted_channels = df_sorted['channel'].values
heatmap_data = df_dims.loc[sorted_channels, dim_names].values

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(dim_names)))
ax.set_xticklabels([d.replace('\n', ' ') for d in dim_names], rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(sorted_channels)))
ax.set_yticklabels(sorted_channels, fontsize=9)

# Add cluster color indicators on the left
for i, ch in enumerate(sorted_channels):
    c = df[df['channel']==ch]['cluster'].values[0]
    ax.add_patch(plt.Rectangle((-0.7, i-0.4), 0.3, 0.8, 
                               color=cluster_colors[c], clip_on=False))

# Add values in cells
for i in range(len(sorted_channels)):
    for j in range(len(dim_names)):
        val = heatmap_data[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.colorbar(im, ax=ax, label='Normalized Score', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_dimension_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_dimension_heatmap.png")

# ---- FIG 3: Cluster Dendrogram ----
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Family Channel Clustering by Exploitation Pattern', fontsize=16, fontweight='bold')

# Color function for dendrogram
def color_func(k):
    return '#333333'

dend = dendrogram(Z, labels=df['channel'].values, ax=ax, leaf_rotation=45, leaf_font_size=9,
                  color_threshold=Z[-2, 2])  # Cut at 3 clusters

ax.set_ylabel('Distance (Ward linkage)', fontsize=11)
ax.set_title('Channels that cluster together share similar exploitation patterns', 
            fontsize=10, color='gray', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_dendrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_dendrogram.png")

# ---- FIG 4: Key Dimension Scatter Plots (within-family only) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Exploitation Dimensions: Within-Family Variation', fontsize=16, fontweight='bold')

# 4a: Labor vs Content
ax = axes[0, 0]
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['videos_per_week'], sub['content_exploit'], 
              c=cluster_colors[c], s=100, alpha=0.8, label=cluster_labels[c],
              edgecolors='white', linewidth=0.5)
for _, row in df.iterrows():
    ax.annotate(row['channel'], (row['videos_per_week'], row['content_exploit']),
               fontsize=7, ha='left', va='bottom')
ax.set_xlabel('Upload Frequency (videos/week)')
ax.set_ylabel('Content Exploitation Score')
ax.set_title('A. Labor Intensity vs Content Exploitation')
ax.legend(fontsize=8)

# 4b: Commercial vs Network
ax = axes[0, 1]
fp_col = 'family_partners' if 'family_partners' in df.columns else 'family_partners_net'
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['sponsor_rate'] * 100, sub[fp_col].fillna(0), 
              c=cluster_colors[c], s=100, alpha=0.8, label=cluster_labels[c],
              edgecolors='white', linewidth=0.5)
for _, row in df.iterrows():
    ax.annotate(row['channel'], (row['sponsor_rate']*100, row.get(fp_col, 0)),
               fontsize=7, ha='left', va='bottom')
ax.set_xlabel('Sponsorship Rate (%)')
ax.set_ylabel('Family Channel Partners')
ax.set_title('B. Commercial vs Network Exploitation')
ax.legend(fontsize=8)

# 4c: YouTube vs TikTok activity
ax = axes[1, 0]
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['n_videos'].fillna(0), sub['tiktok_videos'].fillna(0),
              c=cluster_colors[c], s=100, alpha=0.8, label=cluster_labels[c],
              edgecolors='white', linewidth=0.5)
for _, row in df.iterrows():
    if row.get('tiktok_videos', 0) > 100 or row.get('n_videos', 0) > 2000:
        ax.annotate(row['channel'], (row.get('n_videos', 0), row.get('tiktok_videos', 0)),
                   fontsize=7, ha='left', va='bottom')
ax.set_xlabel('YouTube Videos (total)')
ax.set_ylabel('TikTok Videos (total)')
ax.set_title('C. Cross-Platform Content Production')
# Add diagonal reference line
max_val = max(df['n_videos'].max(), df['tiktok_videos'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal production')
ax.legend(fontsize=8)

# 4d: Frequency change over time
ax = axes[1, 1]
for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['span_years'].fillna(0), sub['freq_change_pct'].fillna(0),
              c=cluster_colors[c], s=100, alpha=0.8, label=cluster_labels[c],
              edgecolors='white', linewidth=0.5)
for _, row in df.iterrows():
    if abs(row.get('freq_change_pct', 0)) > 30 or row.get('span_years', 0) > 12:
        ax.annotate(row['channel'], (row.get('span_years', 0), row.get('freq_change_pct', 0)),
                   fontsize=7, ha='left', va='bottom')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Channel Age (years)')
ax.set_ylabel('Upload Frequency Change (%)')
ax.set_title('D. Frequency Escalation Over Time')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_dimension_scatters.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_dimension_scatters.png")

# ---- FIG 5: Cluster Summary (radar chart per cluster) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
fig.suptitle('Exploitation Profiles by Cluster', fontsize=16, fontweight='bold', y=1.02)

angles = np.linspace(0, 2 * np.pi, len(dim_names), endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

for idx, c in enumerate(sorted(cluster_profiles.keys())):
    ax = axes[idx]
    profile = cluster_profiles[c]
    values = profile['mean_dims'].tolist()
    values += values[:1]
    
    ax.fill(angles, values, color=cluster_colors[c], alpha=0.25)
    ax.plot(angles, values, color=cluster_colors[c], linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('\n', ' ') for d in dim_names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f'{cluster_labels[c]}\n({profile["n"]} channels)', 
                fontsize=11, fontweight='bold', pad=20)
    
    # List channels
    channels_str = '\n'.join(profile['channels'][:8])
    if len(profile['channels']) > 8:
        channels_str += f'\n+{len(profile["channels"])-8} more'
    ax.text(0.5, -0.15, channels_str, transform=ax.transAxes, fontsize=7,
           ha='center', va='top', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_cluster_radars.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_cluster_radars.png")

# ---- FIG 6: The "Worst Offenders" Deep Dive ----
top5 = df_sorted.head(5)

fig, axes = plt.subplots(1, 5, figsize=(20, 8), subplot_kw=dict(polar=True))
fig.suptitle('Top 5 Highest-Exploitation Family Channels: Detailed Profiles', 
            fontsize=16, fontweight='bold', y=1.02)

for idx, (_, row) in enumerate(top5.iterrows()):
    ax = axes[idx]
    ch = row['channel']
    
    if ch in df_dims.index:
        values = df_dims.loc[ch, dim_names].values.tolist()
    else:
        values = [0] * len(dim_names)
    values += values[:1]
    
    c = row['cluster']
    ax.fill(angles, values, color=cluster_colors[c], alpha=0.3)
    ax.plot(angles, values, color=cluster_colors[c], linewidth=2, marker='o', markersize=4)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('\n', ' ') for d in dim_names], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title(f'{ch}\nScore: {row["overall_exploit"]:.3f}', fontsize=10, fontweight='bold', pad=15)
    
    # Key stats
    stats_text = f'YT: {row.get("n_videos",0):.0f} vids, {row.get("videos_per_week",0):.1f}/wk'
    if row.get('tiktok_videos', 0) > 0:
        stats_text += f'\nTT: {row["tiktok_videos"]:.0f} vids'
    stats_text += f'\nSponsor: {row.get("sponsor_rate",0)*100:.0f}%'
    ax.text(0.5, -0.12, stats_text, transform=ax.transAxes, fontsize=7,
           ha='center', va='top', color='#555')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_top5_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_top5_profiles.png")

# ============================================================
# 6. Print final summary
# ============================================================
print("\n" + "="*60)
print("FINAL WITHIN-FAMILY SUMMARY")
print("="*60)

for c in sorted(cluster_profiles.keys()):
    label = cluster_labels[c]
    profile = cluster_profiles[c]
    print(f"\n{label} ({profile['n']} channels):")
    print(f"  Channels: {', '.join(profile['channels'])}")
    print(f"  Dominant exploitation: {profile['dominant_dim']}")
    print(f"  Overall score: {profile['overall_mean']:.3f}")
    
    # What makes this cluster different
    mean_dims = profile['mean_dims']
    for i, (name, val) in enumerate(zip(dim_names, mean_dims)):
        print(f"    {name.replace(chr(10), ' '):25s}: {val:.3f}")

print("\n\nDone! All figures saved to", FIG_DIR)
