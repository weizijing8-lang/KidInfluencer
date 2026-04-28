"""
Paper 1: Structural Exploitation of Child Influencers
=====================================================
RQ: How much do child influencers actually "work"? How does this compare 
    to legal standards for child performers?

Core metrics (all objective, no NLP):
1. Upload frequency (videos/week)
2. Video duration → total content hours
3. Cross-platform burden (YouTube + TikTok total output)
4. Commercial exploitation (sponsor rate, child brand targeting)
5. Temporal trends (is it getting worse?)
6. Comparison to child labor law benchmarks
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from datetime import datetime

BASE_DIR = '/home/ubuntu/KidInfluencer'
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
RESULTS_DIR = os.path.join(BASE_DIR, 'data/paper1_structural')
FIG_DIR = os.path.join(BASE_DIR, 'figures_paper1')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("="*60)
print("PAPER 1: STRUCTURAL EXPLOITATION OF CHILD INFLUENCERS")
print("="*60)

# ============================================================
# 1. Load all data sources
# ============================================================
print("\n--- Loading data ---")

# Within-family profiles
df_profiles = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/within_family_profiles.csv'))
# Filter out channels with very few videos (< 50)
df_profiles = df_profiles[df_profiles['n_videos'] >= 50].copy()
print(f"Family channels with >=50 videos: {len(df_profiles)}")

# Full video-level data
df_all = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/full_results_v4.csv'))
df_all = df_all.rename(columns={'channel_short_name': 'channel', 'channel_category': 'category'})
df_fam = df_all[df_all['category'] == 'family'].copy()
df_adult = df_all[df_all['category'] == 'adult'].copy()
print(f"Family videos: {len(df_fam)}, Adult videos: {len(df_adult)}")

# Duration data
dur_file = os.path.join(BASE_DIR, 'data/durations/duration_summary.csv')
if os.path.exists(dur_file):
    df_dur = pd.read_csv(dur_file)
    print(f"Duration data: {len(df_dur)} channels")
else:
    df_dur = pd.DataFrame()
    print("No separate duration file, using profiles data")

# TikTok data
tiktok_file = os.path.join(BASE_DIR, 'data/tiktok/tiktok_summary.csv')
if os.path.exists(tiktok_file):
    df_tiktok = pd.read_csv(tiktok_file)
    print(f"TikTok data: {len(df_tiktok)} channels")
else:
    df_tiktok = pd.DataFrame()

# Commercial data
comm_file = os.path.join(BASE_DIR, 'data/results_v4/commercial_summary.csv')
if os.path.exists(comm_file):
    df_comm = pd.read_csv(comm_file)
    print(f"Commercial data: {len(df_comm)} channels")
else:
    df_comm = pd.DataFrame()

# ============================================================
# 2. CHILD LABOR LAW BENCHMARKS
# ============================================================
print("\n" + "="*60)
print("CHILD LABOR LAW BENCHMARKS")
print("="*60)

# Source: California Labor Code, SAG-AFTRA regulations
# These are the strictest in the US (Hollywood standards)
benchmarks = {
    'CA_infant_0_6mo': {'max_hours_day': 0.33, 'max_hours_week': 2, 'description': 'California: Infant (0-6 months)'},
    'CA_baby_6mo_2yr': {'max_hours_day': 2, 'max_hours_week': 10, 'description': 'California: Baby (6mo-2yr)'},
    'CA_child_2_6yr': {'max_hours_day': 3, 'max_hours_week': 15, 'description': 'California: Child (2-6yr)'},
    'CA_child_6_9yr': {'max_hours_day': 4, 'max_hours_week': 20, 'description': 'California: Child (6-9yr)'},
    'CA_child_9_16yr': {'max_hours_day': 5, 'max_hours_week': 25, 'description': 'California: Child (9-16yr)'},
    'EU_child_under15': {'max_hours_day': 2, 'max_hours_week': 12, 'description': 'EU Directive 94/33/EC: Under 15'},
    'France_kidfluencer': {'max_hours_day': 0, 'max_hours_week': 0, 'description': 'France 2020 Law: Kidfluencer (income to trust)'},
}

for k, v in benchmarks.items():
    print(f"  {v['description']}: max {v['max_hours_week']} hrs/week")

# ============================================================
# 3. ESTIMATE ACTUAL LABOR HOURS
# ============================================================
print("\n" + "="*60)
print("ESTIMATED LABOR HOURS PER CHANNEL")
print("="*60)

# Conservative estimate: production time = 3x video duration (filming + setup + retakes)
# For vlog-style content, this is actually very conservative
# Real estimate from industry: 5-10x for edited content
PRODUCTION_MULTIPLIER = 3  # conservative

labor_data = []
for _, row in df_profiles.iterrows():
    ch = row['channel']
    vpw = row['videos_per_week']
    dur_min = row['mean_duration_min']
    
    # If duration is 0 (API quota exceeded), estimate from similar channels
    if dur_min == 0:
        # Use median of channels with known duration
        known_dur = df_profiles[df_profiles['mean_duration_min'] > 0]['mean_duration_min']
        dur_min = known_dur.median()
        dur_estimated = True
    else:
        dur_estimated = False
    
    # Screen time per week (just the video itself)
    screen_hours_week = vpw * dur_min / 60
    
    # Estimated production time (conservative 3x multiplier)
    production_hours_week = screen_hours_week * PRODUCTION_MULTIPLIER
    
    # Total content hours over channel lifetime
    total_screen_hours = row['n_videos'] * dur_min / 60
    total_production_hours = total_screen_hours * PRODUCTION_MULTIPLIER
    
    # Cross-platform: add TikTok (assume 1 min avg per TikTok, 2x production)
    tiktok_vids = row.get('tiktok_videos', 0)
    if pd.isna(tiktok_vids):
        tiktok_vids = 0
    tiktok_screen_hours = tiktok_vids * 1 / 60  # 1 min avg
    tiktok_production_hours = tiktok_screen_hours * 2  # simpler production
    
    # Total cross-platform
    total_cross_platform_production = production_hours_week + (tiktok_production_hours / (row['span_years'] * 52) if row['span_years'] > 0 else 0)
    
    # Compare to benchmarks
    # Assume child age range: most start around 3-5, currently 8-16
    # Use CA 6-9yr benchmark (4 hrs/day, 20 hrs/week) as reference
    exceeds_ca_child = production_hours_week > 20
    exceeds_eu_child = production_hours_week > 12
    
    labor_data.append({
        'channel': ch,
        'videos_per_week': vpw,
        'mean_duration_min': dur_min,
        'duration_estimated': dur_estimated,
        'screen_hours_week': screen_hours_week,
        'production_hours_week': production_hours_week,
        'total_screen_hours_lifetime': total_screen_hours,
        'total_production_hours_lifetime': total_production_hours,
        'tiktok_videos': tiktok_vids,
        'tiktok_production_hours_lifetime': tiktok_production_hours,
        'cross_platform_production_hours_week': total_cross_platform_production,
        'span_years': row['span_years'],
        'n_videos_youtube': row['n_videos'],
        'total_videos_all_platforms': row['total_videos_cross_platform'],
        'sponsor_rate': row['sponsor_rate'],
        'n_child_brands': row['n_child_brands'],
        'exceeds_CA_20hr': exceeds_ca_child,
        'exceeds_EU_12hr': exceeds_eu_child,
        'freq_change_pct': row['freq_change_pct'],
    })

df_labor = pd.DataFrame(labor_data).sort_values('production_hours_week', ascending=False)

print("\nTop 15 channels by estimated production hours/week:")
print(df_labor[['channel', 'videos_per_week', 'mean_duration_min', 'screen_hours_week', 
                'production_hours_week', 'exceeds_CA_20hr', 'exceeds_EU_12hr']].head(15).to_string(index=False))

print(f"\nChannels exceeding CA child labor limit (20 hrs/week): {df_labor['exceeds_CA_20hr'].sum()}/{len(df_labor)}")
print(f"Channels exceeding EU child labor limit (12 hrs/week): {df_labor['exceeds_EU_12hr'].sum()}/{len(df_labor)}")

df_labor.to_csv(os.path.join(RESULTS_DIR, 'labor_hours_analysis.csv'), index=False)

# ============================================================
# 4. LIFETIME LABOR BURDEN
# ============================================================
print("\n" + "="*60)
print("LIFETIME LABOR BURDEN")
print("="*60)

print("\nTop 10 by total production hours (lifetime):")
df_lifetime = df_labor.sort_values('total_production_hours_lifetime', ascending=False)
for _, row in df_lifetime.head(10).iterrows():
    yt = row['total_production_hours_lifetime']
    tt = row['tiktok_production_hours_lifetime']
    total = yt + tt
    print(f"  {row['channel']:25s}: {total:8.0f} hrs total ({yt:.0f} YT + {tt:.0f} TT) over {row['span_years']:.1f} years = {total/row['span_years']:.0f} hrs/year")

# ============================================================
# 5. COMMERCIAL EXPLOITATION
# ============================================================
print("\n" + "="*60)
print("COMMERCIAL EXPLOITATION")
print("="*60)

print("\nSponsor rate and child brand targeting:")
df_comm_sorted = df_labor.sort_values('sponsor_rate', ascending=False)
for _, row in df_comm_sorted.head(10).iterrows():
    print(f"  {row['channel']:25s}: {row['sponsor_rate']*100:.1f}% sponsored, {row['n_child_brands']:.0f} child brand mentions")

# Correlation: sponsor rate ↔ upload frequency
r_sp, p_sp = stats.pearsonr(df_labor['sponsor_rate'], df_labor['videos_per_week'])
print(f"\nSponsor rate ↔ Upload frequency: r={r_sp:.4f}, p={p_sp:.4f}")

# ============================================================
# 6. TEMPORAL TRENDS
# ============================================================
print("\n" + "="*60)
print("TEMPORAL TRENDS: IS IT GETTING WORSE?")
print("="*60)

# Parse dates and compute quarterly upload rates
df_fam['published_date'] = pd.to_datetime(df_fam['publishedAt'], errors='coerce')
df_fam['year'] = df_fam['published_date'].dt.year
df_fam['quarter'] = df_fam['published_date'].dt.to_period('Q')

# Only channels with data
fam_channels = df_labor['channel'].tolist()
df_fam_valid = df_fam[df_fam['channel'].isin(fam_channels)].copy()

# Quarterly video count per channel
quarterly = df_fam_valid.groupby(['channel', 'quarter']).size().reset_index(name='n_videos')
quarterly['quarter_str'] = quarterly['quarter'].astype(str)

# Overall trend: total family channel videos per quarter
overall_quarterly = df_fam_valid.groupby('quarter').size().reset_index(name='total_videos')
overall_quarterly['quarter_str'] = overall_quarterly['quarter'].astype(str)
overall_quarterly['year'] = overall_quarterly['quarter'].apply(lambda x: x.start_time.year)
overall_quarterly = overall_quarterly[overall_quarterly['year'] >= 2015]

print(f"Quarterly video production (all family channels):")
for _, row in overall_quarterly.tail(10).iterrows():
    print(f"  {row['quarter_str']}: {row['total_videos']} videos")

# Trend test
if len(overall_quarterly) > 4:
    x = np.arange(len(overall_quarterly))
    y = overall_quarterly['total_videos'].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    print(f"\nLinear trend: slope={slope:.2f} videos/quarter, r={r:.4f}, p={p:.4f}")

# ============================================================
# 7. FIGURES
# ============================================================
print("\n--- Generating figures ---")
plt.style.use('seaborn-v0_8-whitegrid')

# --- Fig 1: Labor hours vs child labor law benchmarks ---
fig, ax = plt.subplots(figsize=(14, 8))

df_plot = df_labor.sort_values('production_hours_week', ascending=True)
y_pos = range(len(df_plot))

# Color by severity
colors = []
for _, row in df_plot.iterrows():
    if row['production_hours_week'] > 20:
        colors.append('#e74c3c')  # exceeds CA limit
    elif row['production_hours_week'] > 12:
        colors.append('#f39c12')  # exceeds EU limit
    else:
        colors.append('#2ecc71')  # within limits

bars = ax.barh(y_pos, df_plot['production_hours_week'], color=colors, height=0.7, alpha=0.8)

# Add benchmark lines
ax.axvline(x=12, color='#3498db', linestyle='--', linewidth=2, label='EU limit (12 hrs/week)')
ax.axvline(x=20, color='#e74c3c', linestyle='--', linewidth=2, label='CA limit: age 6-9 (20 hrs/week)')
ax.axvline(x=25, color='#c0392b', linestyle=':', linewidth=2, label='CA limit: age 9-16 (25 hrs/week)')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['channel'], fontsize=8)
ax.set_xlabel('Estimated Production Hours per Week\n(Conservative: 3x video duration)', fontsize=11)
ax.set_title('Child Influencer Labor Hours vs. Child Performer Legal Limits\n(Within Family Channels Only)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Annotation
n_exceed_eu = df_labor['exceeds_EU_12hr'].sum()
n_exceed_ca = df_labor['exceeds_CA_20hr'].sum()
ax.annotate(f'{n_exceed_eu}/{len(df_labor)} exceed EU limit\n{n_exceed_ca}/{len(df_labor)} exceed CA limit',
           xy=(0.98, 0.05), xycoords='axes fraction', ha='right', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_labor_vs_law.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig1_labor_vs_law.png")

# --- Fig 2: Lifetime labor burden ---
fig, ax = plt.subplots(figsize=(14, 8))

df_life = df_labor.sort_values('total_production_hours_lifetime', ascending=True)
y_pos = range(len(df_life))

# Stacked: YouTube + TikTok
yt_hours = df_life['total_production_hours_lifetime'].values
tt_hours = df_life['tiktok_production_hours_lifetime'].values

ax.barh(y_pos, yt_hours, color='#e74c3c', alpha=0.8, label='YouTube production hours')
ax.barh(y_pos, tt_hours, left=yt_hours, color='#3498db', alpha=0.8, label='TikTok production hours')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_life['channel'], fontsize=8)
ax.set_xlabel('Estimated Total Production Hours (Lifetime)', fontsize=11)
ax.set_title('Lifetime Labor Burden: YouTube + TikTok\n(Conservative estimate: 3x YouTube duration, 2x TikTok duration)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)

# Add year labels
for i, (_, row) in enumerate(df_life.iterrows()):
    total = row['total_production_hours_lifetime'] + row['tiktok_production_hours_lifetime']
    if total > 100:
        ax.text(total + 20, i, f"{row['span_years']:.0f} yrs", va='center', fontsize=7, color='#555555')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_lifetime_burden.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig2_lifetime_burden.png")

# --- Fig 3: Cross-platform total output ---
fig, ax = plt.subplots(figsize=(12, 8))

df_cross = df_labor[df_labor['tiktok_videos'] > 0].sort_values('total_videos_all_platforms', ascending=True)
if len(df_cross) > 0:
    y_pos = range(len(df_cross))
    yt_vids = df_cross['n_videos_youtube'].values
    tt_vids = df_cross['tiktok_videos'].values
    
    ax.barh(y_pos, yt_vids, color='#e74c3c', alpha=0.8, label='YouTube videos')
    ax.barh(y_pos, tt_vids, left=yt_vids, color='#3498db', alpha=0.8, label='TikTok videos')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_cross['channel'], fontsize=9)
    ax.set_xlabel('Total Videos Produced', fontsize=11)
    ax.set_title('Cross-Platform Content Production\n(Family Channels Active on Both YouTube and TikTok)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    
    for i, (_, row) in enumerate(df_cross.iterrows()):
        total = row['n_videos_youtube'] + row['tiktok_videos']
        ax.text(total + 30, i, f"{total:.0f} total", va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_cross_platform.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig3_cross_platform.png")

# --- Fig 4: Commercial exploitation (sponsor rate × child brands) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: sponsor rate ranking
ax = axes[0]
df_sp = df_labor.sort_values('sponsor_rate', ascending=True)
y_pos = range(len(df_sp))
colors_sp = ['#e74c3c' if r > 0.15 else '#f39c12' if r > 0.05 else '#2ecc71' for r in df_sp['sponsor_rate']]
ax.barh(y_pos, df_sp['sponsor_rate'] * 100, color=colors_sp, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_sp['channel'], fontsize=7)
ax.set_xlabel('Sponsor Rate (%)', fontsize=10)
ax.set_title('Percentage of Sponsored Videos', fontsize=11, fontweight='bold')

# Right: child brand mentions
ax = axes[1]
df_cb = df_labor[df_labor['n_child_brands'] > 0].sort_values('n_child_brands', ascending=True)
if len(df_cb) > 0:
    y_pos = range(len(df_cb))
    ax.barh(y_pos, df_cb['n_child_brands'], color='#9b59b6', height=0.7, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_cb['channel'], fontsize=7)
    ax.set_xlabel('Child Brand Mentions', fontsize=10)
    ax.set_title('Child-Targeted Brand Mentions\nin Video Descriptions', fontsize=11, fontweight='bold')

plt.suptitle('Commercial Exploitation of Child Influencers', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_commercial.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig4_commercial.png")

# --- Fig 5: Temporal trend ---
fig, ax = plt.subplots(figsize=(12, 6))

if len(overall_quarterly) > 4:
    x_dates = overall_quarterly['quarter'].apply(lambda x: x.start_time)
    y_vals = overall_quarterly['total_videos'].values
    ax.plot(x_dates, y_vals, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    ax.fill_between(x_dates, y_vals, alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Total Videos Published (All Family Channels)', fontsize=11)
    ax.set_title('Quarterly Video Production Trend\n(Family Channels, 2015-2026)', fontsize=13, fontweight='bold')
    
    # Add trend line
    x_num = np.arange(len(x_dates))
    z = np.polyfit(x_num, y_vals, 1)
    p = np.poly1d(z)
    ax.plot(x_dates, p(x_num), '--', color='black', alpha=0.5, linewidth=1.5, label=f'Trend: {z[0]:+.1f} videos/quarter')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_temporal_trend.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig5_temporal_trend.png")

# --- Fig 6: Multi-dimensional structural exploitation profile ---
fig, ax = plt.subplots(figsize=(14, 10))

# Normalize each dimension to 0-1 for heatmap
dims = ['production_hours_week', 'sponsor_rate', 'n_child_brands', 'tiktok_videos', 'freq_change_pct']
dim_labels = ['Production\nHours/Week', 'Sponsor\nRate', 'Child Brand\nMentions', 'TikTok\nVideos', 'Frequency\nChange %']

df_hm = df_labor.set_index('channel')[dims].copy()
# Normalize
for col in dims:
    mn, mx = df_hm[col].min(), df_hm[col].max()
    if mx > mn:
        df_hm[col] = (df_hm[col] - mn) / (mx - mn)
    else:
        df_hm[col] = 0

# Sort by mean across dimensions
df_hm['mean_score'] = df_hm.mean(axis=1)
df_hm = df_hm.sort_values('mean_score', ascending=True)
df_hm = df_hm.drop('mean_score', axis=1)

im = ax.imshow(df_hm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(df_hm)))
ax.set_yticklabels(df_hm.index, fontsize=8)
ax.set_xticks(range(len(dim_labels)))
ax.set_xticklabels(dim_labels, fontsize=9, ha='center')
ax.set_title('Structural Exploitation Profile\n(Each dimension normalized 0-1, darker = more intense)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Intensity', shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_structural_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig6_structural_heatmap.png")

# ============================================================
# 8. Summary statistics for paper
# ============================================================
print("\n" + "="*60)
print("PAPER 1 SUMMARY STATISTICS")
print("="*60)

print(f"\nSample: {len(df_labor)} family channels, {df_labor['n_videos_youtube'].sum():.0f} YouTube videos")
print(f"Time span: {df_labor['span_years'].min():.1f} - {df_labor['span_years'].max():.1f} years")

print(f"\nLabor intensity:")
print(f"  Mean videos/week: {df_labor['videos_per_week'].mean():.2f} (median: {df_labor['videos_per_week'].median():.2f})")
print(f"  Mean production hrs/week: {df_labor['production_hours_week'].mean():.1f} (median: {df_labor['production_hours_week'].median():.1f})")
print(f"  Max production hrs/week: {df_labor['production_hours_week'].max():.1f} ({df_labor.iloc[0]['channel']})")
print(f"  Channels exceeding EU 12hr limit: {df_labor['exceeds_EU_12hr'].sum()}/{len(df_labor)} ({df_labor['exceeds_EU_12hr'].mean()*100:.0f}%)")
print(f"  Channels exceeding CA 20hr limit: {df_labor['exceeds_CA_20hr'].sum()}/{len(df_labor)} ({df_labor['exceeds_CA_20hr'].mean()*100:.0f}%)")

print(f"\nLifetime burden:")
total_all = (df_labor['total_production_hours_lifetime'] + df_labor['tiktok_production_hours_lifetime']).sum()
print(f"  Total production hours (all channels): {total_all:,.0f}")
print(f"  Equivalent full-time years (2000 hrs/yr): {total_all/2000:.1f}")

print(f"\nCross-platform:")
cross = df_labor[df_labor['tiktok_videos'] > 0]
print(f"  Channels on TikTok: {len(cross)}/{len(df_labor)}")
if len(cross) > 0:
    print(f"  Mean TikTok videos: {cross['tiktok_videos'].mean():.0f}")
    print(f"  Mean total videos (YT+TT): {cross['total_videos_all_platforms'].mean():.0f}")

print(f"\nCommercial exploitation:")
print(f"  Mean sponsor rate: {df_labor['sponsor_rate'].mean()*100:.1f}%")
print(f"  Max sponsor rate: {df_labor['sponsor_rate'].max()*100:.1f}% ({df_labor.sort_values('sponsor_rate', ascending=False).iloc[0]['channel']})")
print(f"  Mean child brand mentions: {df_labor['n_child_brands'].mean():.0f}")

print("\nDone!")
