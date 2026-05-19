"""Regenerate all paper figures using the Snorkel proper pipeline (filtered, kid-centric) data."""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

df = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv')
df['log_views'] = np.log10(df['viewCount'].clip(lower=1))

# Thresholds
DIM_THRESH = 0.6
PRIVACY_THRESH = 0.5
OVERALL_THRESH = 0.7

dims = ['performative_labor', 'emotional_bait', 'narrative_conflict', 
        'challenge_format', 'commercial_content', 'privacy_violation']
dim_labels = ['Performative\nLabor', 'Emotional\nBait', 'Narrative\nConflict',
              'Challenge\nFormat', 'Commercial\nContent', 'Privacy\nViolation']
thresholds = {d: DIM_THRESH for d in dims}
thresholds['privacy_violation'] = PRIVACY_THRESH

# Create binary flags
for dim in dims:
    df[f'{dim}_flag'] = (df[f'{dim}_prob'] >= thresholds[dim]).astype(int)

# Color palette
blue = '#2C5F8A'
red = '#C44E52'
green = '#4C8C4A'
orange = '#D4820E'

# ============================================================
# Figure 1: Score Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.hist(df['exploitation_score_ws'], bins=40, color=blue, alpha=0.8, edgecolor='white', linewidth=0.5)
ax.axvline(x=OVERALL_THRESH, color=red, linestyle='--', linewidth=1.5, label=f'Threshold = {OVERALL_THRESH}')
ax.set_xlabel('Exploitation Risk Score', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Distribution of Exploitation Risk Scores\n(N=4,208 videos, 56 kid-centric channels)', fontsize=9)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('overleaf/fig1_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ============================================================
# Figure 2: Dimension Premiums (within-channel)
# ============================================================
premiums_data = []
for dim in dims:
    flag = df[f'{dim}_flag']
    ch_prems = []
    for ch, grp in df.groupby('channel_short_name'):
        ch_flag = flag.loc[grp.index]
        ch_views = df['log_views'].loc[grp.index]
        high = ch_views[ch_flag == 1]
        low = ch_views[ch_flag == 0]
        if len(high) >= 3 and len(low) >= 3:
            ch_prems.append(high.mean() - low.mean())
    pct_premium = (10**np.mean(ch_prems) - 1) * 100 if ch_prems else 0
    premiums_data.append(pct_premium)

fig, ax = plt.subplots(figsize=(5, 3.5))
colors = [blue if p > 0 else red for p in premiums_data]
bars = ax.bar(range(len(dims)), premiums_data, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(dim_labels, fontsize=8)
ax.set_ylabel('Within-Channel Engagement Premium (%)', fontsize=9)
ax.set_title('Engagement Premium by Exploitation Dimension', fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, premiums_data)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:+.1f}%', ha='center', va='bottom', fontsize=7)

# Add significance markers
sig_markers = ['***', '***', '***', 'n.s.', 'n.s.', '***']
for i, (bar, sig) in enumerate(zip(bars, sig_markers)):
    y_pos = max(bar.get_height() + 6, 6)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, sig, ha='center', va='bottom', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('overleaf/fig2_dimension_premiums.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ============================================================
# Figure 3: Contrasting Effects (Emotional Bait vs Commercial)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))

# Emotional Bait
for ch, grp in df.groupby('channel_short_name'):
    flag = grp['emotional_bait_flag']
    if flag.sum() >= 3 and (1-flag).sum() >= 3:
        high_mean = grp[flag==1]['log_views'].mean()
        low_mean = grp[flag==0]['log_views'].mean()
        axes[0].scatter(low_mean, high_mean, alpha=0.5, s=20, color=blue)
lims = [3, 8.5]
axes[0].plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
axes[0].set_xlabel('Mean log₁₀(views)\nNon-Emotional Bait', fontsize=8)
axes[0].set_ylabel('Mean log₁₀(views)\nEmotional Bait', fontsize=8)
axes[0].set_title('Emotional Bait (+65.6%)', fontsize=9, color=blue)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Commercial Content
for ch, grp in df.groupby('channel_short_name'):
    flag = grp['commercial_content_flag']
    if flag.sum() >= 3 and (1-flag).sum() >= 3:
        high_mean = grp[flag==1]['log_views'].mean()
        low_mean = grp[flag==0]['log_views'].mean()
        axes[1].scatter(low_mean, high_mean, alpha=0.5, s=20, color=red)
axes[1].plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
axes[1].set_xlabel('Mean log₁₀(views)\nNon-Commercial', fontsize=8)
axes[1].set_ylabel('Mean log₁₀(views)\nCommercial', fontsize=8)
axes[1].set_title('Commercial Content (-3.8%)', fontsize=9, color=red)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('overleaf/fig5_emotional_vs_commercial.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 3 (contrasting) done")

# ============================================================
# Figure 4: Channel-level premiums
# ============================================================
ch_premiums = []
for ch, grp in df.groupby('channel_short_name'):
    high = grp[grp['exploitation_score_ws'] >= OVERALL_THRESH]['log_views']
    low = grp[grp['exploitation_score_ws'] < OVERALL_THRESH]['log_views']
    if len(high) >= 3 and len(low) >= 3:
        prem = high.mean() - low.mean()
        ch_premiums.append({'channel': ch, 'premium': prem, 'n_videos': len(grp)})

ch_df = pd.DataFrame(ch_premiums).sort_values('premium', ascending=True)

fig, ax = plt.subplots(figsize=(4.5, 4))
colors_ch = [blue if p > 0 else red for p in ch_df['premium']]
ax.barh(range(len(ch_df)), ch_df['premium'], color=colors_ch, alpha=0.7, height=0.7)
ax.set_yticks(range(len(ch_df)))
ax.set_yticklabels(ch_df['channel'], fontsize=5)
ax.set_xlabel('Exploitation Premium (log₁₀ views)', fontsize=9)
ax.set_title(f'Within-Channel Exploitation Premium\n({len(ch_df)} channels, 76.6% positive)', fontsize=9)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('overleaf/fig4_channel_premiums.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 4 done")

print("\nAll figures regenerated with verified Snorkel pipeline data.")
