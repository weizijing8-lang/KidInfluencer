"""Generate Figure 6: Audience type as moderating variable for exploitation premium."""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGURES_DIR = 'analysis_discovery/paper_figures'

df = pd.read_csv('analysis_discovery/snorkel_results_v2/videos_with_exploitation_scores.csv')
df = df[df['viewCount'].notna() & (df['viewCount'] > 0)].copy()

# Audience classification
audience_map = {
    'likenastya': 'child', 'kidsdianashow': 'child', 'vladandniki': 'child',
    'ryansworld': 'child', 'aforadley': 'child', 'smellybellytv': 'child',
    'ethangamer': 'child', 'funsquadfamily': 'child',
    'brentrivera': 'teen_adult', 'piperrockelle': 'teen_adult',
    'rebeccazamolo': 'teen_adult', 'jordanmatter': 'teen_adult',
    'norrisnuts': 'teen_adult', 'acefamily': 'teen_adult',
    'royaltyfamily': 'teen_adult', 'bratayley': 'teen_adult',
    'gavinmagnus': 'teen_adult', 'wearethedavises': 'teen_adult',
    'tannerites': 'teen_adult', 'itsyeboi': 'teen_adult',
    'shotofyeagers': 'teen_adult', 'piersonwodzynski': 'teen_adult',
    'tydustalbott': 'teen_adult', 'everleighrose': 'teen_adult',
    'dailybumps': 'teen_adult', 'thesacconejolys': 'teen_adult',
    'ballingerfamily': 'teen_adult', 'inghamfamily': 'teen_adult',
    'jhousevlogs': 'teen_adult', 'familyfizz': 'teen_adult',
    'familyfunpack': 'teen_adult', 'familyfudge': 'teen_adult',
    'ohanaadventure': 'teen_adult', 'theweisslife': 'teen_adult',
    'theleray': 'teen_adult', 'jesssfam': 'teen_adult',
    'bonniehoellein': 'teen_adult', 'thatyoutub3family': 'teen_adult',
    'kkandbabyj': 'teen_adult', 'johnsonsfam': 'teen_adult',
    'onyxfamily': 'teen_adult', 'mccluretwins': 'teen_adult',
    'samandnia': 'teen_adult', 'ehbee': 'teen_adult',
    'jillianandaddie': 'teen_adult',
}

# Compute per-channel boost
channel_boosts = []
for ch, ch_data in df.groupby('channel_short_name'):
    if len(ch_data) < 10:
        continue
    q75 = ch_data['exploitation_score'].quantile(0.75)
    q25 = ch_data['exploitation_score'].quantile(0.25)
    high = ch_data[ch_data['exploitation_score'] > q75]
    low = ch_data[ch_data['exploitation_score'] < q25]
    if len(high) < 3 or len(low) < 3:
        continue
    high_median = high['viewCount'].median()
    low_median = low['viewCount'].median()
    boost = (high_median / low_median - 1) * 100
    audience = audience_map.get(ch, 'unknown')
    channel_boosts.append({
        'channel': ch,
        'audience': audience,
        'boost_pct': boost,
    })

boost_df = pd.DataFrame(channel_boosts)
boost_df = boost_df[boost_df['audience'] != 'unknown']

child_boosts = boost_df[boost_df['audience'] == 'child']['boost_pct']
teen_boosts = boost_df[boost_df['audience'] == 'teen_adult']['boost_pct']

# ============================================================
# FIGURE 6: Two-panel figure
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel A: Box/strip plot comparison
ax = axes[0]
colors = {'child': '#3498db', 'teen_adult': '#e74c3c'}
positions = [0, 1]

# Strip plot with jitter
np.random.seed(42)
for i, (label, data) in enumerate([('child', child_boosts), ('teen_adult', teen_boosts)]):
    jitter = np.random.uniform(-0.15, 0.15, len(data))
    ax.scatter([i + j for j in jitter], data, color=colors[label], alpha=0.6, s=60, edgecolor='white', zorder=3)
    # Box plot overlay
    bp = ax.boxplot([data], positions=[i], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor=colors[label], alpha=0.3),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='gray'),
                    capprops=dict(color='gray'),
                    flierprops=dict(marker='', markersize=0))

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Child Audience\n(n=8 channels)', 'Teen/Adult Audience\n(n=37 channels)'])
ax.set_ylabel('Within-Channel Exploitation Premium (%)')
ax.set_title('(a) Exploitation Premium by Target Audience')

# Add significance annotation
u_stat, u_pval = stats.mannwhitneyu(teen_boosts, child_boosts, alternative='greater')
y_max = max(boost_df['boost_pct'].max(), 500)
ax.plot([0, 0, 1, 1], [y_max*0.85, y_max*0.88, y_max*0.88, y_max*0.85], 'k-', linewidth=1)
sig_text = f'Mann-Whitney p={u_pval:.4f}**'
ax.text(0.5, y_max*0.90, sig_text, ha='center', fontsize=10, fontweight='bold')

# Add group means
ax.scatter([0], [child_boosts.mean()], marker='D', color='navy', s=100, zorder=5, label=f'Mean: {child_boosts.mean():+.0f}%')
ax.scatter([1], [teen_boosts.mean()], marker='D', color='darkred', s=100, zorder=5, label=f'Mean: {teen_boosts.mean():+.0f}%')
ax.legend(loc='upper left', frameon=True)

# Panel B: Individual channels sorted by boost, colored by audience
ax = axes[1]
sorted_df = boost_df.sort_values('boost_pct', ascending=True)
bar_colors = [colors[a] for a in sorted_df['audience']]
y_pos = range(len(sorted_df))

ax.barh(y_pos, sorted_df['boost_pct'], color=bar_colors, alpha=0.7, edgecolor='white', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_df['channel'], fontsize=7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Within-Channel Exploitation Premium (%)')
ax.set_title('(b) Per-Channel Exploitation Premium')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['child'], alpha=0.7, label='Child Audience'),
    Patch(facecolor=colors['teen_adult'], alpha=0.7, label='Teen/Adult Audience'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig6_audience_moderation.png')
plt.savefig(f'{FIGURES_DIR}/fig6_audience_moderation.pdf')
plt.close()
print(f"Saved fig6_audience_moderation to {FIGURES_DIR}/")

# Save stats for paper
import json
moderation_stats = {
    'child_channels_n': len(child_boosts),
    'teen_adult_channels_n': len(teen_boosts),
    'child_mean_boost': float(child_boosts.mean()),
    'child_median_boost': float(child_boosts.median()),
    'child_pct_positive': float((child_boosts > 0).mean() * 100),
    'teen_adult_mean_boost': float(teen_boosts.mean()),
    'teen_adult_median_boost': float(teen_boosts.median()),
    'teen_adult_pct_positive': float((teen_boosts > 0).mean() * 100),
    'mann_whitney_u': float(u_stat),
    'mann_whitney_p': float(u_pval),
    'cohens_d': float((teen_boosts.mean() - child_boosts.mean()) / 
                      np.sqrt(((len(teen_boosts)-1)*teen_boosts.std()**2 + 
                               (len(child_boosts)-1)*child_boosts.std()**2) /
                              (len(teen_boosts) + len(child_boosts) - 2))),
}
with open(f'{FIGURES_DIR}/audience_moderation_stats.json', 'w') as f:
    json.dump(moderation_stats, f, indent=2)
print("Saved audience_moderation_stats.json")
print(f"\nKey result: Teen/Adult median boost = {teen_boosts.median():+.1f}% vs Child median = {child_boosts.median():+.1f}%")
print(f"Mann-Whitney p = {u_pval:.4f}")
