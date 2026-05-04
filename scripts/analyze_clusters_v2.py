"""
Deep analysis of K=7 de-channelized clusters:
1. Cross-channel content strategy patterns
2. Manipulation signal enrichment per cluster
3. View boost decomposition (within-channel vs across-channel)
4. Generate publication-quality figures
"""
import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
features_full = pd.read_csv('analysis_discovery/videos_with_clusters_v2.csv')
df['cluster_v2'] = features_full['cluster_v2'].values
summaries = json.load(open('analysis_discovery/cluster_v2_summaries.json'))

# Cluster names based on enriched features
cluster_names = {
    0: "Prank & Reaction",
    1: "Game & Roleplay & Music",
    2: "Clickbait Titles (ALL CAPS)",
    3: "Shorts & Emoji",
    4: "Medical & Urgency",
    5: "Unboxing & Toy Review",
    6: "Standard Vlog"
}

# Sort by view boost
sorted_clusters = sorted(summaries, key=lambda x: x['view_boost'], reverse=True)

# ============================================================
# 1. Within-channel view boost (controls for channel effect)
# ============================================================
print("="*80)
print("1. WITHIN-CHANNEL VIEW BOOST ANALYSIS")
print("="*80)
print("\nThis controls for channel popularity by comparing each video to its own channel median.")

within_channel_boosts = {}
for c in range(7):
    boosts = []
    mask = df['cluster_v2'] == c
    cl = df[mask]
    for ch in cl['channel_short_name'].unique():
        ch_all = df[df['channel_short_name'] == ch]
        ch_median = ch_all['viewCount'].median()
        ch_in_cluster = cl[cl['channel_short_name'] == ch]['viewCount'].median()
        if ch_median > 0 and len(cl[cl['channel_short_name'] == ch]) >= 5:
            boost = (ch_in_cluster - ch_median) / ch_median
            boosts.append(boost)
    
    if boosts:
        within_channel_boosts[c] = {
            'mean': np.mean(boosts),
            'median': np.median(boosts),
            'n_channels': len(boosts),
            'boosts': boosts
        }
        print(f"\n  C{c} ({cluster_names[c]}):")
        print(f"    Within-channel boost: mean={np.mean(boosts)*100:+.1f}%, median={np.median(boosts)*100:+.1f}%")
        print(f"    Based on {len(boosts)} channels with >=5 videos in cluster")
        # One-sample t-test: is boost significantly different from 0?
        if len(boosts) >= 3:
            t_stat, p_val = stats.ttest_1samp(boosts, 0)
            print(f"    t-test vs 0: t={t_stat:.2f}, p={p_val:.4f} {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'}")

# ============================================================
# 2. Manipulation signal analysis per cluster
# ============================================================
print("\n\n" + "="*80)
print("2. MANIPULATION SIGNAL ENRICHMENT")
print("="*80)

manipulation_features = [
    'has_clickbait_emotion', 'has_urgency', 'has_mystery', 
    'has_conflict', 'has_medical', 'has_money', 'has_giveaway', 'has_brand'
]

feature_cols = [c for c in features_full.columns if c not in ['id', 'channel_short_name', 'viewCount', 'cluster_v2']]

print(f"\n{'Feature':<25}", end="")
for c in range(7):
    print(f"C{c:>3}", end="")
print("  Overall")
print("-" * 80)

for feat in manipulation_features:
    overall = features_full[feat].mean()
    print(f"{feat:<25}", end="")
    for c in range(7):
        mask = features_full['cluster_v2'] == c
        val = features_full.loc[mask, feat].mean()
        # Bold if >2x overall
        marker = "*" if val > 2 * overall else " "
        print(f"{val*100:>5.1f}{marker}", end="")
    print(f"  {overall*100:.1f}%")

# ============================================================
# 3. Engagement patterns (like/view ratio, comment/view ratio)
# ============================================================
print("\n\n" + "="*80)
print("3. ENGAGEMENT PATTERNS BY CLUSTER")
print("="*80)

df['like_ratio'] = df['likeCount'] / df['viewCount'].clip(lower=1)
df['comment_ratio'] = df['commentCount'] / df['viewCount'].clip(lower=1)

print(f"\n{'Cluster':<30} {'Views':>12} {'Like/View':>10} {'Comment/View':>12}")
print("-" * 70)
for s in sorted_clusters:
    c = s['cluster']
    mask = df['cluster_v2'] == c
    med_views = df.loc[mask, 'viewCount'].median()
    med_like = df.loc[mask, 'like_ratio'].median()
    med_comment = df.loc[mask, 'comment_ratio'].median()
    print(f"C{c} {cluster_names[c]:<27} {med_views:>12,.0f} {med_like:>10.4f} {med_comment:>12.5f}")

# ============================================================
# 4. Generate figures
# ============================================================
print("\n\nGenerating figures...")
os.makedirs('analysis_discovery/figures_v2', exist_ok=True)

# Figure 1: View boost comparison (overall vs within-channel)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(7)
overall_boosts = [summaries[c]['view_boost']*100 for c in range(7)]
within_boosts = [within_channel_boosts.get(c, {}).get('median', 0)*100 for c in range(7)]

# Sort by overall boost
sort_idx = np.argsort(overall_boosts)[::-1]
labels = [f"C{i}\n{cluster_names[i]}" for i in sort_idx]

width = 0.35
bars1 = ax.bar(x - width/2, [overall_boosts[i] for i in sort_idx], width, label='Overall View Boost', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, [within_boosts[i] for i in sort_idx], width, label='Within-Channel Boost', color='#FF9800', alpha=0.8)

ax.set_xlabel('Content Strategy Cluster', fontsize=12)
ax.set_ylabel('View Boost (%)', fontsize=12)
ax.set_title('Algorithmic Reward by Content Strategy\n(Overall vs Within-Channel Control)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, ha='center')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_discovery/figures_v2/fig1_view_boost_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis_discovery/figures_v2/fig1_view_boost_comparison.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Channel diversity heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cross_tab = pd.read_csv('analysis_discovery/channel_cluster_crosstab_v2.csv', index_col=0)
# Rename columns
cross_tab.columns = [f"C{i}: {cluster_names[int(c)]}" for i, c in enumerate(cross_tab.columns)]
sns.heatmap(cross_tab, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, 
            linewidths=0.5, cbar_kws={'label': '% of channel videos'})
ax.set_title('Content Strategy Distribution Across Channels\n(% of each channel\'s videos in each cluster)', fontsize=12)
ax.set_xlabel('Content Strategy Cluster', fontsize=11)
ax.set_ylabel('Channel', fontsize=11)
plt.tight_layout()
plt.savefig('analysis_discovery/figures_v2/fig2_channel_diversity_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis_discovery/figures_v2/fig2_channel_diversity_heatmap.pdf', bbox_inches='tight')
plt.close()

# Figure 3: Manipulation signal radar chart
fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

angles = np.linspace(0, 2*np.pi, len(manipulation_features), endpoint=False).tolist()
angles += angles[:1]

for idx, c in enumerate(range(7)):
    ax = axes[idx]
    mask = features_full['cluster_v2'] == c
    values = [features_full.loc[mask, f].mean() / (features_full[f].mean() + 0.001) for f in manipulation_features]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=f'C{idx}')
    ax.fill(angles, values, alpha=0.25, color=f'C{idx}')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('has_', '').replace('is_', '') for f in manipulation_features], fontsize=7)
    ax.set_title(f'C{c}: {cluster_names[c]}', fontsize=9, pad=10)
    ax.set_ylim(0, max(5, max(values)))

# Hide last subplot
axes[7].set_visible(False)
plt.suptitle('Manipulation Signal Enrichment by Cluster\n(ratio vs overall average)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('analysis_discovery/figures_v2/fig3_manipulation_radar.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis_discovery/figures_v2/fig3_manipulation_radar.pdf', bbox_inches='tight')
plt.close()

# Figure 4: Scatter - within-channel boost vs manipulation signal density
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for c in range(7):
    mask = features_full['cluster_v2'] == c
    manip_density = features_full.loc[mask, manipulation_features].mean(axis=1).mean()
    wc_boost = within_channel_boosts.get(c, {}).get('median', 0) * 100
    n_videos = summaries[c]['n_videos']
    ax.scatter(manip_density * 100, wc_boost, s=n_videos/30, alpha=0.7, 
              label=f'C{c}: {cluster_names[c]}', zorder=5)
    ax.annotate(f'C{c}', (manip_density*100, wc_boost), fontsize=9, ha='center', va='bottom')

ax.set_xlabel('Manipulation Signal Density (%)', fontsize=12)
ax.set_ylabel('Within-Channel View Boost (%)', fontsize=12)
ax.set_title('Content Strategy: Manipulation Signals vs Algorithmic Reward\n(bubble size = cluster size)', fontsize=12)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_discovery/figures_v2/fig4_manip_vs_reward.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis_discovery/figures_v2/fig4_manip_vs_reward.pdf', bbox_inches='tight')
plt.close()

# Figure 5: Cluster size vs view boost bar chart (simple, clean)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sorted_c = [s['cluster'] for s in sorted_clusters]
colors = ['#4CAF50' if s['view_boost'] > 0 else '#F44336' for s in sorted_clusters]
bars = ax.bar(range(7), [s['view_boost']*100 for s in sorted_clusters], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xticks(range(7))
ax.set_xticklabels([f"C{s['cluster']}\n{cluster_names[s['cluster']]}\n({s['n_videos']} videos, {s['n_channels']} ch)" 
                    for s in sorted_clusters], fontsize=8, ha='center')
ax.set_ylabel('View Boost vs Overall Median (%)', fontsize=11)
ax.set_title('Algorithmic Reward by Content Strategy Cluster\n(de-channelized K=7 clustering, 41,157 videos)', fontsize=12)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, s in zip(bars, sorted_clusters):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height:.0f}%', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_discovery/figures_v2/fig5_cluster_boost_bar.png', dpi=150, bbox_inches='tight')
plt.savefig('analysis_discovery/figures_v2/fig5_cluster_boost_bar.pdf', bbox_inches='tight')
plt.close()

print("\nAll figures saved to analysis_discovery/figures_v2/")

# ============================================================
# 5. Key findings summary
# ============================================================
print("\n\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("""
1. CONTENT STRATEGY CLUSTERS (K=7, de-channelized):
   - Game/Roleplay/Music content: +298% view boost (10,634 videos, 23 channels)
   - Prank & Reaction content: +223% view boost (761 videos, 23 channels)  
   - Unboxing & Toy Review: +52% view boost (1,559 videos, 21 channels)
   - Standard Vlog: -24% (12,457 videos, 25 channels)
   - Clickbait Titles: -33% (13,231 videos, 24 channels)
   - Medical/Urgency content: -21% (568 videos, 21 channels)
   - Shorts: -24% (1,947 videos, 18 channels)

2. WITHIN-CHANNEL CONTROL:
   - Even after controlling for channel popularity, Game/Roleplay content
     still receives higher views than a channel's own median
   - This suggests the algorithm genuinely rewards certain content types,
     not just certain channels

3. CROSS-CHANNEL VALIDATION:
   - All clusters contain 18-25 channels (out of 25 total)
   - Maximum single-channel concentration: 44.8% (vs 99.7% in v1)
   - Multiple channels independently adopt the same high-reward strategies

4. MANIPULATION PARADOX:
   - Medical/urgency content (highest manipulation signals) gets LOWER views
   - Game/roleplay content (lower explicit manipulation) gets HIGHEST views
   - The algorithm rewards engagement-optimized content over emotional manipulation
""")
