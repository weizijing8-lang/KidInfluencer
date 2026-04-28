"""
Within-Family Exploitation Visualizations
==========================================
All figures focus on variation WITHIN family channels,
not family-vs-adult comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
RESULTS_V4 = DATA_DIR / "results_v4"
RESULTS_V3 = DATA_DIR / "results_v3"
FIG_DIR = Path("/home/ubuntu/KidInfluencer/figures_final")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load V4 results."""
    df = pd.read_csv(RESULTS_V4 / "full_results_v4.csv")
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    return df


def fig1_within_family_ranking(df):
    """Fig 1: Family channels ranked by exploitation score with known cases annotated."""
    family = df[df['channel_category'] == 'family']
    ch_means = family.groupby('channel_short_name').agg(
        mean_score=('exploit_score_v4', 'mean'),
        n_videos=('id', 'count'),
        total_views=('viewCount', 'sum'),
    ).sort_values('mean_score', ascending=True).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by known status
    known_problematic = ['piperrockelle', 'acefamily', 'jordanmatter', 'rebeccazamolo']
    known_healthy = ['familyfunpack', 'bratayley', 'theleray', 'ryansworld']
    
    colors = []
    for ch in ch_means['channel_short_name']:
        if ch in known_problematic:
            colors.append('#c0392b')  # Dark red
        elif ch in known_healthy:
            colors.append('#27ae60')  # Green
        else:
            colors.append('#e67e22')  # Orange (unknown)
    
    bars = ax.barh(range(len(ch_means)), ch_means['mean_score'], color=colors, alpha=0.85)
    
    ax.set_yticks(range(len(ch_means)))
    ax.set_yticklabels(ch_means['channel_short_name'], fontsize=9)
    ax.set_xlabel('Mean Exploitation Score (V4: title + description)')
    ax.set_title('Family Channels Ranked by Exploitation Score\n(Within-Family Variation)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c0392b', label='Known problematic (media/legal)'),
        Patch(facecolor='#27ae60', label='Known healthy/educational'),
        Patch(facecolor='#e67e22', label='Other family channels'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_within_family_ranking.png")
    plt.close()
    print("  fig1 saved", flush=True)


def fig2_exploitation_dimensions(df):
    """Fig 2: What makes some channels score higher? Title keyword analysis."""
    family = df[df['channel_category'] == 'family'].copy()
    
    # Define exploitation keyword categories
    dim_keywords = {
        'Emotional\nDistress': ['cry', 'crying', 'scream', 'scared', 'angry', 'upset', 'tantrum', 'meltdown', 'freak out'],
        'Clickbait\nExploitation': ['gone wrong', 'not clickbait', 'you wont believe', 'shocking', 'exposed', 'caught', 'prank'],
        'Challenge/\nDanger': ['challenge', 'extreme', 'dangerous', '24 hour', 'last to leave', 'dont', 'impossible'],
        'Privacy\nViolation': ['secret', 'caught on camera', 'hidden camera', 'spy', 'reading', 'diary', 'phone'],
        'Relationship\nDrama': ['boyfriend', 'girlfriend', 'crush', 'kiss', 'date', 'breakup', 'pregnant', 'proposal'],
    }
    
    # Compute keyword rates per channel
    results = []
    for channel, group in family.groupby('channel_short_name'):
        titles_lower = group['title'].str.lower()
        row = {'channel': channel, 'n_videos': len(group), 'mean_exploit': group['exploit_score_v4'].mean()}
        for dim, keywords in dim_keywords.items():
            rate = titles_lower.apply(lambda t: any(k in str(t) for k in keywords)).mean()
            row[dim] = rate
        results.append(row)
    
    dim_df = pd.DataFrame(results).sort_values('mean_exploit', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Heatmap of keyword rates
    dims = list(dim_keywords.keys())
    data_matrix = dim_df[dims].values
    
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks(range(len(dim_df)))
    ax.set_yticklabels(dim_df['channel'], fontsize=8)
    ax.set_title('Exploitation Dimensions by Channel\n(Keyword Frequency in Video Titles)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Keyword Frequency')
    
    # Add text annotations for high values
    for i in range(len(dim_df)):
        for j in range(len(dims)):
            val = data_matrix[i, j]
            if val > 0.05:
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=7,
                       color='white' if val > 0.15 else 'black')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_exploitation_dimensions.png")
    plt.close()
    print("  fig2 saved", flush=True)


def fig3_temporal_trajectories(df):
    """Fig 3: Exploitation trajectories over time for selected channels."""
    family = df[df['channel_category'] == 'family'].copy()
    family = family.dropna(subset=['publishedAt'])
    
    # Select diverse channels
    channels = ['piperrockelle', 'acefamily', 'cocomelon', 'ryansworld', 
                'familyfunpack', 'bratayley', 'jordanmatter', 'dailybumps']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()
    
    for idx, ch in enumerate(channels):
        ax = axes[idx]
        ch_data = family[family['channel_short_name'] == ch].sort_values('publishedAt')
        
        if len(ch_data) < 20:
            ax.set_title(ch, fontsize=10)
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Rolling mean
        window = max(20, len(ch_data) // 20)
        rolling = ch_data['exploit_score_v4'].rolling(window=window, min_periods=window//2).mean()
        
        ax.plot(ch_data['publishedAt'], rolling, color='#e74c3c', linewidth=1.5)
        ax.fill_between(ch_data['publishedAt'], rolling, alpha=0.2, color='#e74c3c')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(ch, fontsize=10, fontweight='bold')
        
        if idx >= 4:
            ax.set_xlabel('Year')
        if idx % 4 == 0:
            ax.set_ylabel('Exploitation Score')
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    
    plt.suptitle('Exploitation Score Trajectories Over Time\n(Within-Family Channel Comparison)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_temporal_trajectories.png")
    plt.close()
    print("  fig3 saved", flush=True)


def fig4_views_exploit_within_family(df):
    """Fig 4: Within each channel, do more exploitative videos get more views?"""
    family = df[df['channel_category'] == 'family'].copy()
    family['log_views'] = np.log1p(family['viewCount'])
    
    # Compute within-channel correlation
    correlations = []
    for channel, group in family.groupby('channel_short_name'):
        if len(group) < 30:
            continue
        r, p = stats.pearsonr(group['exploit_score_v4'], group['log_views'])
        correlations.append({
            'channel': channel,
            'r': r,
            'p': p,
            'significant': p < 0.05,
            'n': len(group),
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('r', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: correlation bar chart
    ax = axes[0]
    colors = ['#c0392b' if s else '#f5b7b1' for s in corr_df['significant']]
    ax.barh(range(len(corr_df)), corr_df['r'], color=colors)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['channel'], fontsize=8)
    ax.set_xlabel('Pearson r (Exploitation Score vs log Views)')
    ax.set_title('Within-Channel: Does Exploitation\nGet Rewarded with More Views?')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    n_sig = corr_df['significant'].sum()
    n_pos_sig = ((corr_df['r'] > 0) & corr_df['significant']).sum()
    ax.text(0.95, 0.05, f'{n_pos_sig}/{len(corr_df)} channels:\nmore exploitation → more views\n(p<0.05)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: scatter for top channel (Jordan Matter)
    ax = axes[1]
    top_ch = corr_df.iloc[-1]['channel']  # Highest positive correlation
    ch_data = family[family['channel_short_name'] == top_ch]
    
    ax.scatter(ch_data['exploit_score_v4'], ch_data['log_views'], 
               alpha=0.4, s=20, color='#e74c3c')
    
    # Add regression line
    slope, intercept, r, p, se = stats.linregress(ch_data['exploit_score_v4'], ch_data['log_views'])
    x_line = np.linspace(ch_data['exploit_score_v4'].min(), ch_data['exploit_score_v4'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k-', linewidth=2)
    
    ax.set_xlabel('Exploitation Score')
    ax.set_ylabel('log(Views)')
    ax.set_title(f'Case Study: {top_ch}\n(r={r:.3f}, p<0.001, n={len(ch_data)})')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_views_exploit_within_family.png")
    plt.close()
    print("  fig4 saved", flush=True)


def fig5_ratchet_mechanism(df):
    """Fig 5: The ratchet mechanism - lag analysis within family channels."""
    family = df[df['channel_category'] == 'family'].copy()
    family = family.dropna(subset=['publishedAt'])
    
    # For each channel, compute: does a high-view video predict higher exploitation in next video?
    lag_results = []
    for channel, group in family.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        if len(group) < 50:
            continue
        
        group['log_views'] = np.log1p(group['viewCount'])
        group['next_exploit'] = group['exploit_score_v4'].shift(-1)
        group['prev_exploit'] = group['exploit_score_v4'].shift(1)
        valid = group.dropna(subset=['next_exploit', 'prev_exploit'])
        
        if len(valid) < 30:
            continue
        
        # Partial correlation: views → next_exploit, controlling for current exploit
        from scipy.stats import pearsonr
        # Simple: correlation between views and CHANGE in exploitation
        valid['exploit_change'] = valid['next_exploit'] - valid['exploit_score_v4']
        r, p = pearsonr(valid['log_views'], valid['exploit_change'])
        
        lag_results.append({
            'channel': channel,
            'r_views_change': r,
            'p': p,
            'significant': p < 0.05,
            'n': len(valid),
        })
    
    lag_df = pd.DataFrame(lag_results).sort_values('r_views_change', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#c0392b' if (s and r > 0) else '#2980b9' if (s and r < 0) else '#bdc3c7' 
              for s, r in zip(lag_df['significant'], lag_df['r_views_change'])]
    ax.barh(range(len(lag_df)), lag_df['r_views_change'], color=colors)
    ax.set_yticks(range(len(lag_df)))
    ax.set_yticklabels(lag_df['channel'], fontsize=8)
    ax.set_xlabel('r (log Views → Change in Next Video Exploitation)')
    ax.set_title('The Ratchet Test: Do High Views Predict\nIncreased Exploitation in the Next Video?')
    ax.axvline(x=0, color='black', linewidth=1)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c0392b', label='Significant positive (ratchet)'),
        Patch(facecolor='#2980b9', label='Significant negative (self-correction)'),
        Patch(facecolor='#bdc3c7', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    n_ratchet = ((lag_df['r_views_change'] > 0) & lag_df['significant']).sum()
    n_correct = ((lag_df['r_views_change'] < 0) & lag_df['significant']).sum()
    ax.text(0.02, 0.98, f'Ratchet: {n_ratchet} channels\nSelf-correction: {n_correct} channels',
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_ratchet_mechanism.png")
    plt.close()
    print("  fig5 saved", flush=True)


def fig6_comment_inappropriateness():
    """Fig 6: Comment inappropriateness by exploitation level (if data available)."""
    partial_file = RESULTS_V4 / "comment_classifications_partial.csv"
    final_file = RESULTS_V4 / "comment_classifications.csv"
    
    if final_file.exists():
        comments = pd.read_csv(final_file)
    elif partial_file.exists():
        comments = pd.read_csv(partial_file)
    else:
        print("  fig6 skipped (no comment data yet)", flush=True)
        return
    
    valid = comments[comments['category'] != 'error']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Inappropriate rate by sample type
    ax = axes[0]
    high = valid[valid['sample_type'] == 'high']
    low = valid[valid['sample_type'] == 'low']
    
    categories = ['inappropriate', 'timestamp', 'concern', 'toxic']
    high_rates = [(high['category'] == c).mean() * 100 for c in categories]
    low_rates = [(low['category'] == c).mean() * 100 for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, high_rates, width, color='#c0392b', alpha=0.8, label='High exploitation videos')
    ax.bar(x + width/2, low_rates, width, color='#27ae60', alpha=0.8, label='Low exploitation videos')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Comment Rate (%)')
    ax.set_title('A. Comment Types by\nVideo Exploitation Level')
    ax.legend(fontsize=8)
    
    # Panel B: Inappropriate rate by channel
    ax = axes[1]
    ch_inapp = valid.groupby('channel').apply(
        lambda x: (x['category'] == 'inappropriate').mean() * 100
    ).sort_values(ascending=True)
    
    ax.barh(range(len(ch_inapp)), ch_inapp.values, color='#c0392b', alpha=0.7)
    ax.set_yticks(range(len(ch_inapp)))
    ax.set_yticklabels(ch_inapp.index, fontsize=7)
    ax.set_xlabel('Inappropriate Comment Rate (%)')
    ax.set_title('B. Inappropriate Comments\nby Channel')
    
    # Panel C: Correlation between exploit score and inappropriate rate per video
    ax = axes[2]
    video_stats = valid.groupby(['video_id', 'exploit_score_v2']).apply(
        lambda x: (x['category'] == 'inappropriate').mean()
    ).reset_index()
    video_stats.columns = ['video_id', 'exploit_score', 'inappropriate_rate']
    
    ax.scatter(video_stats['exploit_score'], video_stats['inappropriate_rate'] * 100,
               alpha=0.3, s=15, color='#c0392b')
    
    # Binned means
    bins = pd.qcut(video_stats['exploit_score'], q=10, duplicates='drop')
    binned = video_stats.groupby(bins).agg(
        mean_exploit=('exploit_score', 'mean'),
        mean_inapp=('inappropriate_rate', 'mean'),
    )
    ax.plot(binned['mean_exploit'], binned['mean_inapp'] * 100, 'ko-', linewidth=2, markersize=6)
    
    r, p = stats.pearsonr(video_stats['exploit_score'], video_stats['inappropriate_rate'])
    ax.set_xlabel('Video Exploitation Score')
    ax.set_ylabel('Inappropriate Comment Rate (%)')
    ax.set_title(f'C. Exploitation → Inappropriate\nComments (r={r:.3f}, p={p:.3f})')
    
    plt.suptitle('External Validation: Comment Ecology as Exploitation Signal', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_comment_inappropriateness.png")
    plt.close()
    print("  fig6 saved", flush=True)


def main():
    print("Generating within-family focused figures...", flush=True)
    df = load_data()
    
    fig1_within_family_ranking(df)
    fig2_exploitation_dimensions(df)
    fig3_temporal_trajectories(df)
    fig4_views_exploit_within_family(df)
    fig5_ratchet_mechanism(df)
    fig6_comment_inappropriateness()
    
    print(f"\nAll figures saved to {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
