"""
Generate publication-quality visualizations for the Kidfluencer study.
Produces figures suitable for ICWSM submission.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

# Paths
RESULTS_DIR = Path("/home/ubuntu/KidInfluencer/data/results")
FIGURES_DIR = Path("/home/ubuntu/KidInfluencer/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FAMILY_COLOR = '#E74C3C'  # Red
ADULT_COLOR = '#3498DB'   # Blue


def fig1_channel_drift_comparison():
    """Figure 1: Channel-level drift scores, family vs adult."""
    print("Generating Figure 1: Channel drift comparison...")
    
    df = pd.read_csv(RESULTS_DIR / "channel_drift_stats.csv")
    df = df[df['n_videos'] >= 10]  # Exclude channels with very few videos
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    family = df[df['channel_category'] == 'family'].sort_values('mean_drift', ascending=False)
    adult = df[df['channel_category'] == 'adult'].sort_values('mean_drift', ascending=False)
    
    # Plot bars
    x_family = range(len(family))
    x_adult = range(len(family) + 2, len(family) + 2 + len(adult))
    
    bars_f = ax.bar(x_family, family['mean_drift'], color=FAMILY_COLOR, alpha=0.8, label='Family Channels')
    bars_a = ax.bar(x_adult, adult['mean_drift'], color=ADULT_COLOR, alpha=0.8, label='Adult Channels')
    
    # Add group means
    family_mean = family['mean_drift'].mean()
    adult_mean = adult['mean_drift'].mean()
    ax.axhline(family_mean, color=FAMILY_COLOR, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(adult_mean, color=ADULT_COLOR, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Labels
    ax.set_xlabel('Channels (sorted by drift score)')
    ax.set_ylabel('Mean Exploitation Drift Score')
    ax.set_title('Exploitation Drift Scores: Family vs. Adult Channels')
    ax.legend(loc='upper right')
    ax.set_xticks([])
    
    # Annotate means
    ax.annotate(f'Family mean: {family_mean:.3f}', xy=(len(family)//2, family_mean),
               xytext=(len(family)//2, family_mean + 0.015), fontsize=9, color=FAMILY_COLOR,
               ha='center')
    ax.annotate(f'Adult mean: {adult_mean:.3f}', xy=(len(family) + 2 + len(adult)//2, adult_mean),
               xytext=(len(family) + 2 + len(adult)//2, adult_mean + 0.015), fontsize=9, color=ADULT_COLOR,
               ha='center')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_channel_drift_comparison.png")
    plt.close()
    print(f"  Saved: fig1_channel_drift_comparison.png")


def fig2_drift_distribution():
    """Figure 2: Distribution of drift scores by category."""
    print("Generating Figure 2: Drift score distributions...")
    
    df = pd.read_csv(RESULTS_DIR / "full_analysis_results.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Violin plot
    family_scores = df[df['channel_category'] == 'family']['drift_score']
    adult_scores = df[df['channel_category'] == 'adult']['drift_score']
    
    parts = ax1.violinplot([family_scores, adult_scores], positions=[1, 2], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(FAMILY_COLOR)
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor(ADULT_COLOR)
    parts['bodies'][1].set_alpha(0.7)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Family\n(n=41,159)', 'Adult\n(n=57,457)'])
    ax1.set_ylabel('Exploitation Drift Score')
    ax1.set_title('Distribution of Drift Scores')
    
    # Add t-test result
    t_stat, p_val = stats.ttest_ind(family_scores, adult_scores)
    ax1.text(0.5, 0.95, f't={t_stat:.1f}, p<0.001', transform=ax1.transAxes,
            ha='center', va='top', fontsize=10, style='italic')
    
    # KDE plot
    sns.kdeplot(data=family_scores, ax=ax2, color=FAMILY_COLOR, label='Family', fill=True, alpha=0.3)
    sns.kdeplot(data=adult_scores, ax=ax2, color=ADULT_COLOR, label='Adult', fill=True, alpha=0.3)
    ax2.set_xlabel('Exploitation Drift Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Kernel Density Estimation')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_drift_distribution.png")
    plt.close()
    print(f"  Saved: fig2_drift_distribution.png")


def fig3_temporal_drift():
    """Figure 3: Temporal drift trends (quarterly)."""
    print("Generating Figure 3: Temporal drift trends...")
    
    df = pd.read_csv(RESULTS_DIR / "quarterly_drift_trends.csv")
    df = df[df['n_videos'] >= 50]  # Only quarters with enough data
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    family = df[df['channel_category'] == 'family'].copy()
    adult = df[df['channel_category'] == 'adult'].copy()
    
    ax.plot(range(len(family)), family['mean_drift'].values, color=FAMILY_COLOR, 
            linewidth=2, label='Family Channels', marker='o', markersize=3)
    ax.plot(range(len(adult)), adult['mean_drift'].values, color=ADULT_COLOR, 
            linewidth=2, label='Adult Channels', marker='s', markersize=3)
    
    # X-axis labels (show every 4th quarter)
    n_ticks = max(len(family), len(adult))
    tick_positions = range(0, n_ticks, 4)
    if len(family) > 0:
        tick_labels = [family['quarter_str'].iloc[i] if i < len(family) else '' for i in tick_positions]
    else:
        tick_labels = [adult['quarter_str'].iloc[i] if i < len(adult) else '' for i in tick_positions]
    
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Mean Exploitation Drift Score')
    ax.set_title('Temporal Evolution of Exploitation Drift (Quarterly)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_temporal_drift.png")
    plt.close()
    print(f"  Saved: fig3_temporal_drift.png")


def fig4_did_results():
    """Figure 4: DiD analysis visualization."""
    print("Generating Figure 4: DiD results...")
    
    did_df = pd.read_csv(RESULTS_DIR / "did_observations.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Box plot of drift change by category
    family_change = did_df[did_df['is_family'] == 1]['drift_change']
    adult_change = did_df[did_df['is_family'] == 0]['drift_change']
    
    bp = ax1.boxplot([family_change, adult_change], labels=['Family\n(n={})'.format(len(family_change)), 
                                                             'Adult\n(n={})'.format(len(adult_change))],
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(FAMILY_COLOR)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(ADULT_COLOR)
    bp['boxes'][1].set_alpha(0.6)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Drift Score Change (Post - Pre Viral Hit)')
    ax1.set_title('Post-Viral Drift Change by Channel Type')
    
    # Add means
    ax1.scatter([1], [family_change.mean()], color='black', marker='D', s=50, zorder=5, label='Mean')
    ax1.scatter([2], [adult_change.mean()], color='black', marker='D', s=50, zorder=5)
    ax1.legend()
    
    # Right: Scatter plot of viral views vs drift change
    ax2.scatter(np.log10(did_df[did_df['is_family'] == 1]['viral_views'] + 1),
               did_df[did_df['is_family'] == 1]['drift_change'],
               color=FAMILY_COLOR, alpha=0.3, s=20, label='Family')
    ax2.scatter(np.log10(did_df[did_df['is_family'] == 0]['viral_views'] + 1),
               did_df[did_df['is_family'] == 0]['drift_change'],
               color=ADULT_COLOR, alpha=0.3, s=20, label='Adult')
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('log₁₀(Viral Video Views)')
    ax2.set_ylabel('Drift Score Change')
    ax2.set_title('Viral Hit Intensity vs. Content Drift')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_did_results.png")
    plt.close()
    print(f"  Saved: fig4_did_results.png")


def fig5_top_channels_deep_dive():
    """Figure 5: Deep dive into top family channels drift over time."""
    print("Generating Figure 5: Top channels deep dive...")
    
    df = pd.read_csv(RESULTS_DIR / "full_analysis_results.csv")
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Select top family channels by total views
    top_family = ['acefamily', 'ryansworld', 'familyfunpack', 'cocomelon', 'vladandniki']
    top_adult = ['caseyneistat', 'markwiens', 'mrbeast']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top panel: Family channels
    for channel in top_family:
        channel_df = df[df['channel_short_name'] == channel].sort_values('publishedAt')
        if len(channel_df) < 20:
            continue
        # Rolling average
        rolling = channel_df['drift_score'].rolling(window=50, min_periods=10).mean()
        axes[0].plot(channel_df['publishedAt'], rolling, linewidth=1.5, label=channel, alpha=0.8)
    
    axes[0].set_ylabel('Drift Score (50-video rolling mean)')
    axes[0].set_title('Family Channels: Exploitation Drift Over Time')
    axes[0].legend(loc='upper left', ncol=2)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Bottom panel: Adult channels
    for channel in top_adult:
        channel_df = df[df['channel_short_name'] == channel].sort_values('publishedAt')
        if len(channel_df) < 20:
            continue
        rolling = channel_df['drift_score'].rolling(window=50, min_periods=10).mean()
        axes[1].plot(channel_df['publishedAt'], rolling, linewidth=1.5, label=channel, alpha=0.8)
    
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Drift Score (50-video rolling mean)')
    axes[1].set_title('Adult Channels (Control): Exploitation Drift Over Time')
    axes[1].legend(loc='upper left', ncol=2)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_top_channels_deep_dive.png")
    plt.close()
    print(f"  Saved: fig5_top_channels_deep_dive.png")


def fig6_viral_hit_event_study():
    """Figure 6: Event study around viral hits."""
    print("Generating Figure 6: Event study around viral hits...")
    
    df = pd.read_csv(RESULTS_DIR / "full_analysis_results.csv")
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    WINDOW = 20
    
    # For each viral hit, extract drift scores in a window around it
    event_data = {'family': [], 'adult': []}
    
    for channel, group in df.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        category = group['channel_category'].iloc[0]
        viral_indices = group.index[group['is_viral'] == True].tolist()
        
        for vi in viral_indices:
            if vi < WINDOW or vi >= len(group) - WINDOW:
                continue
            window_scores = group.iloc[vi - WINDOW:vi + WINDOW + 1]['drift_score'].values
            if len(window_scores) == 2 * WINDOW + 1:
                event_data[category].append(window_scores)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(-WINDOW, WINDOW + 1)
    
    for category, color, label in [('family', FAMILY_COLOR, 'Family'), ('adult', ADULT_COLOR, 'Adult')]:
        if event_data[category]:
            arr = np.array(event_data[category])
            mean_curve = arr.mean(axis=0)
            se_curve = arr.std(axis=0) / np.sqrt(len(arr))
            
            ax.plot(x, mean_curve, color=color, linewidth=2, label=f'{label} (n={len(arr)})')
            ax.fill_between(x, mean_curve - 1.96*se_curve, mean_curve + 1.96*se_curve,
                          color=color, alpha=0.15)
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Viral Hit')
    ax.set_xlabel('Videos Relative to Viral Hit (t=0)')
    ax.set_ylabel('Mean Exploitation Drift Score')
    ax.set_title('Event Study: Drift Score Around Viral Hits')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_event_study.png")
    plt.close()
    print(f"  Saved: fig6_event_study.png")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    
    fig1_channel_drift_comparison()
    fig2_drift_distribution()
    fig3_temporal_drift()
    fig4_did_results()
    fig5_top_channels_deep_dive()
    fig6_viral_hit_event_study()
    
    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("Done!")
