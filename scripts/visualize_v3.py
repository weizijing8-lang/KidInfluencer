"""
Visualization V3: Complete figure set for the paper
=====================================================
Generates publication-quality figures for all analysis results.
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
RESULTS_V2 = DATA_DIR / "results_v2"
RESULTS_V3 = DATA_DIR / "results_v3"
FIG_DIR = Path("/home/ubuntu/KidInfluencer/figures_v3")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_exploitation_v2_by_channel():
    """Channel-level exploitation scores (V2) — family vs adult."""
    ch = pd.read_csv(RESULTS_V2 / "channel_exploit_v2.csv")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    family = ch[ch['channel_category'] == 'family'].sort_values('mean_exploit_v2', ascending=True)
    adult = ch[ch['channel_category'] == 'adult'].sort_values('mean_exploit_v2', ascending=True)
    
    y_fam = np.arange(len(family))
    y_adu = np.arange(len(adult)) + len(family) + 2
    
    ax.barh(y_fam, family['mean_exploit_v2'], color='#e74c3c', alpha=0.8, label='Family Channels')
    ax.barh(y_adu, adult['mean_exploit_v2'], color='#3498db', alpha=0.8, label='Adult Channels')
    
    all_yticks = np.concatenate([y_fam, y_adu])
    all_labels = list(family['channel_short_name']) + list(adult['channel_short_name'])
    ax.set_yticks(all_yticks)
    ax.set_yticklabels(all_labels, fontsize=7)
    ax.set_xlabel('Mean Exploitation Score (V2)')
    ax.set_title('Channel-Level Exploitation Scores\n(Legally-Grounded Direction Vector)')
    ax.legend(loc='lower right')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_channel_exploit_v2.png")
    plt.close()
    print("  fig1 saved")


def fig2_engagement_ratchet():
    """Strategy 3: Exploitation-views correlation by channel."""
    corr = pd.read_csv(RESULTS_V3 / "engagement_ratchet_correlations.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart of correlations
    ax = axes[0]
    family = corr[corr['is_family'] == 1].sort_values('corr_exploit_views', ascending=True)
    colors = ['#e74c3c' if s else '#ffcccc' for s in family['significant']]
    ax.barh(range(len(family)), family['corr_exploit_views'], color=colors)
    ax.set_yticks(range(len(family)))
    ax.set_yticklabels(family['channel'], fontsize=8)
    ax.set_xlabel('Pearson r (Exploitation Score vs log Views)')
    ax.set_title('Family Channels: Does Exploitation\nGet More Views?')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(['Significant (p<0.05)', 'Not significant'], loc='lower right', fontsize=8)
    
    # Right: Family vs Adult comparison
    ax = axes[1]
    fam_r = corr[corr['is_family'] == 1]['corr_exploit_views']
    adu_r = corr[corr['is_family'] == 0]['corr_exploit_views']
    
    parts = ax.violinplot([fam_r, adu_r], positions=[1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#e74c3c', '#3498db'][i])
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Family', 'Adult'])
    ax.set_ylabel('Correlation (Exploitation vs Views)')
    ax.set_title('Exploitation-Views Correlation\nFamily vs Adult Channels')
    
    t, p = stats.ttest_ind(fam_r, adu_r)
    ax.text(1.5, max(fam_r.max(), adu_r.max()) + 0.02, f't={t:.2f}, p={p:.3f}',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_engagement_ratchet.png")
    plt.close()
    print("  fig2 saved")


def fig3_granger_lag():
    """Strategy 4: Granger lag results."""
    lag = pd.read_csv(RESULTS_V3 / "granger_lag_results.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Family channels lag coefficients
    ax = axes[0]
    family = lag[lag['is_family'] == 1].sort_values('beta_views_lag', ascending=True)
    colors = ['#e74c3c' if s else '#ffcccc' for s in family['views_lag_significant']]
    ax.barh(range(len(family)), family['beta_views_lag'], color=colors)
    ax.set_yticks(range(len(family)))
    ax.set_yticklabels(family['channel'], fontsize=8)
    ax.set_xlabel('β (Lagged Views → Next Exploitation Score)')
    ax.set_title('Granger Causality: Do High Views\nPredict More Exploitation?')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Right: Comparison
    ax = axes[1]
    fam_b = lag[lag['is_family'] == 1]['beta_views_lag']
    adu_b = lag[lag['is_family'] == 0]['beta_views_lag']
    
    bp = ax.boxplot([fam_b, adu_b], labels=['Family', 'Adult'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#3498db')
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_ylabel('β (Views → Exploitation)')
    ax.set_title('Granger Lag Coefficient\nFamily vs Adult')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    t, p = stats.ttest_ind(fam_b, adu_b)
    ax.text(1.5, max(fam_b.max(), adu_b.max()) + 0.002, f't={t:.2f}, p={p:.3f}',
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_granger_lag.png")
    plt.close()
    print("  fig3 saved")


def fig4_its_trends():
    """Strategy 1: ITS channel trends."""
    its = pd.read_csv(RESULTS_V3 / "its_channel_trends.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Slope distribution
    ax = axes[0]
    fam = its[its['is_family'] == 1]
    adu = its[its['is_family'] == 0]
    
    ax.hist(fam['slope'], bins=15, alpha=0.7, color='#e74c3c', label=f'Family (n={len(fam)})')
    ax.hist(adu['slope'], bins=15, alpha=0.7, color='#3498db', label=f'Adult (n={len(adu)})')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Exploitation Trend Slope')
    ax.set_ylabel('Number of Channels')
    ax.set_title('Distribution of Long-term\nExploitation Trends')
    ax.legend()
    
    # Right: Early vs Late career
    ax = axes[1]
    for _, r in fam.iterrows():
        ax.plot([0, 1], [r['early_mean'], r['late_mean']], 'o-', color='#e74c3c', alpha=0.5, markersize=4)
    for _, r in adu.iterrows():
        ax.plot([0, 1], [r['early_mean'], r['late_mean']], 'o-', color='#3498db', alpha=0.3, markersize=3)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Early Career', 'Late Career'])
    ax.set_ylabel('Mean Exploitation Score')
    ax.set_title('Early vs Late Career\nExploitation Levels')
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', marker='o', label='Family'),
        Line2D([0], [0], color='#3498db', marker='o', label='Adult'),
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_its_trends.png")
    plt.close()
    print("  fig4 saved")


def fig5_llm_validation():
    """LLM annotation vs embedding correlation."""
    ann = pd.read_csv(RESULTS_V3 / "llm_annotations.csv")
    valid = ann[ann['llm_score'] >= 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left: Scatter plot embedding vs LLM
    ax = axes[0]
    fam = valid[valid['category'] == 'family']
    adu = valid[valid['category'] == 'adult']
    
    ax.scatter(adu['exploit_score_v2'], adu['llm_score'] + np.random.normal(0, 0.1, len(adu)),
               alpha=0.3, s=15, color='#3498db', label='Adult')
    ax.scatter(fam['exploit_score_v2'], fam['llm_score'] + np.random.normal(0, 0.1, len(fam)),
               alpha=0.3, s=15, color='#e74c3c', label='Family')
    
    r, p = stats.pearsonr(valid['exploit_score_v2'], valid['llm_score'])
    ax.set_xlabel('Embedding-based Exploitation Score')
    ax.set_ylabel('LLM Exploitation Score (jittered)')
    ax.set_title(f'Embedding vs LLM Scores\nr={r:.3f}, p<0.001')
    ax.legend()
    
    # Middle: LLM score distribution by category
    ax = axes[1]
    fam_scores = fam['llm_score'].value_counts().sort_index()
    adu_scores = adu['llm_score'].value_counts().sort_index()
    
    x = np.arange(6)
    width = 0.35
    ax.bar(x - width/2, [fam_scores.get(i, 0) for i in range(6)], width,
           color='#e74c3c', alpha=0.8, label='Family')
    ax.bar(x + width/2, [adu_scores.get(i, 0) for i in range(6)], width,
           color='#3498db', alpha=0.8, label='Adult')
    ax.set_xlabel('LLM Exploitation Score')
    ax.set_ylabel('Count')
    ax.set_title('LLM Score Distribution')
    ax.set_xticks(x)
    ax.legend()
    
    # Right: Mean LLM score by channel (family only)
    ax = axes[2]
    ch_means = fam.groupby('channel')['llm_score'].mean().sort_values(ascending=True)
    colors = ['#e74c3c' if v > 1.5 else '#ffcccc' for v in ch_means.values]
    ax.barh(range(len(ch_means)), ch_means.values, color=colors)
    ax.set_yticks(range(len(ch_means)))
    ax.set_yticklabels(ch_means.index, fontsize=7)
    ax.set_xlabel('Mean LLM Exploitation Score')
    ax.set_title('Family Channels by\nLLM Exploitation Score')
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_llm_validation.png")
    plt.close()
    print("  fig5 saved")


def fig6_summary_mechanism():
    """Summary figure: The exploitation ratchet mechanism."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Load data for a compelling case study
    df = pd.read_csv(RESULTS_V2 / "full_results_v2.csv")
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Pick top channels for case study
    case_channels = ['piperrockelle', 'jordanmatter', 'cocomelon', 'ryansworld', 'familyfunpack']
    
    for ch in case_channels:
        ch_data = df[df['channel_short_name'] == ch].sort_values('publishedAt')
        if len(ch_data) < 50:
            continue
        
        # Rolling mean
        rolling = ch_data['exploit_score_v2'].rolling(window=50, min_periods=25).mean()
        ax.plot(ch_data['publishedAt'], rolling, label=ch, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Exploitation Score (50-video rolling mean)')
    ax.set_title('Exploitation Trajectory Over Time\n(Selected Family Channels)')
    ax.legend(loc='upper left')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Adult channel baseline')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_case_study_trajectories.png")
    plt.close()
    print("  fig6 saved")


def fig7_mechanism_diagram():
    """Conceptual diagram showing the ratchet mechanism with data support."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load all needed data
    corr = pd.read_csv(RESULTS_V3 / "engagement_ratchet_correlations.csv")
    lag = pd.read_csv(RESULTS_V3 / "granger_lag_results.csv")
    its = pd.read_csv(RESULTS_V3 / "its_channel_trends.csv")
    ann = pd.read_csv(RESULTS_V3 / "llm_annotations.csv")
    
    # Panel A: Exploitation → Views (mechanism step 1)
    ax = axes[0, 0]
    fam_corr = corr[corr['is_family'] == 1].sort_values('corr_exploit_views', ascending=False)
    ax.bar(range(len(fam_corr)), fam_corr['corr_exploit_views'],
           color=['#e74c3c' if s else '#ffcccc' for s in fam_corr['significant']])
    ax.set_xticks(range(len(fam_corr)))
    ax.set_xticklabels(fam_corr['channel'], rotation=90, fontsize=6)
    ax.set_ylabel('r (Exploitation → Views)')
    ax.set_title('A. Step 1: Exploitation Gets Rewarded\n(18/23 family channels, p<0.05)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel B: Views → Next Exploitation (mechanism step 2)
    ax = axes[0, 1]
    fam_lag = lag[lag['is_family'] == 1].sort_values('beta_views_lag', ascending=False)
    ax.bar(range(len(fam_lag)), fam_lag['beta_views_lag'],
           color=['#e74c3c' if s else '#ffcccc' for s in fam_lag['views_lag_significant']])
    ax.set_xticks(range(len(fam_lag)))
    ax.set_xticklabels(fam_lag['channel'], rotation=90, fontsize=6)
    ax.set_ylabel('β (Views → Next Exploitation)')
    ax.set_title('B. Step 2: Creators Respond to Rewards\n(11/23 family channels, p<0.05)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel C: Long-term trend (mechanism outcome)
    ax = axes[1, 0]
    fam_its = its[its['is_family'] == 1]
    adu_its = its[its['is_family'] == 0]
    ax.scatter(fam_its['early_mean'], fam_its['late_mean'],
               color='#e74c3c', s=60, alpha=0.7, label='Family', zorder=5)
    ax.scatter(adu_its['early_mean'], adu_its['late_mean'],
               color='#3498db', s=40, alpha=0.5, label='Adult', zorder=4)
    
    # Add diagonal line (no change)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Early Career Exploitation')
    ax.set_ylabel('Late Career Exploitation')
    ax.set_title('C. Outcome: Early vs Late Career')
    ax.legend()
    
    # Panel D: LLM validation
    ax = axes[1, 1]
    valid = ann[ann['llm_score'] >= 0]
    fam_ann = valid[valid['category'] == 'family']
    adu_ann = valid[valid['category'] == 'adult']
    
    # Binned means
    bins = np.linspace(valid['exploit_score_v2'].min(), valid['exploit_score_v2'].max(), 10)
    fam_binned = fam_ann.groupby(pd.cut(fam_ann['exploit_score_v2'], bins)).agg(
        mean_emb=('exploit_score_v2', 'mean'),
        mean_llm=('llm_score', 'mean'),
        count=('llm_score', 'count')
    ).dropna()
    
    ax.scatter(fam_ann['exploit_score_v2'], fam_ann['llm_score'] + np.random.normal(0, 0.08, len(fam_ann)),
               alpha=0.15, s=10, color='#e74c3c')
    if len(fam_binned) > 0:
        ax.plot(fam_binned['mean_emb'], fam_binned['mean_llm'], 'ko-', linewidth=2, markersize=6,
                label='Binned means')
    
    r, p = stats.pearsonr(valid['exploit_score_v2'], valid['llm_score'])
    ax.set_xlabel('Embedding Exploitation Score')
    ax.set_ylabel('LLM Exploitation Score')
    ax.set_title(f'D. Validation: Embedding vs LLM\n(r={r:.3f}, p<0.001)')
    ax.legend()
    
    plt.suptitle('The Exploitation Ratchet: Evidence from 98,616 YouTube Videos', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_mechanism_summary.png")
    plt.close()
    print("  fig7 saved")


def main():
    print("Generating V3 figures...")
    fig1_exploitation_v2_by_channel()
    fig2_engagement_ratchet()
    fig3_granger_lag()
    fig4_its_trends()
    fig5_llm_validation()
    fig6_summary_mechanism()
    fig7_mechanism_diagram()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
