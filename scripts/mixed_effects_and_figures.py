#!/usr/bin/env python3
"""
Mixed-effects regression and figure generation for AIES paper v4.
- Random intercepts for channel
- Fixed effects for exploitation dimensions
- Generates publication-quality figures
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

DATA_DIR = Path("/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v4")
FIG_DIR = Path("/home/ubuntu/KidInfluencer/figures_v4")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DIMENSIONS = [
    'performative_labor', 'emotional_bait', 'narrative_conflict',
    'challenge_format', 'commercial_content', 'privacy_violation'
]

DIM_LABELS = {
    'performative_labor': 'Performative\nLabor',
    'emotional_bait': 'Emotional\nBait',
    'narrative_conflict': 'Narrative\nConflict',
    'challenge_format': 'Challenge\nFormat',
    'commercial_content': 'Commercial\nContent',
    'privacy_violation': 'Privacy\nViolation'
}


def load_data():
    df = pd.read_csv(DATA_DIR / "classified_videos_v4.csv")
    df['viewCount'] = pd.to_numeric(df['viewCount'], errors='coerce')
    df = df[df['viewCount'] > 0].copy()
    df['log_views'] = np.log10(df['viewCount'])
    df['log_views_ln'] = np.log1p(df['viewCount'])
    print(f"Loaded {len(df)} videos with valid view counts, {df['channel_short_name'].nunique()} channels")
    return df


def run_mixed_effects(df):
    """Run mixed-effects linear regression with channel random intercepts."""
    print("\n" + "=" * 60)
    print("MIXED-EFFECTS REGRESSION")
    print("=" * 60)

    # Prepare data
    analysis_df = df.dropna(subset=['exploitation_score_v4', 'log_views', 'channel_short_name']).copy()
    
    # Model 1: Overall exploitation score
    print("\n--- Model 1: Overall Exploitation Score ---")
    try:
        md1 = smf.mixedlm(
            "log_views ~ exploitation_score_v4",
            analysis_df,
            groups=analysis_df["channel_short_name"]
        )
        mdf1 = md1.fit(reml=True)
        print(mdf1.summary())
        
        results_m1 = {
            'coefficient': float(mdf1.fe_params['exploitation_score_v4']),
            'std_err': float(mdf1.bse_fe['exploitation_score_v4']),
            'z_value': float(mdf1.tvalues['exploitation_score_v4']),
            'p_value': float(mdf1.pvalues['exploitation_score_v4']),
            'ci_lower': float(mdf1.conf_int().loc['exploitation_score_v4', 0]),
            'ci_upper': float(mdf1.conf_int().loc['exploitation_score_v4', 1]),
            'n_obs': int(mdf1.nobs),
            'n_groups': int(mdf1.n_groups),
            'random_effect_var': float(mdf1.cov_re.iloc[0, 0]),
        }
        print(f"\n  Coefficient: {results_m1['coefficient']:.4f}")
        print(f"  95% CI: [{results_m1['ci_lower']:.4f}, {results_m1['ci_upper']:.4f}]")
        print(f"  p-value: {results_m1['p_value']:.6e}")
        print(f"  Random effect variance: {results_m1['random_effect_var']:.4f}")
    except Exception as e:
        print(f"  Model 1 failed: {e}")
        results_m1 = None

    # Model 2: Per-dimension fixed effects
    print("\n--- Model 2: Per-Dimension Fixed Effects ---")
    dim_cols = [f"{d}_combined" for d in DIMENSIONS if f"{d}_combined" in analysis_df.columns]
    
    if dim_cols:
        formula = f"log_views ~ {' + '.join(dim_cols)}"
        try:
            md2 = smf.mixedlm(
                formula,
                analysis_df,
                groups=analysis_df["channel_short_name"]
            )
            mdf2 = md2.fit(reml=True)
            print(mdf2.summary())
            
            results_m2 = {}
            for col in dim_cols:
                dim_name = col.replace('_combined', '')
                results_m2[dim_name] = {
                    'coefficient': float(mdf2.fe_params[col]),
                    'std_err': float(mdf2.bse_fe[col]),
                    'z_value': float(mdf2.tvalues[col]),
                    'p_value': float(mdf2.pvalues[col]),
                }
                sig = '***' if mdf2.pvalues[col] < 0.001 else '**' if mdf2.pvalues[col] < 0.01 else '*' if mdf2.pvalues[col] < 0.05 else 'n.s.'
                print(f"  {dim_name}: β={mdf2.fe_params[col]:.4f}, p={mdf2.pvalues[col]:.4e} {sig}")
        except Exception as e:
            print(f"  Model 2 failed: {e}")
            results_m2 = None
    else:
        results_m2 = None

    # Save results
    all_results = {'model1_overall': results_m1, 'model2_dimensions': results_m2}
    with open(DATA_DIR / "mixed_effects_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    return results_m1, results_m2


def compute_effect_sizes(df):
    """Compute Cohen's d and Cliff's delta for each dimension."""
    print("\n" + "=" * 60)
    print("EFFECT SIZES")
    print("=" * 60)
    
    effect_sizes = []
    for dim in DIMENSIONS:
        col = f"{dim}_combined"
        if col not in df.columns:
            continue
        
        high = df[df[col] >= 0.5]['log_views']
        low = df[df[col] < 0.5]['log_views']
        
        if len(high) < 10 or len(low) < 10:
            continue
        
        # Cohen's d
        pooled_std = np.sqrt(((len(high)-1)*high.std()**2 + (len(low)-1)*low.std()**2) / (len(high)+len(low)-2))
        cohens_d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Cliff's delta (non-parametric)
        n1, n2 = len(high), len(low)
        # Efficient computation
        u_stat, _ = stats.mannwhitneyu(high, low, alternative='two-sided')
        cliffs_delta = (2 * u_stat) / (n1 * n2) - 1
        
        effect_sizes.append({
            'dimension': dim,
            'n_high': len(high),
            'n_low': len(low),
            'mean_high': high.mean(),
            'mean_low': low.mean(),
            'cohens_d': cohens_d,
            'cliffs_delta': cliffs_delta,
        })
        
        # Interpret
        d_interp = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        print(f"  {dim}: Cohen's d={cohens_d:.3f} ({d_interp}), Cliff's δ={cliffs_delta:.3f}")
    
    es_df = pd.DataFrame(effect_sizes)
    es_df.to_csv(DATA_DIR / "effect_sizes.csv", index=False)
    return es_df


def fig_score_distribution(df):
    """Figure 1: Distribution of exploitation scores."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.hist(df['exploitation_score_v4'], bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Classification threshold')
    ax.set_xlabel('Exploitation Score')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Distribution of Multi-Modal Exploitation Scores (n=4,307)')
    ax.legend()
    
    # Add annotation
    n_exploit = (df['exploitation_score_v4'] >= 0.5).sum()
    n_clean = (df['exploitation_score_v4'] < 0.5).sum()
    ax.text(0.75, 0.85, f'Exploitative: {n_exploit}\n({100*n_exploit/len(df):.1f}%)',
            transform=ax.transAxes, fontsize=10, color='darkred')
    ax.text(0.15, 0.85, f'Clean: {n_clean}\n({100*n_clean/len(df):.1f}%)',
            transform=ax.transAxes, fontsize=10, color='darkblue')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_score_distribution.png")
    plt.close()
    print("  Saved fig1_score_distribution.png")


def fig_dimension_premiums(df):
    """Figure 2: Per-dimension engagement premiums with confidence intervals."""
    dim_results = pd.read_csv(DATA_DIR / "dimension_results_v4.csv")
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    dims = dim_results['dimension'].tolist()
    premiums = dim_results['mean_premium'].tolist()
    labels = [DIM_LABELS.get(d, d) for d in dims]
    colors = ['#4CAF50' if p > 0 else '#F44336' for p in premiums]
    
    bars = ax.barh(range(len(dims)), premiums, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean Within-Channel Log-View Premium')
    ax.set_title('Engagement Premium by Exploitation Dimension (FDR-corrected)')
    
    # Add significance stars
    for i, row in dim_results.iterrows():
        p = row['p_corrected']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        x_pos = row['mean_premium'] + 0.02 if row['mean_premium'] > 0 else row['mean_premium'] - 0.06
        ax.text(x_pos, i, stars, va='center', fontsize=11, fontweight='bold')
    
    # Add p-values
    for i, row in dim_results.iterrows():
        ax.text(0.95, i, f"p={row['p_corrected']:.4f}", va='center', ha='right',
                transform=ax.get_yaxis_transform(), fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_dimension_premiums.png")
    plt.close()
    print("  Saved fig2_dimension_premiums.png")


def fig_within_channel_scatter(df):
    """Figure 3: Within-channel exploitation score vs views (selected channels)."""
    # Select top 9 channels by video count
    top_channels = df['channel_short_name'].value_counts().head(9).index.tolist()
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ch in enumerate(top_channels):
        ax = axes[i]
        ch_data = df[df['channel_short_name'] == ch]
        
        scatter = ax.scatter(
            ch_data['exploitation_score_v4'], ch_data['log_views'],
            alpha=0.5, s=20, c=ch_data['exploitation_score_v4'],
            cmap='RdYlGn_r', vmin=0, vmax=1
        )
        
        # Trend line
        if len(ch_data) > 5:
            z = np.polyfit(ch_data['exploitation_score_v4'], ch_data['log_views'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(ch_data['exploitation_score_v4'].min(), 
                                  ch_data['exploitation_score_v4'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5)
        
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Log₁₀(Views)', fontsize=8)
        ax.tick_params(labelsize=8)
    
    plt.suptitle('Within-Channel: Exploitation Score vs. View Count', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_within_channel_scatter.png")
    plt.close()
    print("  Saved fig3_within_channel_scatter.png")


def fig_channel_premiums(df):
    """Figure 4: Channel-level premium distribution."""
    ch_df = pd.read_csv(DATA_DIR / "channel_premiums_v4.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: histogram of log premiums
    ax1.hist(ch_df['log_premium'], bins=20, color='#673AB7', alpha=0.7, edgecolor='white')
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.axvline(ch_df['log_premium'].median(), color='green', linestyle='-', linewidth=1.5, 
                label=f"Median = {ch_df['log_premium'].median():.3f}")
    ax1.set_xlabel('Log-View Premium (Exploitative - Clean)')
    ax1.set_ylabel('Number of Channels')
    ax1.set_title('Distribution of Within-Channel Premiums')
    ax1.legend()
    
    n_pos = (ch_df['log_premium'] > 0).sum()
    ax1.text(0.7, 0.9, f'{n_pos}/{len(ch_df)} positive\n({100*n_pos/len(ch_df):.0f}%)',
             transform=ax1.transAxes, fontsize=10)
    
    # Right: view ratio
    ch_sorted = ch_df.sort_values('view_ratio', ascending=True)
    colors = ['#4CAF50' if r > 1 else '#F44336' for r in ch_sorted['view_ratio']]
    ax2.barh(range(len(ch_sorted)), ch_sorted['view_ratio'] - 1, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('View Ratio - 1 (Exploitative / Clean)')
    ax2.set_title('Per-Channel View Ratio')
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_channel_premiums.png")
    plt.close()
    print("  Saved fig4_channel_premiums.png")


def fig_commercial_vs_emotional(df):
    """Figure 5: Commercial vs Emotional exploitation - contrasting effects."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Emotional bait
    col = 'emotional_bait_combined'
    if col in df.columns:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        df['eb_bin'] = pd.cut(df[col], bins=bins)
        grouped = df.groupby('eb_bin')['log_views'].agg(['mean', 'sem', 'count'])
        x = range(len(grouped))
        ax1.bar(x, grouped['mean'], yerr=grouped['sem']*1.96, 
                color='#FF5722', alpha=0.7, capsize=4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=9)
        ax1.set_xlabel('Emotional Bait Score')
        ax1.set_ylabel('Mean Log₁₀(Views)')
        ax1.set_title('Emotional Bait → Higher Views')
    
    # Commercial content
    col = 'commercial_content_combined'
    if col in df.columns:
        df['cc_bin'] = pd.cut(df[col], bins=bins)
        grouped = df.groupby('cc_bin')['log_views'].agg(['mean', 'sem', 'count'])
        x = range(len(grouped))
        ax2.bar(x, grouped['mean'], yerr=grouped['sem']*1.96,
                color='#607D8B', alpha=0.7, capsize=4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=9)
        ax2.set_xlabel('Commercial Content Score')
        ax2.set_ylabel('Mean Log₁₀(Views)')
        ax2.set_title('Commercial Content → Lower Views')
    
    plt.suptitle('Contrasting Effects: Emotional vs. Commercial Exploitation', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_emotional_vs_commercial.png")
    plt.close()
    print("  Saved fig5_emotional_vs_commercial.png")


def main():
    print("=" * 60)
    print("MIXED-EFFECTS REGRESSION & FIGURE GENERATION")
    print("=" * 60)
    
    df = load_data()
    
    # Mixed-effects regression
    results_m1, results_m2 = run_mixed_effects(df)
    
    # Effect sizes
    es_df = compute_effect_sizes(df)
    
    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    fig_score_distribution(df)
    fig_dimension_premiums(df)
    fig_within_channel_scatter(df)
    fig_channel_premiums(df)
    fig_commercial_vs_emotional(df)
    
    print(f"\nAll figures saved to {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
