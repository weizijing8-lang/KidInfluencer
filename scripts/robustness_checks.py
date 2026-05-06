"""
Robustness Checks for AIES 2026 Paper
1. Same-year within-channel comparison (controls for video age)
2. Views per day as alternative DV
3. Report both with and without age control
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load scored data
scored = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v3/videos_with_exploitation_scores.csv')

# Load stratified sample for publishedAt
strat = pd.read_csv('/home/ubuntu/KidInfluencer/data/stratified_sample.csv')

# Merge publishedAt
if 'publishedAt' not in scored.columns:
    strat_dates = strat[['id', 'publishedAt']].drop_duplicates()
    scored = scored.merge(strat_dates, on='id', how='left')

# Parse dates
scored['pub_date'] = pd.to_datetime(scored['publishedAt'], errors='coerce')
scored['pub_year'] = scored['pub_date'].dt.year

# Calculate days since publish (reference: 2026-05-05)
reference_date = pd.Timestamp('2026-05-05', tz='UTC')
scored['pub_date_tz'] = scored['pub_date'].dt.tz_localize('UTC') if scored['pub_date'].dt.tz is None else scored['pub_date']
scored['days_since_publish'] = (reference_date - scored['pub_date_tz']).dt.days
scored['days_since_publish'] = scored['days_since_publish'].clip(lower=1)  # avoid division by zero

# Views per day
scored['views_per_day'] = scored['viewCount'] / scored['days_since_publish']
scored['log_views_per_day'] = np.log1p(scored['views_per_day'])

# Define exploitation dimensions
dimensions = ['performative', 'emotional_bait', 'narrative_conflict', 
              'challenge_format', 'commercial_content', 'privacy_violation']

# Median split for exploitation score
scored['high_exploit'] = (scored['exploitation_score'] >= scored['exploitation_score'].median()).astype(int)

print("=" * 70)
print("ROBUSTNESS CHECK 1: Views Per Day (controls for video age mechanically)")
print("=" * 70)

# Within-channel comparison using views_per_day
channel_boosts_vpd = []
for ch, grp in scored.groupby('channel_short_name'):
    if len(grp) < 10:
        continue
    high = grp[grp['high_exploit'] == 1]['views_per_day']
    low = grp[grp['high_exploit'] == 0]['views_per_day']
    if len(high) >= 3 and len(low) >= 3:
        boost = (high.median() - low.median()) / low.median() * 100 if low.median() > 0 else np.nan
        channel_boosts_vpd.append({'channel': ch, 'boost_pct': boost, 
                                    'n_high': len(high), 'n_low': len(low)})

df_vpd = pd.DataFrame(channel_boosts_vpd).dropna()
print(f"\nChannels with valid data: {len(df_vpd)}")
print(f"Mean boost (views/day): {df_vpd['boost_pct'].mean():.1f}%")
print(f"Median boost (views/day): {df_vpd['boost_pct'].median():.1f}%")
print(f"Channels with positive boost: {(df_vpd['boost_pct'] > 0).sum()}/{len(df_vpd)} ({(df_vpd['boost_pct'] > 0).mean()*100:.0f}%)")

# One-sample t-test: is mean boost > 0?
t_stat, p_val = stats.ttest_1samp(df_vpd['boost_pct'], 0)
print(f"One-sample t-test (boost > 0): t={t_stat:.3f}, p={p_val:.4f}")

# Wilcoxon signed-rank test (non-parametric)
w_stat, w_p = stats.wilcoxon(df_vpd['boost_pct'])
print(f"Wilcoxon signed-rank test: W={w_stat:.0f}, p={w_p:.4f}")

# Spearman correlation: exploitation_score vs views_per_day
rho_vpd, p_vpd = stats.spearmanr(scored['exploitation_score'], scored['views_per_day'])
print(f"\nSpearman ρ (exploitation_score vs views_per_day): {rho_vpd:.4f}, p={p_vpd:.2e}")

print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 2: Same-Year Within-Channel Comparison")
print("=" * 70)

# For each channel-year combination, compare high vs low exploitation
channel_year_boosts = []
for (ch, yr), grp in scored.groupby(['channel_short_name', 'pub_year']):
    if pd.isna(yr) or len(grp) < 6:
        continue
    high = grp[grp['high_exploit'] == 1]['viewCount']
    low = grp[grp['high_exploit'] == 0]['viewCount']
    if len(high) >= 3 and len(low) >= 3:
        boost = (high.median() - low.median()) / low.median() * 100 if low.median() > 0 else np.nan
        channel_year_boosts.append({'channel': ch, 'year': int(yr), 'boost_pct': boost,
                                     'n_high': len(high), 'n_low': len(low)})

df_cy = pd.DataFrame(channel_year_boosts).dropna()
print(f"\nChannel-year groups with valid data: {len(df_cy)}")
print(f"Mean boost (same-year): {df_cy['boost_pct'].mean():.1f}%")
print(f"Median boost (same-year): {df_cy['boost_pct'].median():.1f}%")
print(f"Groups with positive boost: {(df_cy['boost_pct'] > 0).sum()}/{len(df_cy)} ({(df_cy['boost_pct'] > 0).mean()*100:.0f}%)")

# One-sample t-test
t_stat2, p_val2 = stats.ttest_1samp(df_cy['boost_pct'], 0)
print(f"One-sample t-test (boost > 0): t={t_stat2:.3f}, p={p_val2:.4f}")

# Wilcoxon
w_stat2, w_p2 = stats.wilcoxon(df_cy['boost_pct'])
print(f"Wilcoxon signed-rank test: W={w_stat2:.0f}, p={w_p2:.4f}")

print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 3: Per-Dimension Views/Day Analysis")
print("=" * 70)

for dim in dimensions:
    if dim not in scored.columns:
        continue
    has_dim = scored[scored[dim] == 1]['views_per_day']
    no_dim = scored[scored[dim] == 0]['views_per_day']
    
    if len(has_dim) < 5:
        print(f"  {dim}: too few samples ({len(has_dim)})")
        continue
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(has_dim, no_dim, alternative='greater')
    median_boost = (has_dim.median() - no_dim.median()) / no_dim.median() * 100 if no_dim.median() > 0 else np.nan
    
    print(f"  {dim}: n={len(has_dim)}, median boost={median_boost:+.1f}%, Mann-Whitney p={u_p:.4f}")

print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 4: Bonferroni & FDR Correction on Within-Channel Analysis")
print("=" * 70)

# Per-dimension within-channel boost with proper testing
dim_results = []
for dim in dimensions:
    if dim not in scored.columns:
        continue
    
    ch_boosts = []
    for ch, grp in scored.groupby('channel_short_name'):
        has = grp[grp[dim] == 1]['viewCount']
        no = grp[grp[dim] == 0]['viewCount']
        if len(has) >= 3 and len(no) >= 3:
            boost = (has.median() - no.median()) / no.median() * 100 if no.median() > 0 else np.nan
            ch_boosts.append(boost)
    
    ch_boosts = [b for b in ch_boosts if not np.isnan(b)]
    if len(ch_boosts) < 5:
        dim_results.append({'dimension': dim, 'n_channels': len(ch_boosts), 
                           'mean_boost': np.nan, 'p_raw': np.nan})
        continue
    
    t_s, p_raw = stats.ttest_1samp(ch_boosts, 0)
    dim_results.append({
        'dimension': dim,
        'n_channels': len(ch_boosts),
        'mean_boost': np.mean(ch_boosts),
        'median_boost': np.median(ch_boosts),
        'p_raw': p_raw
    })

df_dim = pd.DataFrame(dim_results)

# Bonferroni correction
n_tests = len(df_dim)
df_dim['p_bonferroni'] = (df_dim['p_raw'] * n_tests).clip(upper=1.0)

# FDR (Benjamini-Hochberg) correction
from statsmodels.stats.multitest import multipletests
reject, p_fdr, _, _ = multipletests(df_dim['p_raw'].fillna(1), method='fdr_bh')
df_dim['p_fdr'] = p_fdr
df_dim['sig_fdr'] = reject

print(f"\n{'Dimension':<22} {'N_ch':>4} {'Mean%':>8} {'Med%':>8} {'p_raw':>10} {'p_Bonf':>10} {'p_FDR':>10} {'Sig':>5}")
print("-" * 80)
for _, row in df_dim.iterrows():
    sig = "***" if row['p_fdr'] < 0.001 else "**" if row['p_fdr'] < 0.01 else "*" if row['p_fdr'] < 0.05 else "n.s."
    print(f"  {row['dimension']:<20} {row['n_channels']:>4} {row['mean_boost']:>+8.1f} {row['median_boost']:>+8.1f} {row['p_raw']:>10.4f} {row['p_bonferroni']:>10.4f} {row['p_fdr']:>10.4f} {sig:>5}")

print("\n" + "=" * 70)
print("ROBUSTNESS CHECK 5: Per-Dimension Views/Day Within-Channel (Age-Controlled)")
print("=" * 70)

dim_results_vpd = []
for dim in dimensions:
    if dim not in scored.columns:
        continue
    
    ch_boosts = []
    for ch, grp in scored.groupby('channel_short_name'):
        has = grp[grp[dim] == 1]['views_per_day']
        no = grp[grp[dim] == 0]['views_per_day']
        if len(has) >= 3 and len(no) >= 3:
            boost = (has.median() - no.median()) / no.median() * 100 if no.median() > 0 else np.nan
            ch_boosts.append(boost)
    
    ch_boosts = [b for b in ch_boosts if not np.isnan(b)]
    if len(ch_boosts) < 5:
        dim_results_vpd.append({'dimension': dim, 'n_channels': len(ch_boosts), 
                               'mean_boost': np.nan, 'p_raw': np.nan})
        continue
    
    t_s, p_raw = stats.ttest_1samp(ch_boosts, 0)
    dim_results_vpd.append({
        'dimension': dim,
        'n_channels': len(ch_boosts),
        'mean_boost': np.mean(ch_boosts),
        'median_boost': np.median(ch_boosts),
        'p_raw': p_raw
    })

df_dim_vpd = pd.DataFrame(dim_results_vpd)
n_tests2 = len(df_dim_vpd)
df_dim_vpd['p_bonferroni'] = (df_dim_vpd['p_raw'] * n_tests2).clip(upper=1.0)
reject2, p_fdr2, _, _ = multipletests(df_dim_vpd['p_raw'].fillna(1), method='fdr_bh')
df_dim_vpd['p_fdr'] = p_fdr2
df_dim_vpd['sig_fdr'] = reject2

print(f"\n{'Dimension':<22} {'N_ch':>4} {'Mean%':>8} {'Med%':>8} {'p_raw':>10} {'p_Bonf':>10} {'p_FDR':>10} {'Sig':>5}")
print("-" * 80)
for _, row in df_dim_vpd.iterrows():
    sig = "***" if row['p_fdr'] < 0.001 else "**" if row['p_fdr'] < 0.01 else "*" if row['p_fdr'] < 0.05 else "n.s."
    print(f"  {row['dimension']:<20} {row['n_channels']:>4} {row['mean_boost']:>+8.1f} {row['median_boost']:>+8.1f} {row['p_raw']:>10.4f} {row['p_bonferroni']:>10.4f} {row['p_fdr']:>10.4f} {sig:>5}")

# Save summary
summary = {
    'robustness_views_per_day': {
        'spearman_rho': rho_vpd,
        'spearman_p': p_vpd,
        'within_channel_mean_boost': df_vpd['boost_pct'].mean(),
        'within_channel_median_boost': df_vpd['boost_pct'].median(),
        'pct_positive': (df_vpd['boost_pct'] > 0).mean() * 100,
        'wilcoxon_p': w_p
    },
    'robustness_same_year': {
        'n_channel_year_groups': len(df_cy),
        'mean_boost': df_cy['boost_pct'].mean(),
        'median_boost': df_cy['boost_pct'].median(),
        'pct_positive': (df_cy['boost_pct'] > 0).mean() * 100,
        'wilcoxon_p': w_p2
    }
}

import json
with open('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v3/robustness_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n\nResults saved to analysis_discovery/snorkel_results_v3/robustness_results.json")
print("\nDONE.")
