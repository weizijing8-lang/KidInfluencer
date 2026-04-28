"""
Causal Analysis V3: Multiple Identification Strategies
========================================================
1. Interrupted Time Series (ITS): Long-term exploitation drift trends
2. Enhanced DiD with continuous treatment intensity
3. Cumulative viral exposure model
4. LLM annotation robustness check (sampled titles)

Key insight: The "ratchet" mechanism may be gradual, not triggered by single events.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
RESULTS_V2 = DATA_DIR / "results_v2"
RESULTS_V3 = DATA_DIR / "results_v3"
RESULTS_V3.mkdir(parents=True, exist_ok=True)


def load_v2_data():
    """Load the V2 results with exploitation scores."""
    df = pd.read_csv(RESULTS_V2 / "full_results_v2.csv")
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    print(f"Loaded {len(df):,} videos from {df['channel_short_name'].nunique()} channels")
    return df


# ============================================================
# STRATEGY 1: Interrupted Time Series (ITS)
# ============================================================

def its_analysis(df):
    """
    ITS: For each channel, fit a linear trend to exploitation score over time.
    Compare trend slopes between family and adult channels.
    
    If ratchet exists: family channels should have POSITIVE slopes
    (increasing exploitation over time), while adult channels should be flat.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 1: Interrupted Time Series (ITS)")
    print("=" * 60)
    
    channel_trends = []
    
    for channel, group in df.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        if len(group) < 50:  # Need enough data for trend
            continue
        
        category = group['channel_category'].iloc[0]
        
        # Time variable: video sequence number (normalized to [0, 1])
        t = np.arange(len(group)) / len(group)
        y = group['exploit_score_v2'].values
        
        # Linear regression: exploit_score = α + β*t
        X = sm.add_constant(t)
        model = sm.OLS(y, X).fit()
        
        slope = model.params[1]
        slope_p = model.pvalues[1]
        
        # Also compute by halves (early vs late career)
        mid = len(group) // 2
        early_mean = group.iloc[:mid]['exploit_score_v2'].mean()
        late_mean = group.iloc[mid:]['exploit_score_v2'].mean()
        
        channel_trends.append({
            'channel': channel,
            'category': category,
            'is_family': 1 if category == 'family' else 0,
            'n_videos': len(group),
            'slope': slope,
            'slope_p': slope_p,
            'slope_significant': slope_p < 0.05,
            'early_mean': early_mean,
            'late_mean': late_mean,
            'drift_early_to_late': late_mean - early_mean,
            'mean_exploit': group['exploit_score_v2'].mean(),
        })
    
    trends_df = pd.DataFrame(channel_trends)
    
    # Compare slopes
    family_slopes = trends_df[trends_df['is_family'] == 1]['slope']
    adult_slopes = trends_df[trends_df['is_family'] == 0]['slope']
    
    t_stat, p_val = stats.ttest_ind(family_slopes, adult_slopes)
    
    print(f"\n  Channel trend slopes:")
    print(f"  Family: mean={family_slopes.mean():.6f}, median={family_slopes.median():.6f}")
    print(f"  Adult:  mean={adult_slopes.mean():.6f}, median={adult_slopes.median():.6f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # How many channels have significant positive slopes?
    fam_pos_sig = trends_df[(trends_df['is_family']==1) & (trends_df['slope']>0) & (trends_df['slope_significant'])].shape[0]
    fam_neg_sig = trends_df[(trends_df['is_family']==1) & (trends_df['slope']<0) & (trends_df['slope_significant'])].shape[0]
    fam_total = trends_df[trends_df['is_family']==1].shape[0]
    
    adu_pos_sig = trends_df[(trends_df['is_family']==0) & (trends_df['slope']>0) & (trends_df['slope_significant'])].shape[0]
    adu_neg_sig = trends_df[(trends_df['is_family']==0) & (trends_df['slope']<0) & (trends_df['slope_significant'])].shape[0]
    adu_total = trends_df[trends_df['is_family']==0].shape[0]
    
    print(f"\n  Significant trends (p<0.05):")
    print(f"  Family ({fam_total} channels): {fam_pos_sig} increasing, {fam_neg_sig} decreasing")
    print(f"  Adult  ({adu_total} channels): {adu_pos_sig} increasing, {adu_neg_sig} decreasing")
    
    # Early vs Late comparison
    family_drift = trends_df[trends_df['is_family'] == 1]['drift_early_to_late']
    adult_drift = trends_df[trends_df['is_family'] == 0]['drift_early_to_late']
    t2, p2 = stats.ttest_ind(family_drift, adult_drift)
    
    print(f"\n  Early-to-Late career drift:")
    print(f"  Family: mean={family_drift.mean():.6f}")
    print(f"  Adult:  mean={adult_drift.mean():.6f}")
    print(f"  t-test: t={t2:.3f}, p={p2:.4f}")
    
    # Print individual channel trends for family
    print(f"\n  Family channel trends (sorted by slope):")
    fam_trends = trends_df[trends_df['is_family']==1].sort_values('slope', ascending=False)
    for _, r in fam_trends.iterrows():
        sig = "*" if r['slope_significant'] else " "
        print(f"    {r['channel']:25s}: slope={r['slope']:+.4f}{sig}  "
              f"early={r['early_mean']:.4f} → late={r['late_mean']:.4f}  (n={r['n_videos']})")
    
    trends_df.to_csv(RESULTS_V3 / "its_channel_trends.csv", index=False)
    return trends_df


# ============================================================
# STRATEGY 2: Cumulative Viral Exposure Model
# ============================================================

def cumulative_viral_model(df):
    """
    Instead of looking at individual viral hits, compute cumulative viral exposure
    and test whether channels with MORE viral hits show MORE exploitation drift.
    
    For each channel: cumulative_viral_count at time t = number of viral hits before t
    Then: exploit_score_t ~ cumulative_viral_t + channel_FE + time_FE
    """
    print("\n" + "=" * 60)
    print("STRATEGY 2: Cumulative Viral Exposure Model")
    print("=" * 60)
    
    # Identify viral hits per channel
    channel_stats = df.groupby('channel_short_name')['viewCount'].agg(['mean', 'std']).reset_index()
    channel_stats.columns = ['channel_short_name', 'ch_mean', 'ch_std']
    df = df.merge(channel_stats, on='channel_short_name', how='left')
    df['is_viral'] = df['viewCount'] > (df['ch_mean'] + 2 * df['ch_std'])
    
    # For each video, compute cumulative viral count up to that point
    results = []
    for channel, group in df.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        if len(group) < 50:
            continue
        
        category = group['channel_category'].iloc[0]
        cum_viral = group['is_viral'].cumsum()
        
        # Normalize by video index to get viral rate
        video_idx = np.arange(1, len(group) + 1)
        viral_rate = cum_viral / video_idx
        
        # Split into quartiles of cumulative viral exposure
        group['cum_viral'] = cum_viral.values
        group['video_idx'] = video_idx
        group['viral_rate'] = viral_rate.values
        group['is_family'] = 1 if category == 'family' else 0
        
        results.append(group[['channel_short_name', 'channel_category', 'is_family',
                              'exploit_score_v2', 'cum_viral', 'video_idx', 'viral_rate',
                              'viewCount', 'publishedAt']])
    
    panel_df = pd.concat(results, ignore_index=True)
    
    # Panel regression: exploit_score ~ viral_rate * is_family
    print(f"\n  Panel data: {len(panel_df):,} observations")
    
    # Interaction model
    panel_df['viral_rate_x_family'] = panel_df['viral_rate'] * panel_df['is_family']
    panel_df['log_video_idx'] = np.log1p(panel_df['video_idx'])
    
    X = panel_df[['is_family', 'viral_rate', 'viral_rate_x_family', 'log_video_idx']].copy()
    X = sm.add_constant(X)
    y = panel_df['exploit_score_v2']
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': panel_df['channel_short_name']})
    
    print(f"\n  --- Panel Regression (clustered SE by channel) ---")
    print(model.summary2().tables[1].to_string())
    
    # Key coefficient: viral_rate_x_family
    coef = model.params['viral_rate_x_family']
    pval = model.pvalues['viral_rate_x_family']
    print(f"\n  Key: β(viral_rate × is_family) = {coef:.6f}, p = {pval:.4f}")
    if pval < 0.05:
        print(f"  *** SIGNIFICANT: Family channels' exploitation increases MORE with viral exposure ***")
    
    with open(RESULTS_V3 / "cumulative_viral_regression.txt", 'w') as f:
        f.write(str(model.summary2()))
    
    return panel_df, model


# ============================================================
# STRATEGY 3: Views-Weighted Exploitation (Engagement Ratchet)
# ============================================================

def engagement_ratchet_analysis(df):
    """
    Test the "engagement ratchet" hypothesis directly:
    Do videos with HIGHER exploitation scores get MORE views (within family channels)?
    And is this relationship STRONGER for family channels than adult channels?
    
    This is the mechanism test: if exploitation is rewarded with views,
    that creates the incentive for the ratchet.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 3: Engagement Ratchet (Views ~ Exploitation)")
    print("=" * 60)
    
    # Per-channel: correlation between exploit_score and log(views)
    correlations = []
    for channel, group in df.groupby('channel_short_name'):
        if len(group) < 50:
            continue
        category = group['channel_category'].iloc[0]
        
        log_views = np.log1p(group['viewCount'])
        exploit = group['exploit_score_v2']
        
        r, p = stats.pearsonr(exploit, log_views)
        
        correlations.append({
            'channel': channel,
            'category': category,
            'is_family': 1 if category == 'family' else 0,
            'n_videos': len(group),
            'corr_exploit_views': r,
            'corr_p': p,
            'significant': p < 0.05,
        })
    
    corr_df = pd.DataFrame(correlations)
    
    family_corr = corr_df[corr_df['is_family'] == 1]['corr_exploit_views']
    adult_corr = corr_df[corr_df['is_family'] == 0]['corr_exploit_views']
    
    t_stat, p_val = stats.ttest_ind(family_corr, adult_corr)
    
    print(f"\n  Correlation between exploitation score and log(views):")
    print(f"  Family channels: mean r = {family_corr.mean():.4f} (median {family_corr.median():.4f})")
    print(f"  Adult channels:  mean r = {adult_corr.mean():.4f} (median {adult_corr.median():.4f})")
    print(f"  Difference test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # How many channels show significant POSITIVE correlation?
    fam_pos = corr_df[(corr_df['is_family']==1) & (corr_df['corr_exploit_views']>0) & (corr_df['significant'])].shape[0]
    fam_neg = corr_df[(corr_df['is_family']==1) & (corr_df['corr_exploit_views']<0) & (corr_df['significant'])].shape[0]
    adu_pos = corr_df[(corr_df['is_family']==0) & (corr_df['corr_exploit_views']>0) & (corr_df['significant'])].shape[0]
    adu_neg = corr_df[(corr_df['is_family']==0) & (corr_df['corr_exploit_views']<0) & (corr_df['significant'])].shape[0]
    
    print(f"\n  Channels with significant (p<0.05) correlation:")
    print(f"  Family: {fam_pos} positive, {fam_neg} negative (out of {len(family_corr)})")
    print(f"  Adult:  {adu_pos} positive, {adu_neg} negative (out of {len(adult_corr)})")
    
    print(f"\n  Family channels detail:")
    fam_corr_df = corr_df[corr_df['is_family']==1].sort_values('corr_exploit_views', ascending=False)
    for _, r in fam_corr_df.iterrows():
        sig = "*" if r['significant'] else " "
        print(f"    {r['channel']:25s}: r={r['corr_exploit_views']:+.4f}{sig} (n={r['n_videos']})")
    
    corr_df.to_csv(RESULTS_V3 / "engagement_ratchet_correlations.csv", index=False)
    return corr_df


# ============================================================
# STRATEGY 4: Granger-style Lag Analysis
# ============================================================

def granger_lag_analysis(df):
    """
    Test: Do high-view videos PREDICT increased exploitation in subsequent videos?
    For each channel, compute: exploit_score_t ~ exploit_score_{t-1} + log_views_{t-1}
    
    If views Granger-cause exploitation: β(log_views_{t-1}) > 0 for family channels.
    """
    print("\n" + "=" * 60)
    print("STRATEGY 4: Granger-style Lag Analysis")
    print("=" * 60)
    
    lag_results = []
    
    for channel, group in df.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        if len(group) < 100:
            continue
        
        category = group['channel_category'].iloc[0]
        
        # Create lagged variables
        y = group['exploit_score_v2'].iloc[1:].values
        exploit_lag = group['exploit_score_v2'].iloc[:-1].values
        views_lag = np.log1p(group['viewCount'].iloc[:-1].values)
        
        X = np.column_stack([exploit_lag, views_lag])
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        lag_results.append({
            'channel': channel,
            'category': category,
            'is_family': 1 if category == 'family' else 0,
            'n_videos': len(group),
            'beta_exploit_lag': model.params[1],
            'beta_views_lag': model.params[2],
            'views_lag_p': model.pvalues[2],
            'views_lag_significant': model.pvalues[2] < 0.05,
        })
    
    lag_df = pd.DataFrame(lag_results)
    
    family_beta = lag_df[lag_df['is_family'] == 1]['beta_views_lag']
    adult_beta = lag_df[lag_df['is_family'] == 0]['beta_views_lag']
    
    t_stat, p_val = stats.ttest_ind(family_beta, adult_beta)
    
    print(f"\n  β(lagged_views → exploitation):")
    print(f"  Family: mean β = {family_beta.mean():.6f}")
    print(f"  Adult:  mean β = {adult_beta.mean():.6f}")
    print(f"  Difference: t={t_stat:.3f}, p={p_val:.4f}")
    
    # How many family channels show views → exploitation?
    fam_pos = lag_df[(lag_df['is_family']==1) & (lag_df['beta_views_lag']>0) & (lag_df['views_lag_significant'])].shape[0]
    fam_total = lag_df[lag_df['is_family']==1].shape[0]
    print(f"\n  Family channels with significant views→exploitation: {fam_pos}/{fam_total}")
    
    print(f"\n  Family channels detail:")
    fam_lag = lag_df[lag_df['is_family']==1].sort_values('beta_views_lag', ascending=False)
    for _, r in fam_lag.iterrows():
        sig = "*" if r['views_lag_significant'] else " "
        print(f"    {r['channel']:25s}: β(views_lag)={r['beta_views_lag']:+.6f}{sig}")
    
    lag_df.to_csv(RESULTS_V3 / "granger_lag_results.csv", index=False)
    return lag_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CAUSAL ANALYSIS V3: MULTIPLE IDENTIFICATION STRATEGIES")
    print("=" * 70)
    
    df = load_v2_data()
    
    # Strategy 1: ITS
    its_df = its_analysis(df)
    
    # Strategy 2: Cumulative Viral Exposure
    panel_df, cum_model = cumulative_viral_model(df)
    
    # Strategy 3: Engagement Ratchet
    corr_df = engagement_ratchet_analysis(df)
    
    # Strategy 4: Granger Lag
    lag_df = granger_lag_analysis(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL STRATEGIES")
    print("=" * 70)
    
    print("""
    Strategy 1 (ITS - Long-term trends):
      → Do family channels show increasing exploitation over time?
    
    Strategy 2 (Cumulative Viral):
      → Does viral exposure amplify exploitation more for family channels?
    
    Strategy 3 (Engagement Ratchet):
      → Is exploitation rewarded with more views in family channels?
    
    Strategy 4 (Granger Lag):
      → Do high-view videos predict increased exploitation in next videos?
    """)
    
    print("All results saved to:", RESULTS_V3)


if __name__ == "__main__":
    main()
