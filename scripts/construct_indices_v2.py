"""
Construct Commercialization Index (CI) and Labor Intensity Index (LII) - V2
Key improvements:
1. Use sponsorship data from V4 (description-level NLP detection)
2. Include brand collaboration network data
3. Use V4 full dataset (98k videos) for more robust channel-level estimates
4. Include adult creator control group for comparison
5. Use PCA for index construction instead of equal weights
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/ubuntu/KidInfluencer'
OUTPUT = f'{BASE}/analysis_paper1_v2'
os.makedirs(OUTPUT, exist_ok=True)

# ============================================================
# 1. LOAD AND MERGE ALL DATA
# ============================================================
print("=" * 60)
print("PHASE 1: LOADING ALL DATA SOURCES")
print("=" * 60)

# Upload frequency (includes both family and adult channels)
uf = pd.read_csv(f'{BASE}/data/results_v4/upload_frequency_metrics.csv')
print(f"Upload Frequency: {uf.shape} | Categories: {uf['category'].value_counts().to_dict()}")

# Sponsorship data
spon = pd.read_csv(f'{BASE}/data/results_v4/sponsorship_by_channel.csv')
print(f"Sponsorship: {spon.shape}")

# Network centrality
net = pd.read_csv(f'{BASE}/data/results_v4/network_centrality.csv')
print(f"Network Centrality: {net.shape}")

# V4 channel summary (exploit scores)
v4_ch = pd.read_csv(f'{BASE}/data/results_v4/channel_summary_v4.csv')
print(f"V4 Channel Summary: {v4_ch.shape}")

# Full V4 results for temporal analysis
v4_full = pd.read_csv(f'{BASE}/data/results_v4/full_results_v4.csv',
                       usecols=['id', 'channel_short_name', 'channel_category',
                                'viewCount', 'likeCount', 'commentCount',
                                'exploit_score_v4', 'publishedAt'])
v4_full = v4_full[v4_full['channel_category'].isin(['adult', 'family'])]
print(f"V4 Full (filtered): {v4_full.shape}")

# Composite exploitation index (family channels only)
cei = pd.read_csv(f'{BASE}/data/results_v4/composite_exploitation_index.csv')
print(f"Composite Exploitation: {cei.shape}")

# Within-family dimensions
wfd = pd.read_csv(f'{BASE}/data/results_v4/within_family_dimensions.csv', index_col=0)
print(f"Within-Family Dimensions: {wfd.shape}")

# ============================================================
# 2. BUILD UNIFIED CHANNEL DATASET
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: BUILDING UNIFIED CHANNEL DATASET")
print("=" * 60)

# Start with upload frequency as base (has both family and adult)
channels = uf[['channel', 'category', 'n_videos', 'span_days', 'span_years',
               'videos_per_week', 'videos_per_month', 'median_interval_days',
               'mean_duration_min', 'total_content_hours', 'monthly_content_hours',
               'weekly_production_hours_est', 'freq_first_half_per_week',
               'freq_second_half_per_week', 'freq_change_pct']].copy()

# Merge sponsorship
channels = channels.merge(spon[['channel', 'n_sponsored', 'sponsor_rate', 'n_child_brands']],
                          on='channel', how='left')

# Merge network centrality
channels = channels.merge(net[['channel', 'degree', 'family_partners', 'adult_partners']],
    on='channel', how='left')

# Merge V4 channel summary (exploit scores)
channels = channels.merge(v4_ch.rename(columns={'channel_short_name': 'channel'})[
    ['channel', 'mean_exploit_v4', 'std_exploit_v4', 'total_views']],
    on='channel', how='left')

# Fill NaN for network features (channels not in network = 0 connections)
for col in ['degree', 'family_partners', 'adult_partners']:
    channels[col] = channels[col].fillna(0)
channels['n_sponsored'] = channels['n_sponsored'].fillna(0)
channels['sponsor_rate'] = channels['sponsor_rate'].fillna(0)
channels['n_child_brands'] = channels['n_child_brands'].fillna(0)

print(f"Unified channel dataset: {channels.shape}")
print(f"Family: {(channels['category']=='family').sum()}, Adult: {(channels['category']=='adult').sum()}")

# ============================================================
# 3. CONSTRUCT COMMERCIALIZATION INDEX (CI) - REVISED
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: COMMERCIALIZATION INDEX (REVISED)")
print("=" * 60)

"""
Revised CI components (all measurable from metadata/NLP, no LLM needed):
1. sponsor_rate: % of videos with sponsorship mentions in description
2. n_child_brands: Number of unique child-oriented brands mentioned
3. total_views: Total channel views (proxy for ad revenue potential)
4. degree: Network connections (brand collaboration network size)
5. n_family_connections: Connections to other family channels (industry integration)
"""

ci_components = ['sponsor_rate', 'n_child_brands', 'total_views', 'degree', 'n_family_connections']

# Log-transform skewed features
channels['log_total_views'] = np.log10(channels['total_views'].clip(lower=1))
channels['log_n_child_brands'] = np.log1p(channels['n_child_brands'])

ci_features_transformed = ['sponsor_rate', 'log_n_child_brands', 'log_total_views', 
                           'degree', 'family_partners']

# Normalize to [0,1]
scaler_ci = MinMaxScaler()
ci_matrix = channels[ci_features_transformed].values
ci_normalized = scaler_ci.fit_transform(ci_matrix)

# Use PCA to find optimal weighting
pca_ci = PCA(n_components=1)
ci_pca_scores = pca_ci.fit_transform(ci_normalized)
channels['ci_pca'] = MinMaxScaler().fit_transform(ci_pca_scores)

# Also compute equal-weight version for comparison
channels['ci_equal'] = ci_normalized.mean(axis=1)

print(f"PCA explained variance: {pca_ci.explained_variance_ratio_[0]:.4f}")
print(f"PCA loadings: {dict(zip(ci_features_transformed, pca_ci.components_[0].round(3)))}")
print(f"\nCI (PCA) by category:")
print(channels.groupby('category')['ci_pca'].describe().to_string())

# ============================================================
# 4. CONSTRUCT LABOR INTENSITY INDEX (LII) - REVISED
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: LABOR INTENSITY INDEX (REVISED)")
print("=" * 60)

"""
Revised LII components:
1. videos_per_week: Upload frequency (more videos = more filming sessions)
2. mean_duration_min: Average video length (longer = more time on camera)
3. weekly_production_hours_est: Estimated weekly production hours
4. freq_change_pct: Whether frequency is escalating over time
5. mean_exploit_v4: Content exploitation score (emotional/clickbait intensity)
"""

lii_features = ['videos_per_week', 'mean_duration_min', 'weekly_production_hours_est',
                'mean_exploit_v4']

# Handle missing exploit scores
channels['mean_exploit_v4'] = channels['mean_exploit_v4'].fillna(0)

# Normalize
scaler_lii = MinMaxScaler()
lii_matrix = channels[lii_features].values
lii_normalized = scaler_lii.fit_transform(lii_matrix)

# PCA
pca_lii = PCA(n_components=1)
lii_pca_scores = pca_lii.fit_transform(lii_normalized)
channels['lii_pca'] = MinMaxScaler().fit_transform(lii_pca_scores)

# Equal weight
channels['lii_equal'] = lii_normalized.mean(axis=1)

print(f"PCA explained variance: {pca_lii.explained_variance_ratio_[0]:.4f}")
print(f"PCA loadings: {dict(zip(lii_features, pca_lii.components_[0].round(3)))}")
print(f"\nLII (PCA) by category:")
print(channels.groupby('category')['lii_pca'].describe().to_string())

# ============================================================
# 5. KEY ANALYSIS: CI → LII RELATIONSHIP
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: CI → LII RELATIONSHIP ANALYSIS")
print("=" * 60)

# Overall correlation
r_all, p_all = stats.pearsonr(channels['ci_pca'], channels['lii_pca'])
rho_all, prho_all = stats.spearmanr(channels['ci_pca'], channels['lii_pca'])
print(f"ALL CHANNELS (n={len(channels)}):")
print(f"  Pearson r = {r_all:.4f}, p = {p_all:.6f}")
print(f"  Spearman rho = {rho_all:.4f}, p = {prho_all:.6f}")

# Family channels only
family = channels[channels['category'] == 'family']
r_fam, p_fam = stats.pearsonr(family['ci_pca'], family['lii_pca'])
rho_fam, prho_fam = stats.spearmanr(family['ci_pca'], family['lii_pca'])
print(f"\nFAMILY CHANNELS ONLY (n={len(family)}):")
print(f"  Pearson r = {r_fam:.4f}, p = {p_fam:.6f}")
print(f"  Spearman rho = {rho_fam:.4f}, p = {prho_fam:.6f}")

# Adult channels only
adult = channels[channels['category'] == 'adult']
r_adu, p_adu = stats.pearsonr(adult['ci_pca'], adult['lii_pca'])
rho_adu, prho_adu = stats.spearmanr(adult['ci_pca'], adult['lii_pca'])
print(f"\nADULT CHANNELS ONLY (n={len(adult)}):")
print(f"  Pearson r = {r_adu:.4f}, p = {p_adu:.6f}")
print(f"  Spearman rho = {rho_adu:.4f}, p = {prho_adu:.6f}")

# ============================================================
# 6. FAMILY vs ADULT COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: FAMILY vs ADULT COMPARISON")
print("=" * 60)

comparison_vars = ['videos_per_week', 'mean_duration_min', 'weekly_production_hours_est',
                   'sponsor_rate', 'n_child_brands', 'mean_exploit_v4',
                   'ci_pca', 'lii_pca']

print(f"{'Variable':<30} {'Family Mean':>12} {'Adult Mean':>12} {'U-stat':>10} {'p-value':>10} {'Effect':>8}")
print("-" * 85)

for var in comparison_vars:
    f_vals = family[var].dropna()
    a_vals = adult[var].dropna()
    if len(f_vals) > 0 and len(a_vals) > 0:
        u, p = stats.mannwhitneyu(f_vals, a_vals, alternative='two-sided')
        # Cohen's d
        pooled_std = np.sqrt((f_vals.std()**2 + a_vals.std()**2) / 2)
        d = (f_vals.mean() - a_vals.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '*' if p < 0.05 else ''
        print(f"{var:<30} {f_vals.mean():>12.4f} {a_vals.mean():>12.4f} {u:>10.1f} {p:>10.4f} {d:>7.3f}{sig}")

# ============================================================
# 7. TEMPORAL ANALYSIS: FREQUENCY ESCALATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: FREQUENCY ESCALATION ANALYSIS")
print("=" * 60)

# freq_change_pct: positive = frequency increased over time
print("Frequency escalation (freq_change_pct):")
print(f"  Family: mean={family['freq_change_pct'].mean():.2f}%, median={family['freq_change_pct'].median():.2f}%")
print(f"  Adult: mean={adult['freq_change_pct'].mean():.2f}%, median={adult['freq_change_pct'].median():.2f}%")

# % of channels that increased frequency
fam_escalated = (family['freq_change_pct'] > 0).mean()
adu_escalated = (adult['freq_change_pct'] > 0).mean()
print(f"  Family channels with increasing frequency: {fam_escalated:.1%}")
print(f"  Adult channels with increasing frequency: {adu_escalated:.1%}")

u, p = stats.mannwhitneyu(family['freq_change_pct'].dropna(), 
                           adult['freq_change_pct'].dropna(), 
                           alternative='two-sided')
print(f"  Mann-Whitney U: U={u:.1f}, p={p:.4f}")

# ============================================================
# 8. WITHIN-FAMILY: HIGH vs LOW COMMERCIALIZATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 8: WITHIN-FAMILY HIGH vs LOW CI COMPARISON")
print("=" * 60)

median_ci_fam = family['ci_pca'].median()
high_ci_fam = family[family['ci_pca'] >= median_ci_fam]
low_ci_fam = family[family['ci_pca'] < median_ci_fam]

print(f"High CI family channels: n={len(high_ci_fam)}")
print(f"Low CI family channels: n={len(low_ci_fam)}")

labor_vars = ['videos_per_week', 'mean_duration_min', 'weekly_production_hours_est',
              'mean_exploit_v4', 'lii_pca']

print(f"\n{'Variable':<30} {'High CI Mean':>12} {'Low CI Mean':>12} {'U-stat':>10} {'p-value':>10}")
print("-" * 75)

for var in labor_vars:
    h = high_ci_fam[var].dropna()
    l = low_ci_fam[var].dropna()
    if len(h) > 0 and len(l) > 0:
        u, p = stats.mannwhitneyu(h, l, alternative='two-sided')
        print(f"{var:<30} {h.mean():>12.4f} {l.mean():>12.4f} {u:>10.1f} {p:>10.4f}")

# ============================================================
# 9. ENGAGEMENT ANALYSIS: DOES LABOR PAY OFF?
# ============================================================
print("\n" + "=" * 60)
print("PHASE 9: DOES MORE LABOR = MORE VIEWS?")
print("=" * 60)

# For family channels: correlation between LII and total views
r_labor_views, p_labor_views = stats.pearsonr(
    family['lii_pca'].dropna(), 
    family.loc[family['lii_pca'].notna(), 'log_total_views'].dropna()
)
print(f"Family: LII vs log(total_views): r={r_labor_views:.4f}, p={p_labor_views:.6f}")

# For adult channels
r_labor_views_a, p_labor_views_a = stats.pearsonr(
    adult['lii_pca'].dropna(),
    adult.loc[adult['lii_pca'].notna(), 'log_total_views'].dropna()
)
print(f"Adult: LII vs log(total_views): r={r_labor_views_a:.4f}, p={p_labor_views_a:.6f}")

# CI → Views
r_ci_views, p_ci_views = stats.pearsonr(family['ci_pca'], family['log_total_views'])
print(f"\nFamily: CI vs log(total_views): r={r_ci_views:.4f}, p={p_ci_views:.6f}")

r_ci_views_a, p_ci_views_a = stats.pearsonr(adult['ci_pca'], adult['log_total_views'])
print(f"Adult: CI vs log(total_views): r={r_ci_views_a:.4f}, p={p_ci_views_a:.6f}")

# ============================================================
# 10. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 10: SAVING RESULTS")
print("=" * 60)

channels.to_csv(f'{OUTPUT}/channels_with_indices_v2.csv', index=False)
print(f"Saved: channels_with_indices_v2.csv ({channels.shape})")

# Save summary
summary = {
    'n_channels_total': len(channels),
    'n_family': len(family),
    'n_adult': len(adult),
    'ci_pca_variance_explained': float(pca_ci.explained_variance_ratio_[0]),
    'lii_pca_variance_explained': float(pca_lii.explained_variance_ratio_[0]),
    'ci_lii_correlation': {
        'all': {'r': float(r_all), 'p': float(p_all), 'rho': float(rho_all)},
        'family': {'r': float(r_fam), 'p': float(p_fam), 'rho': float(rho_fam)},
        'adult': {'r': float(r_adu), 'p': float(p_adu), 'rho': float(rho_adu)},
    },
    'family_vs_adult': {
        'videos_per_week': {'family': float(family['videos_per_week'].mean()),
                           'adult': float(adult['videos_per_week'].mean())},
        'mean_duration_min': {'family': float(family['mean_duration_min'].mean()),
                             'adult': float(adult['mean_duration_min'].mean())},
        'exploit_score': {'family': float(family['mean_exploit_v4'].mean()),
                         'adult': float(adult['mean_exploit_v4'].mean())},
    },
    'labor_views_correlation': {
        'family': {'r': float(r_labor_views), 'p': float(p_labor_views)},
        'adult': {'r': float(r_labor_views_a), 'p': float(p_labor_views_a)},
    },
    'frequency_escalation': {
        'family_pct_escalated': float(fam_escalated),
        'adult_pct_escalated': float(adu_escalated),
    }
}

with open(f'{OUTPUT}/analysis_summary_v2.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: analysis_summary_v2.json")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
