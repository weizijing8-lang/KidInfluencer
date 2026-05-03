#!/usr/bin/env python3
"""
Kidfluencer Dataset Analysis V2
================================
- Improved commercial detection (description analysis)
- Regression models (what predicts higher views?)
- Channel-level risk scoring
- Statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
import re

# ── Config ──
DATA_DIR = '/home/ubuntu/KidInfluencer/data'
OUT_DIR = '/home/ubuntu/KidInfluencer/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Load Combined Data ──
channels = pd.read_csv(os.path.join(DATA_DIR, 'combined_channels.csv'))
videos = pd.read_csv(os.path.join(DATA_DIR, 'combined_videos.csv'))
print(f"Loaded {len(channels)} channels, {len(videos)} videos")

# ═══════════════════════════════════════════════════════════════
# 1. IMPROVED COMMERCIAL DETECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. IMPROVED COMMERCIAL DETECTION")
print("="*60)

def detect_commercial(row):
    """Enhanced commercial detection using title + description."""
    title = str(row.get('title', '')).lower()
    desc = str(row.get('description_snippet', '')).lower()
    combined = title + ' ' + desc
    
    # Direct ad markers
    direct_markers = ['#ad', '#sponsored', '#paidpartnership', '#brandpartner',
                      'paid partnership', 'sponsored by', 'in partnership with',
                      'brand partner', 'ad |', '| ad']
    
    # Affiliate/promo indicators
    affiliate_markers = ['use code', 'discount code', 'promo code', 'affiliate link',
                        'use my link', 'shop now', 'link below', 'link in description',
                        'check out', 'available at', '% off', 'coupon']
    
    # Brand mention patterns (common in kid content)
    brand_patterns = ['thanks to', 'thank you to', 'shoutout to', 'gifted by',
                     'sent me', 'collab with', 'collaboration with']
    
    # URL patterns suggesting commercial content
    url_patterns = ['amzn.to', 'bit.ly', 'shopify', 'amazon.com', 'walmart.com',
                   'target.com', '.shop', 'store.']
    
    score = 0
    if any(m in combined for m in direct_markers):
        score += 3
    if any(m in combined for m in affiliate_markers):
        score += 2
    if any(m in combined for m in brand_patterns):
        score += 1
    if any(m in combined for m in url_patterns):
        score += 1
    
    return score

videos['commercial_score'] = videos.apply(detect_commercial, axis=1)
videos['is_commercial_v2'] = videos['commercial_score'] >= 2

commercial_count = videos['is_commercial_v2'].sum()
print(f"Commercial videos detected (v2): {commercial_count} ({commercial_count/len(videos)*100:.1f}%)")

# Commercial by channel
videos_with_ch = videos.merge(
    channels[['channel_id', 'title', 'subscribers', 'total_videos', 'total_views', 'cross_platform_count']],
    on='channel_id', how='inner', suffixes=('', '_channel')
)
print(f"Videos merged with channel info: {len(videos_with_ch)}")

commercial_by_ch = videos_with_ch.groupby('channel_id').agg(
    channel_title=('title_channel', 'first'),
    subscribers=('subscribers', 'first'),
    total_vids=('video_id', 'count'),
    commercial_count=('is_commercial_v2', 'sum'),
    avg_commercial_score=('commercial_score', 'mean'),
).reset_index()
commercial_by_ch['commercial_rate'] = commercial_by_ch['commercial_count'] / commercial_by_ch['total_vids']

print("\nTop 10 Most Commercialized Channels:")
top_commercial = commercial_by_ch.nlargest(10, 'commercial_rate')
for _, row in top_commercial.iterrows():
    print(f"  {row['channel_title'][:35]:35s} | {row['commercial_count']:2d}/{row['total_vids']:2d} ({row['commercial_rate']:.0%}) | {row['subscribers']:>12,.0f} subs")

# ═══════════════════════════════════════════════════════════════
# 2. REGRESSION ANALYSIS: What predicts views?
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. REGRESSION ANALYSIS")
print("="*60)

# Prepare regression data
reg_data = videos_with_ch.copy()
reg_data['log_views'] = np.log1p(reg_data['views'])
reg_data['log_subs'] = np.log1p(reg_data['subscribers'])
reg_data['duration_minutes'] = reg_data['length_seconds'] / 60
reg_data['has_emotional_title'] = reg_data['has_emotional_title'].astype(int)
reg_data['is_commercial_v2'] = reg_data['is_commercial_v2'].astype(int)

# Title length
reg_data['title_length'] = reg_data['title'].str.len()

# Simple OLS using numpy (no statsmodels needed)
from numpy.linalg import lstsq

# Features: log_subs, duration_minutes, has_emotional_title, is_commercial_v2, title_length, cross_platform_count
features = ['log_subs', 'duration_minutes', 'has_emotional_title', 'is_commercial_v2', 'title_length', 'cross_platform_count']
X = reg_data[features].fillna(0).values
X = np.column_stack([np.ones(len(X)), X])  # Add intercept
y = reg_data['log_views'].values

# Remove any NaN/inf
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

# OLS
coeffs, residuals, rank, sv = lstsq(X, y, rcond=None)
y_pred = X @ coeffs
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"\nOLS Regression: log(views) ~ features")
print(f"R² = {r_squared:.4f}")
print(f"N = {len(y)}")
print(f"\nCoefficients:")
coeff_names = ['intercept'] + features
for name, coeff in zip(coeff_names, coeffs):
    print(f"  {name:25s}: {coeff:8.4f}")

# Calculate standard errors and t-stats
n = len(y)
p = X.shape[1]
mse = ss_res / (n - p)
var_coeff = mse * np.linalg.inv(X.T @ X).diagonal()
se = np.sqrt(var_coeff)
t_stats = coeffs / se
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))

print(f"\n{'Variable':<25s} {'Coeff':>8s} {'SE':>8s} {'t-stat':>8s} {'p-value':>10s} {'Sig':>5s}")
print("-" * 70)
for name, coeff, s, t, pv in zip(coeff_names, coeffs, se, t_stats, p_values):
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    print(f"  {name:<25s} {coeff:8.4f} {s:8.4f} {t:8.2f} {pv:10.4f} {sig:>5s}")

# ── Fig: Coefficient plot ──
fig, ax = plt.subplots(figsize=(10, 5))
feature_coeffs = coeffs[1:]  # exclude intercept
feature_ses = se[1:]
feature_pvals = p_values[1:]

colors = ['#F44336' if p < 0.05 else '#9E9E9E' for p in feature_pvals]
y_pos = range(len(features))
ax.barh(y_pos, feature_coeffs, xerr=1.96*feature_ses, color=colors, edgecolor='white', alpha=0.8, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient (effect on log views)')
ax.set_title('What Predicts Video Views? (OLS Regression)\nRed = significant (p < 0.05), Gray = not significant')
fig.savefig(os.path.join(OUT_DIR, 'fig9_regression_coefficients.png'))
plt.close()
print("\nSaved fig9_regression_coefficients.png")

# ═══════════════════════════════════════════════════════════════
# 3. CHANNEL-LEVEL RISK SCORING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. CHANNEL-LEVEL RISK SCORING")
print("="*60)

# Compute risk features per channel
risk_features = videos_with_ch.groupby('channel_id').agg(
    channel_title=('title_channel', 'first'),
    subscribers=('subscribers', 'first'),
    total_videos_on_channel=('total_videos', 'first'),
    cross_platform=('cross_platform_count', 'first'),
    # Content intensity
    avg_duration_min=('length_seconds', lambda x: x.mean() / 60),
    # Emotional manipulation
    emotional_rate=('has_emotional_title', 'mean'),
    # Commercialization
    commercial_rate=('is_commercial_v2', 'mean'),
    avg_commercial_score=('commercial_score', 'mean'),
    # Engagement
    avg_views=('views', 'mean'),
    median_views=('views', 'median'),
).reset_index()

# Normalize features to 0-1 for scoring
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0, index=series.index)
    return (series - mn) / (mx - mn)

# Risk score = weighted combination
risk_features['risk_score'] = (
    0.35 * normalize(risk_features['emotional_rate']) +
    0.25 * normalize(risk_features['avg_commercial_score']) +
    0.20 * normalize(risk_features['total_videos_on_channel']) +
    0.10 * normalize(risk_features['cross_platform']) +
    0.10 * normalize(risk_features['avg_duration_min'])
)

risk_features = risk_features.sort_values('risk_score', ascending=False)

print("\nTop 15 Highest Risk Channels:")
print(f"{'Channel':<35s} {'Risk':>6s} {'Emot%':>6s} {'Comm%':>6s} {'Vids':>6s} {'Subs':>12s}")
print("-" * 80)
for _, row in risk_features.head(15).iterrows():
    print(f"  {row['channel_title'][:33]:33s} {row['risk_score']:.3f} {row['emotional_rate']*100:5.1f}% {row['commercial_rate']*100:5.1f}% {row['total_videos_on_channel']:5.0f} {row['subscribers']:>11,.0f}")

# Save risk scores
risk_features.to_csv(os.path.join(OUT_DIR, 'channel_risk_scores.csv'), index=False)
print(f"\nSaved channel_risk_scores.csv")

# ── Fig: Risk Score Distribution ──
fig, ax = plt.subplots()
ax.hist(risk_features['risk_score'], bins=20, color='#E91E63', edgecolor='white', alpha=0.8)
ax.set_xlabel('Risk Score (0-1)')
ax.set_ylabel('Number of Channels')
ax.set_title('Distribution of Channel Risk Scores')
ax.axvline(risk_features['risk_score'].quantile(0.75), color='red', linestyle='--',
           label=f'75th percentile: {risk_features["risk_score"].quantile(0.75):.3f}')
ax.legend()
fig.savefig(os.path.join(OUT_DIR, 'fig10_risk_score_distribution.png'))
plt.close()
print("Saved fig10_risk_score_distribution.png")

# ── Fig: Top 15 Risk Channels ──
fig, ax = plt.subplots(figsize=(12, 7))
top15_risk = risk_features.head(15)
bars = ax.barh(range(len(top15_risk)), top15_risk['risk_score'], color='#E91E63', edgecolor='white')
ax.set_yticks(range(len(top15_risk)))
ax.set_yticklabels(top15_risk['channel_title'].values)
ax.set_xlabel('Risk Score')
ax.set_title('Top 15 Highest-Risk Kidfluencer Channels')
ax.invert_yaxis()
fig.savefig(os.path.join(OUT_DIR, 'fig11_top15_risk_channels.png'))
plt.close()
print("Saved fig11_top15_risk_channels.png")

# ═══════════════════════════════════════════════════════════════
# 4. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. STATISTICAL TESTS")
print("="*60)

# T-test: Do emotional titles get more views?
emotional_views = videos_with_ch[videos_with_ch['has_emotional_title'] == True]['views']
normal_views = videos_with_ch[videos_with_ch['has_emotional_title'] == False]['views']

# Use log-transformed views for normality
t_stat, p_val = stats.ttest_ind(np.log1p(emotional_views), np.log1p(normal_views))
print(f"\nT-test: Emotional vs Normal titles (log views)")
print(f"  Emotional: n={len(emotional_views)}, mean={emotional_views.mean():,.0f}, median={emotional_views.median():,.0f}")
print(f"  Normal:    n={len(normal_views)}, mean={normal_views.mean():,.0f}, median={normal_views.median():,.0f}")
print(f"  t-stat = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = stats.mannwhitneyu(emotional_views, normal_views, alternative='greater')
print(f"\nMann-Whitney U test (emotional > normal):")
print(f"  U = {u_stat:.0f}, p-value = {u_pval:.6f}")
print(f"  Significant: {'YES' if u_pval < 0.05 else 'NO'}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((emotional_views.std()**2 + normal_views.std()**2) / 2)
cohens_d = (emotional_views.mean() - normal_views.mean()) / pooled_std
print(f"\nCohen's d (effect size): {cohens_d:.4f}")

# Correlation: cross-platform count vs commercial rate
corr, corr_p = stats.pearsonr(
    risk_features['cross_platform'].fillna(0),
    risk_features['commercial_rate'].fillna(0)
)
print(f"\nCorrelation: Cross-platform count vs Commercial rate")
print(f"  r = {corr:.4f}, p = {corr_p:.4f}")

# ═══════════════════════════════════════════════════════════════
# 5. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════
report = {
    'dataset': {
        'channels': len(channels),
        'videos': len(videos_with_ch),
        'channels_1M_plus': int((channels['subscribers'] >= 1e6).sum()),
    },
    'commercial_detection_v2': {
        'commercial_videos': int(commercial_count),
        'commercial_rate': float(commercial_count / len(videos)),
    },
    'regression': {
        'r_squared': float(r_squared),
        'n_observations': int(n),
        'emotional_title_coeff': float(coeffs[features.index('has_emotional_title') + 1]),
        'emotional_title_pvalue': float(p_values[features.index('has_emotional_title') + 1]),
        'commercial_coeff': float(coeffs[features.index('is_commercial_v2') + 1]),
        'commercial_pvalue': float(p_values[features.index('is_commercial_v2') + 1]),
    },
    'statistical_tests': {
        'emotional_ttest_pvalue': float(p_val),
        'emotional_mannwhitney_pvalue': float(u_pval),
        'cohens_d': float(cohens_d),
        'crossplatform_commercial_corr': float(corr),
    },
    'risk_scoring': {
        'highest_risk_channel': risk_features.iloc[0]['channel_title'],
        'highest_risk_score': float(risk_features.iloc[0]['risk_score']),
        'mean_risk_score': float(risk_features['risk_score'].mean()),
        'channels_above_75th_pct': int((risk_features['risk_score'] > risk_features['risk_score'].quantile(0.75)).sum()),
    }
}

with open(os.path.join(OUT_DIR, 'analysis_report_v2.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS V2 COMPLETE")
print("="*60)
print(json.dumps(report, indent=2))
