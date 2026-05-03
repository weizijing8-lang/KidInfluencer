#!/usr/bin/env python3
"""
Kidfluencer Analysis V3 — With LLM Annotations
================================================
Clean annotations, merge with video data, run regression and risk analysis.
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

# ── Config ──
DATA_DIR = '/home/ubuntu/KidInfluencer/data'
OUT_DIR = '/home/ubuntu/KidInfluencer/analysis_v3'
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.figsize': (10, 6), 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# ── Load Data ──
annotations = pd.read_csv(os.path.join(DATA_DIR, 'annotations_merged.csv'))
videos = pd.read_csv(os.path.join(DATA_DIR, 'combined_videos.csv'))
channels = pd.read_csv(os.path.join(DATA_DIR, 'combined_channels.csv'))

print(f"Annotations: {len(annotations)}")
print(f"Videos: {len(videos)}")
print(f"Channels: {len(channels)}")

# ═══════════════════════════════════════════════════════════════
# 1. CLEAN & STANDARDIZE ANNOTATIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. CLEANING ANNOTATIONS")
print("="*60)

# Remove error rows
clean = annotations[annotations['content_type'] != 'error'].copy()
print(f"After removing errors: {len(clean)} rows ({len(clean)/len(annotations)*100:.1f}%)")

# Standardize emotional_manipulation
em_map = {
    'none': 'none', 'mild': 'mild', 'moderate': 'moderate', 'severe': 'severe',
    'low': 'mild', 'Low': 'mild', 'medium': 'moderate', 'Medium': 'moderate',
    'High': 'severe', 'high': 'severe', 'Moderate': 'moderate',
    'sensational': 'moderate', 'Emotional': 'moderate', 'emotional_appeal': 'mild',
    'implied': 'mild', 'positive': 'none', 'Humor': 'none', 'Empathy': 'none',
    'not_applicable': 'none', 'no_emotional_manipulation': 'none',
    'Humor, Curiosity': 'none', 'Humor, Empathy': 'none',
}
clean['emotional_manipulation'] = clean['emotional_manipulation'].map(em_map).fillna('none')

# Standardize commercial_signals
cs_map = {
    'none': 'none', 'brand_mention': 'brand_mention', 'likely_sponsored': 'likely_sponsored',
    'affiliate': 'affiliate', 'product_placement': 'product_placement',
    'no': 'none', 'no_commercial_content': 'none', 'no_commercial_signals': 'none',
    'low': 'none', 'medium': 'brand_mention', 'high': 'likely_sponsored',
    'other': 'none', 'direct_promotion': 'likely_sponsored',
    'Product/Service Promotion': 'likely_sponsored',
    'Potential (displaying products)': 'product_placement',
}
clean['commercial_signals'] = clean['commercial_signals'].map(cs_map).fillna('none')

# Standardize child_role
cr_map = {
    'protagonist': 'protagonist', 'co_star': 'co_star', 'cameo': 'cameo', 'unclear': 'unclear',
    'featured': 'protagonist', 'Central': 'protagonist', 'main_protagonist': 'protagonist',
    'main_subject': 'protagonist', 'Featured': 'protagonist', 'Actors': 'protagonist',
    'none': 'unclear', 'not_applicable': 'unclear', 'Not Applicable': 'unclear',
    'Not applicable': 'unclear', 'Mentioned': 'cameo', 'Peripheral': 'cameo',
    'Implied': 'unclear', 'Implied presence': 'unclear',
}
clean['child_role'] = clean['child_role'].map(cr_map).fillna('unclear')

# Standardize privacy_concern
pc_map = {
    'none': 'none', 'low': 'low', 'moderate': 'moderate', 'high': 'high',
    'mild': 'low', 'minimal': 'low', 'Medium': 'moderate', 'medium': 'moderate',
    'High': 'high', 'Significant': 'high', 'Low': 'low', 'Minor': 'low',
    'no_privacy_concern': 'none', 'not_applicable': 'none',
}
clean['privacy_concern'] = clean['privacy_concern'].map(pc_map).fillna('none')

# Standardize clickbait_level
cb_map = {
    'none': 'none', 'mild': 'mild', 'moderate': 'moderate', 'severe': 'severe',
    'Low': 'mild', 'low': 'mild', 'Medium': 'moderate', 'medium': 'moderate',
    'High': 'severe', 'high': 'severe', 'Moderate': 'moderate',
}
clean['clickbait_level'] = clean['clickbait_level'].map(cb_map).fillna('none')

# Create numeric versions
clean['emotional_score'] = clean['emotional_manipulation'].map({'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3})
clean['commercial_binary'] = (clean['commercial_signals'] != 'none').astype(int)
clean['privacy_score'] = clean['privacy_concern'].map({'none': 0, 'low': 1, 'moderate': 2, 'high': 3})
clean['clickbait_score'] = clean['clickbait_level'].map({'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3})
clean['child_protagonist'] = (clean['child_role'] == 'protagonist').astype(int)

print(f"\nCleaned distributions:")
for col in ['content_type', 'emotional_manipulation', 'commercial_signals', 'child_role', 'privacy_concern', 'clickbait_level']:
    print(f"\n{col}:")
    print(clean[col].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════
# 2. MERGE WITH VIDEO METADATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. MERGING WITH VIDEO METADATA")
print("="*60)

# Merge annotations with video data
merged = clean.merge(videos[['video_id', 'channel_id', 'length_seconds', 'views']], on='video_id', how='inner')
merged = merged.merge(channels[['channel_id', 'title', 'subscribers', 'total_videos', 'total_views', 'cross_platform_count']], 
                      on='channel_id', how='inner', suffixes=('', '_channel'))
print(f"Merged dataset: {len(merged)} videos across {merged['channel_id'].nunique()} channels")

# ═══════════════════════════════════════════════════════════════
# 3. CONTENT TYPE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. CONTENT TYPE ANALYSIS")
print("="*60)

# Views by content type
content_views = merged.groupby('content_type').agg(
    count=('views', 'count'),
    mean_views=('views', 'mean'),
    median_views=('views', 'median'),
).sort_values('mean_views', ascending=False)
print("\nViews by Content Type:")
print(content_views.to_string())

# Fig: Content type distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ct_counts = merged['content_type'].value_counts()
axes[0].barh(ct_counts.index, ct_counts.values, color='#2196F3', edgecolor='white')
axes[0].set_xlabel('Number of Videos')
axes[0].set_title('Content Type Distribution')
axes[0].invert_yaxis()

ct_views = content_views[content_views['count'] >= 20].sort_values('median_views')
axes[1].barh(ct_views.index, ct_views['median_views'] / 1000, color='#FF9800', edgecolor='white')
axes[1].set_xlabel('Median Views (thousands)')
axes[1].set_title('Median Views by Content Type')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_content_types.png'))
plt.close()
print("Saved fig1_content_types.png")

# ═══════════════════════════════════════════════════════════════
# 4. EMOTIONAL MANIPULATION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. EMOTIONAL MANIPULATION ANALYSIS")
print("="*60)

em_views = merged.groupby('emotional_manipulation').agg(
    count=('views', 'count'),
    mean_views=('views', 'mean'),
    median_views=('views', 'median'),
)
print("\nViews by Emotional Manipulation Level:")
print(em_views.to_string())

# T-test: any emotional vs none
emotional_vids = merged[merged['emotional_score'] > 0]['views']
none_vids = merged[merged['emotional_score'] == 0]['views']
t_stat, p_val = stats.ttest_ind(np.log1p(emotional_vids), np.log1p(none_vids), nan_policy='omit')
u_stat, u_pval = stats.mannwhitneyu(emotional_vids.dropna(), none_vids.dropna(), alternative='greater')
print(f"\nT-test (log views): emotional vs none: t={t_stat:.3f}, p={p_val:.6f}")
print(f"Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.6f}")
print(f"Effect: emotional mean={emotional_vids.mean():,.0f} vs none mean={none_vids.mean():,.0f}")

# Fig: Emotional manipulation vs views
fig, ax = plt.subplots(figsize=(8, 5))
order = ['none', 'mild', 'moderate', 'severe']
em_data = merged[merged['emotional_manipulation'].isin(order)]
sns.boxplot(data=em_data, x='emotional_manipulation', y='views', order=order, ax=ax,
            palette=['#4CAF50', '#FFC107', '#FF9800', '#F44336'], showfliers=False)
ax.set_yscale('log')
ax.set_xlabel('Emotional Manipulation Level')
ax.set_ylabel('Views (log scale)')
ax.set_title('Video Views by Emotional Manipulation Level\n(higher manipulation → higher views)')
# Add sample sizes
for i, level in enumerate(order):
    n = len(em_data[em_data['emotional_manipulation'] == level])
    ax.text(i, ax.get_ylim()[0] * 1.5, f'n={n}', ha='center', fontsize=9)
fig.savefig(os.path.join(OUT_DIR, 'fig2_emotional_vs_views.png'))
plt.close()
print("Saved fig2_emotional_vs_views.png")

# ═══════════════════════════════════════════════════════════════
# 5. COMMERCIAL SIGNALS ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. COMMERCIAL SIGNALS ANALYSIS")
print("="*60)

cs_views = merged.groupby('commercial_signals').agg(
    count=('views', 'count'),
    mean_views=('views', 'mean'),
    median_views=('views', 'median'),
)
print("\nViews by Commercial Signals:")
print(cs_views.to_string())

commercial_rate = merged['commercial_binary'].mean()
print(f"\nOverall commercial rate: {commercial_rate:.1%}")

# By channel
# Get channel title column name
title_col = 'title'
ch_commercial = merged.groupby('channel_id').agg(
    channel_title=(title_col, 'first'),
    subscribers=('subscribers', 'first'),
    commercial_rate=('commercial_binary', 'mean'),
    n_videos=('video_id', 'count'),
).sort_values('commercial_rate', ascending=False)
print("\nTop 10 Most Commercialized Channels:")
for _, row in ch_commercial.head(10).iterrows():
    print(f"  {row['channel_title'][:35]:35s} | {row['commercial_rate']:.0%} ({row['n_videos']} vids) | {row['subscribers']:>12,.0f} subs")

# ═══════════════════════════════════════════════════════════════
# 6. REGRESSION ANALYSIS (IMPROVED)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. REGRESSION ANALYSIS (WITH LLM FEATURES)")
print("="*60)

reg = merged.copy()
reg['log_views'] = np.log1p(reg['views'])
reg['log_subs'] = np.log1p(reg['subscribers'])
reg['duration_min'] = reg['length_seconds'] / 60

# Features
features = ['log_subs', 'duration_min', 'emotional_score', 'commercial_binary', 
            'privacy_score', 'clickbait_score', 'child_protagonist', 'cross_platform_count']
X = reg[features].fillna(0).values
X = np.column_stack([np.ones(len(X)), X])
y = reg['log_views'].values

mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X, y = X[mask], y[mask]

from numpy.linalg import lstsq
coeffs, _, _, _ = lstsq(X, y, rcond=None)
y_pred = X @ coeffs
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot

n, p = len(y), X.shape[1]
mse = ss_res / (n - p)
var_coeff = mse * np.linalg.inv(X.T @ X).diagonal()
se = np.sqrt(np.abs(var_coeff))
t_stats_reg = coeffs / se
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats_reg), n - p))

print(f"\nOLS: log(views) ~ LLM features + controls")
print(f"R² = {r_squared:.4f}, N = {n}")
print(f"\n{'Variable':<25s} {'Coeff':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'Sig':>5s}")
print("-" * 65)
coeff_names = ['intercept'] + features
for name, c, s, t, pv in zip(coeff_names, coeffs, se, t_stats_reg, p_values):
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    print(f"  {name:<23s} {c:8.4f} {s:8.4f} {t:8.2f} {pv:10.6f} {sig}")

# Fig: Regression coefficients
fig, ax = plt.subplots(figsize=(10, 6))
feat_coeffs = coeffs[1:]
feat_ses = se[1:]
feat_pvals = p_values[1:]
colors = ['#F44336' if p < 0.05 else '#9E9E9E' for p in feat_pvals]
y_pos = range(len(features))
ax.barh(y_pos, feat_coeffs, xerr=1.96*feat_ses, color=colors, edgecolor='white', alpha=0.8, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient (effect on log views)')
ax.set_title(f'What Predicts Video Views? (OLS, R²={r_squared:.3f})\nRed = significant (p < 0.05)')
fig.savefig(os.path.join(OUT_DIR, 'fig3_regression_v3.png'))
plt.close()
print("Saved fig3_regression_v3.png")

# ═══════════════════════════════════════════════════════════════
# 7. UPDATED RISK SCORING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. UPDATED RISK SCORING (DATA-DRIVEN)")
print("="*60)

# Channel-level aggregation
ch_risk = merged.groupby('channel_id').agg(
    channel_title=('title', 'first'),
    subscribers=('subscribers', 'first'),
    n_videos=('video_id', 'count'),
    total_videos_on_channel=('total_videos', 'first'),
    cross_platform=('cross_platform_count', 'first'),
    # LLM-based features
    emotional_score_mean=('emotional_score', 'mean'),
    commercial_rate=('commercial_binary', 'mean'),
    privacy_score_mean=('privacy_score', 'mean'),
    clickbait_score_mean=('clickbait_score', 'mean'),
    child_protagonist_rate=('child_protagonist', 'mean'),
    # Views
    avg_views=('views', 'mean'),
    median_views=('views', 'median'),
).reset_index()

# Data-driven risk score: use regression coefficients as weights
# Normalize each feature to 0-1
def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(0, index=s.index)

# Weight by significance and coefficient magnitude from regression
# emotional_score: 0.1729, clickbait: 0.1502, commercial: 0.4269, privacy: -0.0282
# Use absolute regression coefficients (excluding controls) as weights
risk_weights = {
    'emotional_score_mean': abs(coeffs[features.index('emotional_score') + 1]),
    'clickbait_score_mean': abs(coeffs[features.index('clickbait_score') + 1]),
    'commercial_rate': abs(coeffs[features.index('commercial_binary') + 1]),
    'privacy_score_mean': abs(coeffs[features.index('privacy_score') + 1]),
    'child_protagonist_rate': abs(coeffs[features.index('child_protagonist') + 1]),
}
# Normalize weights to sum to 1
total_w = sum(risk_weights.values())
risk_weights = {k: v/total_w for k, v in risk_weights.items()}
print(f"Data-driven risk weights: {json.dumps({k: f'{v:.3f}' for k, v in risk_weights.items()}, indent=2)}")

ch_risk['risk_score'] = sum(
    risk_weights[feat] * norm(ch_risk[feat]) for feat in risk_weights
)
ch_risk = ch_risk.sort_values('risk_score', ascending=False)

print("\nTop 15 Highest Risk Channels (data-driven weights):")
print(f"{'Channel':<35s} {'Risk':>6s} {'Emot':>5s} {'Comm%':>6s} {'Priv':>5s} {'Click':>5s} {'ChildProt':>9s}")
print("-" * 80)
for _, row in ch_risk.head(15).iterrows():
    print(f"  {row['channel_title'][:33]:33s} {row['risk_score']:.3f} {row['emotional_score_mean']:.2f} {row['commercial_rate']:.0%} {row['privacy_score_mean']:.2f} {row['clickbait_score_mean']:.2f} {row['child_protagonist_rate']:.0%}")

ch_risk.to_csv(os.path.join(OUT_DIR, 'channel_risk_scores_v3.csv'), index=False)

# Fig: Top 15 risk channels
fig, ax = plt.subplots(figsize=(12, 7))
top15 = ch_risk.head(15)
colors_risk = plt.cm.Reds(np.linspace(0.4, 0.9, 15))
bars = ax.barh(range(len(top15)), top15['risk_score'], color=colors_risk, edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['channel_title'].values)
ax.set_xlabel('Risk Score (data-driven)')
ax.set_title('Top 15 Highest-Risk Kidfluencer Channels\n(weights derived from regression coefficients)')
ax.invert_yaxis()
fig.savefig(os.path.join(OUT_DIR, 'fig4_top15_risk_v3.png'))
plt.close()
print("Saved fig4_top15_risk_v3.png")

# ═══════════════════════════════════════════════════════════════
# 8. PLATFORM INCENTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. PLATFORM INCENTIVE ANALYSIS")
print("="*60)

# Does the platform reward problematic content?
# Compare views across emotional manipulation levels, controlling for channel size
# Use within-channel comparison
within_ch = merged.groupby(['channel_id', 'emotional_manipulation']).agg(
    mean_views=('views', 'mean'),
    count=('video_id', 'count'),
).reset_index()

# Also fix title reference in within-channel section

# For channels with both emotional and non-emotional content
ch_with_both = within_ch.pivot(index='channel_id', columns='emotional_manipulation', values='mean_views')
if 'none' in ch_with_both.columns and 'mild' in ch_with_both.columns:
    both = ch_with_both[['none', 'mild']].dropna()
    if len(both) > 5:
        ratio = both['mild'] / both['none']
        print(f"\nWithin-channel comparison (same channel, emotional vs non-emotional):")
        print(f"  Channels with both types: {len(both)}")
        print(f"  Mean ratio (mild/none): {ratio.mean():.2f}x")
        print(f"  Median ratio: {ratio.median():.2f}x")
        t, p = stats.ttest_rel(np.log1p(both['mild']), np.log1p(both['none']))
        print(f"  Paired t-test (log views): t={t:.3f}, p={p:.6f}")

# Content type that gets most views
print("\nPlatform reward by content type (median views):")
ct_reward = merged.groupby('content_type')['views'].median().sort_values(ascending=False)
print(ct_reward.head(10).to_string())

# ═══════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════
summary = {
    'dataset': {
        'annotated_videos': len(clean),
        'merged_with_metadata': len(merged),
        'channels': int(merged['channel_id'].nunique()),
        'annotation_success_rate': f"{len(clean)/len(annotations)*100:.1f}%",
    },
    'key_findings': {
        'emotional_manipulation_rate': f"{(merged['emotional_score'] > 0).mean()*100:.1f}%",
        'commercial_rate': f"{commercial_rate*100:.1f}%",
        'child_protagonist_rate': f"{merged['child_protagonist'].mean()*100:.1f}%",
        'privacy_concern_rate': f"{(merged['privacy_score'] > 0).mean()*100:.1f}%",
        'clickbait_rate': f"{(merged['clickbait_score'] > 0).mean()*100:.1f}%",
    },
    'regression': {
        'r_squared': f"{r_squared:.4f}",
        'emotional_coeff': f"{coeffs[features.index('emotional_score') + 1]:.4f}",
        'emotional_p': f"{p_values[features.index('emotional_score') + 1]:.6f}",
        'clickbait_coeff': f"{coeffs[features.index('clickbait_score') + 1]:.4f}",
        'clickbait_p': f"{p_values[features.index('clickbait_score') + 1]:.6f}",
        'commercial_coeff': f"{coeffs[features.index('commercial_binary') + 1]:.4f}",
        'commercial_p': f"{p_values[features.index('commercial_binary') + 1]:.6f}",
    },
    'statistical_tests': {
        'emotional_ttest_p': f"{p_val:.6f}",
        'emotional_mannwhitney_p': f"{u_pval:.6f}",
    },
}

with open(os.path.join(OUT_DIR, 'summary_v3.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS V3 COMPLETE")
print("="*60)
print(json.dumps(summary, indent=2))
