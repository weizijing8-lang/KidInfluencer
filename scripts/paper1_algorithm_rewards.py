"""
Paper 1: Does the Algorithm Reward Exploitation?
=================================================
RQ: Does YouTube's recommendation algorithm systematically reward 
    more exploitative content within family channels?

Method: Within-channel OLS regression with controls
  DV: log(view_count)
  IV: exploit_score (title-only and title+desc)
  Controls: video duration, day of week, hour of day, year, 
            video age, title length, has_sponsor
  Fixed Effects: channel (within-channel variation only)

This is PURELY within-family analysis.
"""

import pandas as pd
import numpy as np
import os
import json
import statsmodels.api as sm
from scipy import stats

BASE_DIR = '/home/ubuntu/KidInfluencer'
RESULTS_DIR = os.path.join(BASE_DIR, 'data/paper1_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("PAPER 1: DOES THE ALGORITHM REWARD EXPLOITATION?")
print("="*60)

# ============================================================
# 1. Load and prepare data
# ============================================================
print("\n--- Loading data ---")
df = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/full_results_v4.csv'))
print(f"Total videos: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Rename columns for clarity
df = df.rename(columns={
    'channel_short_name': 'channel',
    'channel_category': 'category',
    'viewCount': 'view_count',
    'likeCount': 'like_count',
    'commentCount': 'comment_count',
    'publishedAt': 'published_at',
})

# Filter to family channels only
df_fam = df[df['category'] == 'family'].copy()
print(f"Family videos: {len(df_fam)}")
print(f"Family channels: {df_fam['channel'].nunique()}")

# ============================================================
# 2. Feature engineering
# ============================================================
print("\n--- Feature engineering ---")

# Parse datetime
df_fam['published_at'] = pd.to_datetime(df_fam['published_at'], errors='coerce')
df_fam = df_fam.dropna(subset=['published_at'])

# DV: log views (add 1 to avoid log(0))
df_fam['log_views'] = np.log1p(df_fam['view_count'].fillna(0))
df_fam['log_likes'] = np.log1p(df_fam['like_count'].fillna(0))
df_fam['log_comments'] = np.log1p(df_fam['comment_count'].fillna(0))

# IV: exploitation scores
df_fam['exploit_title'] = df_fam['exploit_score_title_only'].fillna(0)
df_fam['exploit_combined'] = df_fam['exploit_score_v4'].fillna(0)

# Controls
df_fam['year'] = df_fam['published_at'].dt.year
df_fam['month'] = df_fam['published_at'].dt.month
df_fam['day_of_week'] = df_fam['published_at'].dt.dayofweek  # 0=Monday
df_fam['hour'] = df_fam['published_at'].dt.hour
df_fam['is_weekend'] = (df_fam['day_of_week'] >= 5).astype(int)

# Title length (proxy for clickbait effort)
df_fam['title_length'] = df_fam['title'].fillna('').str.len()
df_fam['title_word_count'] = df_fam['title'].fillna('').str.split().str.len()

# Has caps in title (clickbait signal)
df_fam['title_caps_ratio'] = df_fam['title'].fillna('').apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

# Has exclamation/question marks
df_fam['has_exclamation'] = df_fam['title'].fillna('').str.contains('!').astype(int)
df_fam['has_question'] = df_fam['title'].fillna('').str.contains(r'\?').astype(int)

# Sponsor detection from description
sponsor_keywords = ['#ad', '#sponsored', 'paid partnership', 'sponsor', 'promo code', 
                    'use code', 'discount code', 'affiliate']
df_fam['has_sponsor'] = df_fam['description'].fillna('').str.lower().apply(
    lambda x: int(any(kw in x for kw in sponsor_keywords)))

# Video age (days since publication to now)
df_fam['published_at'] = df_fam['published_at'].dt.tz_localize(None)
df_fam['video_age_days'] = (pd.Timestamp.now() - df_fam['published_at']).dt.days
df_fam['log_video_age'] = np.log1p(df_fam['video_age_days'])

# Channel-level demeaning for within-channel analysis
channel_means = df_fam.groupby('channel')['log_views'].transform('mean')
df_fam['log_views_demeaned'] = df_fam['log_views'] - channel_means

# Drop rows with missing key variables
df_fam = df_fam.dropna(subset=['log_views', 'exploit_title', 'year'])
print(f"Videos after cleaning: {len(df_fam)}")

# ============================================================
# 3. Model 1: Simple within-channel correlation
# ============================================================
print("\n" + "="*60)
print("MODEL 1: Within-Channel Correlation (exploit → views)")
print("="*60)

results_by_channel = []
for ch in sorted(df_fam['channel'].unique()):
    sub = df_fam[df_fam['channel'] == ch]
    if len(sub) < 30:  # Need enough data
        continue
    
    r_title, p_title = stats.pearsonr(sub['exploit_title'], sub['log_views'])
    r_combined, p_combined = stats.pearsonr(sub['exploit_combined'], sub['log_views'])
    
    results_by_channel.append({
        'channel': ch,
        'n_videos': len(sub),
        'r_title': r_title,
        'p_title': p_title,
        'r_combined': r_combined,
        'p_combined': p_combined,
        'sig_title': p_title < 0.05,
        'positive_title': r_title > 0,
        'sig_combined': p_combined < 0.05,
        'positive_combined': r_combined > 0,
    })

df_corr = pd.DataFrame(results_by_channel)
print(f"\nChannels with enough data: {len(df_corr)}")
print(f"\nTitle-only exploit score → log(views):")
print(f"  Significant positive: {((df_corr['sig_title']) & (df_corr['positive_title'])).sum()}/{len(df_corr)}")
print(f"  Significant negative: {((df_corr['sig_title']) & (~df_corr['positive_title'])).sum()}/{len(df_corr)}")
print(f"  Not significant: {(~df_corr['sig_title']).sum()}/{len(df_corr)}")
print(f"  Mean r: {df_corr['r_title'].mean():.4f}")

print(f"\nCombined exploit score → log(views):")
print(f"  Significant positive: {((df_corr['sig_combined']) & (df_corr['positive_combined'])).sum()}/{len(df_corr)}")
print(f"  Significant negative: {((df_corr['sig_combined']) & (~df_corr['positive_combined'])).sum()}/{len(df_corr)}")
print(f"  Not significant: {(~df_corr['sig_combined']).sum()}/{len(df_corr)}")
print(f"  Mean r: {df_corr['r_combined'].mean():.4f}")

print("\nPer-channel results:")
for _, row in df_corr.sort_values('r_title', ascending=False).iterrows():
    sig = '***' if row['p_title'] < 0.001 else '**' if row['p_title'] < 0.01 else '*' if row['p_title'] < 0.05 else ''
    print(f"  {row['channel']:25s} r={row['r_title']:+.4f} {sig:3s} (n={row['n_videos']})")

df_corr.to_csv(os.path.join(RESULTS_DIR, 'model1_within_channel_correlations.csv'), index=False)

# ============================================================
# 4. Model 2: Pooled OLS with Channel Fixed Effects + Controls
# ============================================================
print("\n" + "="*60)
print("MODEL 2: Pooled OLS with Channel FE + Controls")
print("="*60)

# Create channel dummies
channel_dummies = pd.get_dummies(df_fam['channel'], prefix='ch', drop_first=True)

# Year dummies
year_dummies = pd.get_dummies(df_fam['year'], prefix='yr', drop_first=True)

# Build feature matrix
controls = ['log_video_age', 'is_weekend', 'title_length', 'title_caps_ratio',
            'has_exclamation', 'has_question', 'has_sponsor']

# Model 2a: exploit_title only (no controls, with channel FE)
X_2a = pd.concat([df_fam[['exploit_title']], channel_dummies], axis=1).astype(float)
X_2a = sm.add_constant(X_2a)
y = df_fam['log_views'].astype(float)

# Drop any remaining NaN
mask = X_2a.notna().all(axis=1) & y.notna()
X_2a = X_2a[mask]
y_2a = y[mask]

model_2a = sm.OLS(y_2a, X_2a).fit(cov_type='HC1')
print(f"\nModel 2a: exploit_title + channel FE")
print(f"  β(exploit_title) = {model_2a.params['exploit_title']:.4f}")
print(f"  SE = {model_2a.bse['exploit_title']:.4f}")
print(f"  t = {model_2a.tvalues['exploit_title']:.4f}")
print(f"  p = {model_2a.pvalues['exploit_title']:.6f}")
print(f"  R² = {model_2a.rsquared:.4f}")
print(f"  N = {model_2a.nobs:.0f}")

# Model 2b: exploit_title + controls + channel FE
X_2b = pd.concat([df_fam[['exploit_title'] + controls], channel_dummies, year_dummies], axis=1).astype(float)
X_2b = sm.add_constant(X_2b)
mask = X_2b.notna().all(axis=1) & y.notna()
X_2b = X_2b[mask]
y_2b = y[mask]

model_2b = sm.OLS(y_2b, X_2b).fit(cov_type='HC1')
print(f"\nModel 2b: exploit_title + controls + channel FE + year FE")
print(f"  β(exploit_title) = {model_2b.params['exploit_title']:.4f}")
print(f"  SE = {model_2b.bse['exploit_title']:.4f}")
print(f"  t = {model_2b.tvalues['exploit_title']:.4f}")
print(f"  p = {model_2b.pvalues['exploit_title']:.6f}")
print(f"  R² = {model_2b.rsquared:.4f}")
print(f"  N = {model_2b.nobs:.0f}")

# Print control variable coefficients
print(f"\n  Control variable effects:")
for ctrl in controls:
    if ctrl in model_2b.params:
        sig = '***' if model_2b.pvalues[ctrl] < 0.001 else '**' if model_2b.pvalues[ctrl] < 0.01 else '*' if model_2b.pvalues[ctrl] < 0.05 else ''
        print(f"    {ctrl:25s} β={model_2b.params[ctrl]:+.4f} (p={model_2b.pvalues[ctrl]:.4f}) {sig}")

# Model 2c: exploit_combined + controls + channel FE
X_2c = pd.concat([df_fam[['exploit_combined'] + controls], channel_dummies, year_dummies], axis=1).astype(float)
X_2c = sm.add_constant(X_2c)
mask = X_2c.notna().all(axis=1) & y.notna()
X_2c = X_2c[mask]
y_2c = y[mask]

model_2c = sm.OLS(y_2c, X_2c).fit(cov_type='HC1')
print(f"\nModel 2c: exploit_combined + controls + channel FE + year FE")
print(f"  β(exploit_combined) = {model_2c.params['exploit_combined']:.4f}")
print(f"  SE = {model_2c.bse['exploit_combined']:.4f}")
print(f"  t = {model_2c.tvalues['exploit_combined']:.4f}")
print(f"  p = {model_2c.pvalues['exploit_combined']:.6f}")
print(f"  R² = {model_2c.rsquared:.4f}")

# ============================================================
# 5. Model 3: Also test with likes and comments as DV
# ============================================================
print("\n" + "="*60)
print("MODEL 3: Robustness - Different DVs (likes, comments)")
print("="*60)

for dv_name, dv_col in [('log_likes', 'log_likes'), ('log_comments', 'log_comments')]:
    X = pd.concat([df_fam[['exploit_title'] + controls], channel_dummies, year_dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y_alt = df_fam[dv_col].astype(float)
    mask = X.notna().all(axis=1) & y_alt.notna()
    X_m = X[mask]
    y_m = y_alt[mask]
    
    model = sm.OLS(y_m, X_m).fit(cov_type='HC1')
    sig = '***' if model.pvalues['exploit_title'] < 0.001 else '**' if model.pvalues['exploit_title'] < 0.01 else '*' if model.pvalues['exploit_title'] < 0.05 else ''
    print(f"  DV={dv_name}: β(exploit_title)={model.params['exploit_title']:+.4f}, p={model.pvalues['exploit_title']:.6f} {sig}, R²={model.rsquared:.4f}")

# ============================================================
# 6. Model 4: Quantile regression (does exploit help MORE for viral hits?)
# ============================================================
print("\n" + "="*60)
print("MODEL 4: Quantile Regression (effect at different view quantiles)")
print("="*60)

X_qr = pd.concat([df_fam[['exploit_title'] + controls], channel_dummies], axis=1).astype(float)
X_qr = sm.add_constant(X_qr)
y_qr = df_fam['log_views'].astype(float)
mask = X_qr.notna().all(axis=1) & y_qr.notna()
X_qr = X_qr[mask]
y_qr = y_qr[mask]

quantile_results = []
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    try:
        model_q = sm.QuantReg(y_qr, X_qr).fit(q=q, max_iter=1000)
        beta = model_q.params['exploit_title']
        pval = model_q.pvalues['exploit_title']
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  Q{q:.2f}: β(exploit_title) = {beta:+.4f}, p = {pval:.6f} {sig}")
        quantile_results.append({'quantile': q, 'beta': beta, 'pvalue': pval})
    except Exception as e:
        print(f"  Q{q:.2f}: Failed - {e}")

pd.DataFrame(quantile_results).to_csv(os.path.join(RESULTS_DIR, 'model4_quantile_regression.csv'), index=False)

# ============================================================
# 7. Summary table for paper
# ============================================================
print("\n" + "="*60)
print("REGRESSION TABLE FOR PAPER")
print("="*60)

summary = {
    'Model': ['2a: FE only', '2b: FE + Controls', '2c: Combined score'],
    'IV': ['exploit_title', 'exploit_title', 'exploit_combined'],
    'DV': ['log(views)', 'log(views)', 'log(views)'],
    'beta': [model_2a.params['exploit_title'], model_2b.params['exploit_title'], model_2c.params['exploit_combined']],
    'se': [model_2a.bse['exploit_title'], model_2b.bse['exploit_title'], model_2c.bse['exploit_combined']],
    't': [model_2a.tvalues['exploit_title'], model_2b.tvalues['exploit_title'], model_2c.tvalues['exploit_combined']],
    'p': [model_2a.pvalues['exploit_title'], model_2b.pvalues['exploit_title'], model_2c.pvalues['exploit_combined']],
    'R2': [model_2a.rsquared, model_2b.rsquared, model_2c.rsquared],
    'N': [model_2a.nobs, model_2b.nobs, model_2c.nobs],
    'Channel_FE': ['Yes', 'Yes', 'Yes'],
    'Year_FE': ['No', 'Yes', 'Yes'],
    'Controls': ['No', 'Yes', 'Yes'],
}

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(RESULTS_DIR, 'regression_table.csv'), index=False)

# ============================================================
# 8. Effect size interpretation
# ============================================================
print("\n" + "="*60)
print("EFFECT SIZE INTERPRETATION")
print("="*60)

beta = model_2b.params['exploit_title']
# exploit_title ranges from about -0.05 to 0.20
exploit_range = df_fam['exploit_title'].quantile(0.75) - df_fam['exploit_title'].quantile(0.25)
print(f"Exploitation score IQR: {exploit_range:.4f}")
print(f"β(exploit_title) = {beta:.4f}")
print(f"Moving from Q1 to Q3 of exploitation → {beta * exploit_range:.4f} change in log(views)")
print(f"  = {(np.exp(beta * exploit_range) - 1) * 100:.1f}% change in views")

# Mean views for context
mean_views = df_fam['view_count'].mean()
median_views = df_fam['view_count'].median()
print(f"\nMean views per video: {mean_views:,.0f}")
print(f"Median views per video: {median_views:,.0f}")
print(f"A {(np.exp(beta * exploit_range) - 1) * 100:.1f}% increase on median = {median_views * (np.exp(beta * exploit_range) - 1):,.0f} additional views")

print("\nDone! Results saved to", RESULTS_DIR)
