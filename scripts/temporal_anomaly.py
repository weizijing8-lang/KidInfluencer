"""
Temporal Anomaly Detection & Within-Group Analysis
====================================================
1. Within family channels: do behavioral features predict exploitation?
2. Changepoint detection: when do channels shift to higher-intensity patterns?
3. Ablation: remove engagement features, test with pure temporal/linguistic only
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/home/ubuntu/KidInfluencer/analysis_behavioral/figures', exist_ok=True)

print("=" * 60)
print("TEMPORAL ANOMALY & WITHIN-GROUP ANALYSIS")
print("=" * 60)

# Load data
v4 = pd.read_csv('/home/ubuntu/KidInfluencer/data/results_v4/full_results_v4.csv',
                 usecols=['id', 'title', 'publishedAt', 'channel_short_name', 
                          'channel_category', 'viewCount', 'likeCount', 
                          'commentCount', 'exploit_score_v4'])
v4['publishedAt'] = pd.to_datetime(v4['publishedAt'], errors='coerce')
v4 = v4.dropna(subset=['publishedAt'])
v4 = v4[v4['channel_category'].isin(['adult', 'family'])]

# Load behavioral features
features_df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_behavioral/channel_behavioral_features.csv', index_col=0)

# ============================================================
# PART 1: Feature Ablation Study
# ============================================================
print("\n[1/4] Feature Ablation Study...")

categories = features_df['category'].values
y = (categories == 'family').astype(int)

# Define feature groups
all_features = [c for c in features_df.columns if c != 'category']

temporal_features = ['uploads_per_week', 'uploads_per_month', 'interval_mean_hours',
                     'interval_median_hours', 'interval_std_hours', 'interval_cv',
                     'interval_min_hours', 'interval_max_hours', 'burst_ratio',
                     'weekend_ratio', 'dow_entropy', 'hour_mean', 'hour_std',
                     'evening_ratio', 'hour_entropy', 'upload_trend_slope',
                     'upload_trend_r2', 'holiday_upload_ratio']

linguistic_features = ['title_length_mean', 'title_length_std', 'title_caps_ratio',
                       'title_exclaim_ratio', 'title_question_ratio', 'title_emoji_ratio',
                       'title_allcaps_ratio', 'title_number_ratio', 'title_emotional_ratio',
                       'title_word_count_mean']

engagement_features = ['views_mean', 'views_median', 'views_std', 'views_cv',
                       'views_skew', 'views_max_ratio', 'like_view_ratio_mean',
                       'comment_view_ratio_mean', 'views_growth_ratio']

exploit_features = ['exploit_score_mean', 'exploit_score_std', 'exploit_score_max',
                    'high_exploit_ratio']

scale_features = ['total_videos', 'channel_age_days']

# Filter to only features that exist in our dataframe
temporal_features = [f for f in temporal_features if f in features_df.columns]
linguistic_features = [f for f in linguistic_features if f in features_df.columns]
engagement_features = [f for f in engagement_features if f in features_df.columns]
exploit_features = [f for f in exploit_features if f in features_df.columns]
scale_features = [f for f in scale_features if f in features_df.columns]

feature_groups = {
    'All Features': all_features,
    'Temporal Only': temporal_features,
    'Linguistic Only': linguistic_features,
    'Engagement Only': engagement_features,
    'Temporal + Linguistic': temporal_features + linguistic_features,
    'Temporal + Linguistic + Scale': temporal_features + linguistic_features + scale_features,
    'Without Engagement': temporal_features + linguistic_features + exploit_features + scale_features,
    'Without Exploit Score': temporal_features + linguistic_features + engagement_features + scale_features,
}

ablation_results = {}
for group_name, group_features in feature_groups.items():
    X = features_df[group_features].fillna(0).astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    cv_auc = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    cv_f1 = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')
    
    # LOO
    loo_correct = 0
    for train_idx, test_idx in LeaveOneOut().split(X_scaled):
        rf_loo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf_loo.fit(X_scaled[train_idx], y[train_idx])
        pred = rf_loo.predict(X_scaled[test_idx])
        loo_correct += (pred == y[test_idx]).sum()
    
    ablation_results[group_name] = {
        'AUC': cv_auc.mean(),
        'AUC_std': cv_auc.std(),
        'F1': cv_f1.mean(),
        'F1_std': cv_f1.std(),
        'LOO_Acc': loo_correct / len(y),
        'n_features': len(group_features)
    }
    print(f"  {group_name:35s}: AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}, F1={cv_f1.mean():.3f}, LOO={loo_correct/len(y):.3f} ({len(group_features)} features)")

# ============================================================
# PART 2: Within-Group Analysis (Family channels only)
# ============================================================
print("\n[2/4] Within-Group Analysis (Family channels)...")

family_df = features_df[features_df['category'] == 'family'].copy()
print(f"  Family channels: {len(family_df)}")

# Can behavioral features predict exploit_score within family channels?
family_features = [c for c in family_df.columns if c not in ['category', 'exploit_score_mean', 'exploit_score_std', 'exploit_score_max', 'high_exploit_ratio']]
X_family = family_df[family_features].fillna(0).astype(float).values
y_exploit = family_df['exploit_score_mean'].astype(float).values

scaler = StandardScaler()
X_family_scaled = scaler.fit_transform(X_family)

# Regression: predict exploit score from behavioral features
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
rf_reg.fit(X_family_scaled, y_exploit)
y_pred = rf_reg.predict(X_family_scaled)
r2_train = r2_score(y_exploit, y_pred)
print(f"  RF Regression (exploit_score ~ behavioral): Train R²={r2_train:.3f}")

# LOO R² for small sample
loo_preds = np.zeros(len(y_exploit))
for i in range(len(y_exploit)):
    mask = np.ones(len(y_exploit), dtype=bool)
    mask[i] = False
    rf_loo = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
    rf_loo.fit(X_family_scaled[mask], y_exploit[mask])
    loo_preds[i] = rf_loo.predict(X_family_scaled[~mask])[0]

loo_r2 = r2_score(y_exploit, loo_preds)
print(f"  LOO R²: {loo_r2:.3f}")

# Correlations within family channels
print(f"\n  Key correlations (within family channels):")
corr_features = ['uploads_per_week', 'burst_ratio', 'title_emotional_ratio', 
                 'title_caps_ratio', 'views_mean', 'interval_mean_hours', 'total_videos']
for feat in corr_features:
    if feat in family_df.columns:
        r, p = stats.pearsonr(family_df[feat].astype(float).fillna(0), family_df['exploit_score_mean'].astype(float))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {feat:30s} ↔ exploit_score: r={r:.3f}, p={p:.4f} {sig}")

# ============================================================
# PART 3: Temporal Changepoint Detection
# ============================================================
print("\n[3/4] Temporal Changepoint Detection...")

def detect_changepoints(channel_df, window_months=6):
    """Detect behavioral changepoints in a channel's history."""
    df = channel_df.sort_values('publishedAt').copy()
    df['month'] = df['publishedAt'].dt.to_period('M')
    
    monthly = df.groupby('month').agg(
        n_videos=('id', 'count'),
        mean_views=('viewCount', 'mean'),
        mean_exploit=('exploit_score_v4', 'mean'),
        emotional_ratio=('title', lambda x: x.astype(str).apply(
            lambda t: any(w in t.lower() for w in ['amazing','incredible','shocking','crazy','challenge','prank','surprise'])
        ).mean())
    ).reset_index()
    
    if len(monthly) < 12:
        return None
    
    # Rolling statistics
    monthly['upload_rolling'] = monthly['n_videos'].rolling(window_months, min_periods=3).mean()
    monthly['exploit_rolling'] = monthly['mean_exploit'].rolling(window_months, min_periods=3).mean()
    
    # Simple changepoint: find the month where upload frequency increases most
    if len(monthly) > window_months * 2:
        diffs = monthly['upload_rolling'].diff()
        if diffs.max() > 0:
            changepoint_idx = diffs.idxmax()
            return {
                'channel': channel_df['channel_short_name'].iloc[0],
                'changepoint_month': str(monthly.loc[changepoint_idx, 'month']),
                'upload_before': monthly.loc[:changepoint_idx, 'n_videos'].mean(),
                'upload_after': monthly.loc[changepoint_idx:, 'n_videos'].mean(),
                'exploit_before': monthly.loc[:changepoint_idx, 'mean_exploit'].mean(),
                'exploit_after': monthly.loc[changepoint_idx:, 'mean_exploit'].mean(),
                'upload_increase_ratio': monthly.loc[changepoint_idx:, 'n_videos'].mean() / max(monthly.loc[:changepoint_idx, 'n_videos'].mean(), 0.1),
                'monthly_data': monthly
            }
    return None

# Run changepoint detection on family channels
family_channels = v4[v4['channel_category'] == 'family']['channel_short_name'].unique()
changepoints = {}
for ch in family_channels:
    ch_data = v4[v4['channel_short_name'] == ch]
    if len(ch_data) >= 50:
        result = detect_changepoints(ch_data)
        if result:
            changepoints[ch] = result

print(f"  Detected changepoints in {len(changepoints)}/{len(family_channels)} family channels")

# Analyze: does exploit score increase after upload frequency increases?
if changepoints:
    increases = []
    for ch, cp in changepoints.items():
        increases.append({
            'channel': ch,
            'upload_increase': cp['upload_increase_ratio'],
            'exploit_before': cp['exploit_before'],
            'exploit_after': cp['exploit_after'],
            'exploit_change': cp['exploit_after'] - cp['exploit_before']
        })
    
    inc_df = pd.DataFrame(increases)
    print(f"\n  Upload frequency increase ratio (mean): {inc_df['upload_increase'].mean():.2f}")
    print(f"  Exploit score before changepoint (mean): {inc_df['exploit_before'].mean():.4f}")
    print(f"  Exploit score after changepoint (mean): {inc_df['exploit_after'].mean():.4f}")
    
    # Paired test: exploit before vs after
    if len(inc_df) > 5:
        t_stat, t_p = stats.ttest_rel(inc_df['exploit_before'], inc_df['exploit_after'])
        print(f"  Paired t-test (exploit before vs after): t={t_stat:.3f}, p={t_p:.4f}")
        
        # Correlation: bigger upload increase → bigger exploit increase?
        r, p = stats.pearsonr(inc_df['upload_increase'], inc_df['exploit_change'])
        print(f"  Correlation (upload_increase ↔ exploit_change): r={r:.3f}, p={p:.4f}")

# ============================================================
# PART 4: Visualizations
# ============================================================
print("\n[4/4] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Behavioral Fingerprinting: Deep Analysis', fontsize=14, fontweight='bold')

# Plot 1: Ablation study
ax = axes[0, 0]
abl_df = pd.DataFrame(ablation_results).T.sort_values('AUC', ascending=True)
colors = ['#2ecc71' if 'All' in idx else '#3498db' if 'Without' in idx else '#e74c3c' if 'Only' in idx else '#f39c12' for idx in abl_df.index]
ax.barh(range(len(abl_df)), abl_df['AUC'].values, color=colors, alpha=0.8)
ax.errorbar(abl_df['AUC'].values, range(len(abl_df)), xerr=abl_df['AUC_std'].values, fmt='none', color='black', capsize=3)
ax.set_yticks(range(len(abl_df)))
ax.set_yticklabels(abl_df.index, fontsize=9)
ax.set_xlabel('AUC-ROC')
ax.set_title('Feature Ablation Study\n(Which features matter most?)')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(0.4, 1.0)

# Plot 2: Within-group scatter (exploit vs upload frequency)
ax = axes[0, 1]
if len(family_df) > 0:
    x_vals = family_df['uploads_per_week'].astype(float)
    y_vals = family_df['exploit_score_mean'].astype(float)
    ax.scatter(x_vals, y_vals, c='#e74c3c', s=100, alpha=0.7, edgecolors='white')
    # Add channel labels
    for i, (idx, row) in enumerate(family_df.iterrows()):
        ax.annotate(idx[:10], (x_vals.iloc[i], y_vals.iloc[i]), fontsize=7, alpha=0.7)
    # Trend line
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    if mask.sum() > 3:
        slope, intercept, r, p, se = stats.linregress(x_vals[mask], y_vals[mask])
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
        ax.text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', transform=ax.transAxes, fontsize=10)
ax.set_xlabel('Uploads per Week')
ax.set_ylabel('Mean Exploit Score')
ax.set_title('Within Family Channels:\nUpload Frequency vs Exploitation')

# Plot 3: Temporal changepoint example (pick channel with biggest change)
ax = axes[1, 0]
if changepoints:
    # Pick the channel with the biggest upload increase
    best_ch = max(changepoints.keys(), key=lambda k: changepoints[k]['upload_increase_ratio'])
    cp = changepoints[best_ch]
    monthly = cp['monthly_data']
    
    x_months = range(len(monthly))
    ax.bar(x_months, monthly['n_videos'].values, alpha=0.6, color='#3498db', label='Monthly Videos')
    if monthly['upload_rolling'].notna().any():
        ax.plot(x_months, monthly['upload_rolling'].values, 'r-', linewidth=2, label='6-month Rolling Avg')
    
    # Mark changepoint
    cp_idx = monthly[monthly['month'].astype(str) == cp['changepoint_month']].index
    if len(cp_idx) > 0:
        ax.axvline(x=cp_idx[0], color='green', linestyle='--', linewidth=2, label=f'Changepoint: {cp["changepoint_month"]}')
    
    ax.set_xlabel('Month Index')
    ax.set_ylabel('Videos per Month')
    ax.set_title(f'Temporal Changepoint: {best_ch}\n(Upload increase: {cp["upload_increase_ratio"]:.1f}x)')
    ax.legend(fontsize=8)

# Plot 4: Exploit score before vs after changepoint
ax = axes[1, 1]
if changepoints and len(inc_df) > 3:
    ax.scatter(inc_df['exploit_before'], inc_df['exploit_after'], s=100, c='#e74c3c', alpha=0.7, edgecolors='white')
    # Diagonal line (no change)
    lim = max(inc_df['exploit_before'].max(), inc_df['exploit_after'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='No change')
    for _, row in inc_df.iterrows():
        ax.annotate(row['channel'][:8], (row['exploit_before'], row['exploit_after']), fontsize=7, alpha=0.7)
    ax.set_xlabel('Exploit Score BEFORE Changepoint')
    ax.set_ylabel('Exploit Score AFTER Changepoint')
    ax.set_title('Exploitation Before vs After\nUpload Frequency Increase')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Exploitation Before vs After')

plt.tight_layout()
plt.savefig('/home/ubuntu/KidInfluencer/analysis_behavioral/figures/deep_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\n--- Ablation Study ---")
print("Key insight: Which feature groups drive the detection?")
for name, res in sorted(ablation_results.items(), key=lambda x: x[1]['AUC'], reverse=True):
    print(f"  {name:35s}: AUC={res['AUC']:.3f}, LOO={res['LOO_Acc']:.3f}")

print(f"\n--- Within-Group Analysis ---")
print(f"  Can behavioral features predict exploitation within family channels?")
print(f"  LOO R² = {loo_r2:.3f}")

if changepoints:
    print(f"\n--- Temporal Changepoints ---")
    print(f"  {len(changepoints)} channels showed upload frequency changepoints")
    print(f"  Mean exploit score change after changepoint: {inc_df['exploit_change'].mean():.4f}")

print("\nDone!")
