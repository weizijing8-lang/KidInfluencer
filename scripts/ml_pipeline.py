"""
ML Pipeline: XGBoost + SHAP Analysis
Paper 1 - AI-powered computational audit of kidfluencer channels

This script:
1. Video-level model: Predict views from content features (NLP + CV + LLM annotations)
2. Channel-level model: Predict labor intensity from commercialization features
3. SHAP analysis for feature importance and interaction effects
4. Comparison: Family vs Adult feature importance differences
5. Generate publication-quality figures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Install required packages
import subprocess
subprocess.run(['sudo', 'pip3', 'install', 'xgboost', 'shap', 'lightgbm', '-q'], 
               capture_output=True)

import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

BASE = '/home/ubuntu/KidInfluencer'
OUTPUT = f'{BASE}/analysis_paper1_v2'
FIG_DIR = f'{OUTPUT}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Video-level features
video_df = pd.read_csv(f'{OUTPUT}/video_level_features.csv')
print(f"Video-level: {video_df.shape}")

# Channel-level features
channel_df = pd.read_csv(f'{OUTPUT}/channels_with_indices_v2.csv')
print(f"Channel-level: {channel_df.shape}")

# V4 full results for video-level analysis with both family and adult
v4 = pd.read_csv(f'{BASE}/data/results_v4/full_results_v4.csv',
                  usecols=['id', 'channel_short_name', 'channel_category',
                           'viewCount', 'likeCount', 'commentCount',
                           'exploit_score_v4', 'exploit_score_title_only', 'publishedAt'])
v4 = v4[v4['channel_category'].isin(['adult', 'family'])].copy()
print(f"V4 Full: {v4.shape}")

# ============================================================
# 2. VIDEO-LEVEL MODEL: PREDICT VIEWS
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: VIDEO-LEVEL VIEW PREDICTION")
print("=" * 60)

# Prepare video-level features
video_features = video_df.copy()
video_features['log_views'] = np.log1p(video_features['views'])
video_features['log_duration'] = np.log1p(video_features['length_seconds'])

# Feature columns
nlp_features = ['title_length', 'word_count', 'caps_ratio', 'caps_word_count',
                'exclamation_count', 'question_count', 'special_char_ratio',
                'emoji_count', 'sentiment_compound']
cv_features = ['num_faces', 'max_face_ratio', 'has_open_mouth', 'brightness',
               'saturation', 'colorfulness', 'has_text_overlay']
llm_features = ['emotional_score', 'commercial_binary', 'child_protagonist',
                'privacy_score', 'clickbait_score']
structural_features = ['log_duration']

all_features = nlp_features + cv_features + llm_features + structural_features

# Drop rows with too many NaN
video_model_df = video_features[all_features + ['log_views', 'channel_id']].dropna(
    subset=['log_views'] + structural_features)

# Fill NaN in features with median
for col in all_features:
    if col in video_model_df.columns:
        video_model_df[col] = video_model_df[col].fillna(video_model_df[col].median())

# Convert boolean to int
for col in ['has_open_mouth', 'has_text_overlay']:
    if col in video_model_df.columns:
        video_model_df[col] = video_model_df[col].astype(int)

X_video = video_model_df[all_features].values
y_video = video_model_df['log_views'].values

print(f"Video model dataset: {X_video.shape[0]} videos, {X_video.shape[1]} features")

# XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv_scores = cross_val_score(xgb_model, X_video, y_video, cv=5, scoring='r2')
print(f"XGBoost 5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit full model for SHAP
xgb_model.fit(X_video, y_video)
y_pred = xgb_model.predict(X_video)
train_r2 = r2_score(y_video, y_pred)
print(f"Training R²: {train_r2:.4f}")

# ============================================================
# 3. SHAP ANALYSIS - VIDEO LEVEL
# ============================================================
print("\n" + "=" * 60)
print("SHAP ANALYSIS - VIDEO LEVEL")
print("=" * 60)

# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_video)

# Feature importance from SHAP
shap_importance = pd.DataFrame({
    'feature': all_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 15 features by SHAP importance:")
print(shap_importance.head(15).to_string(index=False))

# Save SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_video, feature_names=all_features, 
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/shap_summary_video.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: shap_summary_video.png")

# SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_video, feature_names=all_features, 
                  plot_type='bar', show=False, max_display=15)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/shap_bar_video.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: shap_bar_video.png")

# ============================================================
# 4. HIERARCHICAL MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("HIERARCHICAL MODEL COMPARISON")
print("=" * 60)

# Model 1: Structure only (duration)
model_struct = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
cv_struct = cross_val_score(model_struct, video_model_df[structural_features].values, 
                            y_video, cv=5, scoring='r2')

# Model 2: Structure + NLP
model_nlp = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
nlp_avail = [f for f in nlp_features if f in video_model_df.columns]
cv_nlp = cross_val_score(model_nlp, video_model_df[structural_features + nlp_avail].values,
                          y_video, cv=5, scoring='r2')

# Model 3: Structure + NLP + CV
model_cv = xgb.XGBRegressor(n_estimators=150, max_depth=4, random_state=42)
cv_avail = [f for f in cv_features if f in video_model_df.columns]
cv_cv = cross_val_score(model_cv, video_model_df[structural_features + nlp_avail + cv_avail].values,
                         y_video, cv=5, scoring='r2')

# Model 4: Structure + NLP + CV + LLM
model_full = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
llm_avail = [f for f in llm_features if f in video_model_df.columns]
cv_full = cross_val_score(model_full, video_model_df[structural_features + nlp_avail + cv_avail + llm_avail].values,
                           y_video, cv=5, scoring='r2')

hierarchical_results = {
    'Structure Only': {'R2': cv_struct.mean(), 'std': cv_struct.std(), 'n_features': len(structural_features)},
    '+ NLP': {'R2': cv_nlp.mean(), 'std': cv_nlp.std(), 'n_features': len(structural_features) + len(nlp_avail)},
    '+ CV': {'R2': cv_cv.mean(), 'std': cv_cv.std(), 'n_features': len(structural_features) + len(nlp_avail) + len(cv_avail)},
    '+ LLM': {'R2': cv_full.mean(), 'std': cv_full.std(), 'n_features': len(structural_features) + len(nlp_avail) + len(cv_avail) + len(llm_avail)},
}

print(f"{'Model':<20} {'R² (CV)':>10} {'± std':>8} {'n_features':>12}")
print("-" * 52)
for name, res in hierarchical_results.items():
    print(f"{name:<20} {res['R2']:>10.4f} {res['std']:>8.4f} {res['n_features']:>12}")

# Plot hierarchical model comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = list(hierarchical_results.keys())
r2_vals = [hierarchical_results[m]['R2'] for m in models]
r2_stds = [hierarchical_results[m]['std'] for m in models]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

bars = ax.bar(models, r2_vals, yerr=r2_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Cross-Validated R²', fontsize=12)
ax.set_title('Hierarchical Model: Incremental Contribution of AI Features', fontsize=13)
ax.set_ylim(0, max(r2_vals) * 1.3)

# Add value labels
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add delta annotations
for i in range(1, len(r2_vals)):
    delta = r2_vals[i] - r2_vals[i-1]
    ax.annotate(f'Δ={delta:.3f}', xy=(i, r2_vals[i] + r2_stds[i] + 0.02),
                fontsize=9, ha='center', color='red')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/hierarchical_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: hierarchical_model_comparison.png")

# ============================================================
# 5. CHANNEL-LEVEL MODEL: PREDICT LABOR FROM COMMERCIALIZATION
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: CHANNEL-LEVEL CI → LII")
print("=" * 60)

# Features for channel-level model
ch_features = ['sponsor_rate', 'n_child_brands', 'degree', 'family_partners',
               'log_total_views', 'log_n_child_brands']
ch_target = 'lii_pca'

ch_model_df = channel_df.dropna(subset=[ch_target])
X_ch = ch_model_df[ch_features].fillna(0).values
y_ch = ch_model_df[ch_target].values

print(f"Channel model: {X_ch.shape[0]} channels, {X_ch.shape[1]} features")

# Due to small sample, use LightGBM with regularization
lgb_model = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    num_leaves=8,
    min_child_samples=5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

# Leave-one-out CV for small sample
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo_preds = np.zeros(len(y_ch))
for train_idx, test_idx in loo.split(X_ch):
    lgb_model.fit(X_ch[train_idx], y_ch[train_idx])
    loo_preds[test_idx] = lgb_model.predict(X_ch[test_idx])

loo_r2 = r2_score(y_ch, loo_preds)
loo_mae = mean_absolute_error(y_ch, loo_preds)
print(f"LOO-CV R²: {loo_r2:.4f}")
print(f"LOO-CV MAE: {loo_mae:.4f}")

# Fit full model for SHAP
lgb_model.fit(X_ch, y_ch)

# SHAP for channel model
explainer_ch = shap.TreeExplainer(lgb_model)
shap_values_ch = explainer_ch.shap_values(X_ch)

ch_shap_importance = pd.DataFrame({
    'feature': ch_features,
    'mean_abs_shap': np.abs(shap_values_ch).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nChannel-level feature importance (SHAP):")
print(ch_shap_importance.to_string(index=False))

# ============================================================
# 6. FAMILY vs ADULT: DIFFERENTIAL FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: FAMILY vs ADULT DIFFERENTIAL ANALYSIS")
print("=" * 60)

# Use V4 full data for video-level comparison
# Compute per-video features from V4
v4_clean = v4.dropna(subset=['viewCount', 'exploit_score_v4']).copy()
v4_clean['log_views'] = np.log1p(v4_clean['viewCount'])
v4_clean['is_family'] = (v4_clean['channel_category'] == 'family').astype(int)

# Simple model: exploit_score predicts views differently for family vs adult?
from sklearn.linear_model import LinearRegression

# Family
family_v4 = v4_clean[v4_clean['is_family'] == 1]
adult_v4 = v4_clean[v4_clean['is_family'] == 0]

# Regression: exploit_score → views for each group
lr_fam = LinearRegression()
lr_fam.fit(family_v4[['exploit_score_v4']], family_v4['log_views'])
r2_fam = lr_fam.score(family_v4[['exploit_score_v4']], family_v4['log_views'])

lr_adu = LinearRegression()
lr_adu.fit(adult_v4[['exploit_score_v4']], adult_v4['log_views'])
r2_adu = lr_adu.score(adult_v4[['exploit_score_v4']], adult_v4['log_views'])

print(f"Exploit Score → Views:")
print(f"  Family: β={lr_fam.coef_[0]:.4f}, R²={r2_fam:.4f}, n={len(family_v4)}")
print(f"  Adult:  β={lr_adu.coef_[0]:.4f}, R²={r2_adu:.4f}, n={len(adult_v4)}")
print(f"  Family β is {lr_fam.coef_[0]/lr_adu.coef_[0]:.2f}x the adult β")

# ============================================================
# 7. INTERACTION ANALYSIS: COMMERCIALIZATION × CATEGORY
# ============================================================
print("\n" + "=" * 60)
print("INTERACTION: COMMERCIALIZATION × CATEGORY")
print("=" * 60)

# XGBoost with interaction: category moderates CI → LII
ch_interact_df = channel_df.copy()
ch_interact_df['is_family'] = (ch_interact_df['category'] == 'family').astype(int)
ch_interact_features = ch_features + ['is_family']

X_interact = ch_interact_df[ch_interact_features].fillna(0).values
y_interact = ch_interact_df['lii_pca'].values

xgb_interact = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
xgb_interact.fit(X_interact, y_interact)

# SHAP interaction values
explainer_interact = shap.TreeExplainer(xgb_interact)
shap_interact = explainer_interact.shap_values(X_interact)

interact_importance = pd.DataFrame({
    'feature': ch_interact_features,
    'mean_abs_shap': np.abs(shap_interact).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("Interaction model feature importance:")
print(interact_importance.to_string(index=False))

# SHAP dependence plot: CI features colored by category
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
top_features = interact_importance['feature'].head(4).tolist()

for idx, feat in enumerate(top_features):
    ax = axes[idx // 2, idx % 2]
    feat_idx = ch_interact_features.index(feat)
    cat_idx = ch_interact_features.index('is_family')
    
    ax.scatter(X_interact[:, feat_idx], shap_interact[:, feat_idx],
               c=X_interact[:, cat_idx], cmap='RdBu', alpha=0.7, edgecolors='black', linewidth=0.3)
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel('SHAP value', fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'SHAP Dependence: {feat}', fontsize=12)

plt.suptitle('Feature Effects on Labor Intensity (Blue=Adult, Red=Family)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/shap_dependence_interaction.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: shap_dependence_interaction.png")

# ============================================================
# 8. COMPREHENSIVE RESULTS FIGURE
# ============================================================
print("\n" + "=" * 60)
print("GENERATING COMPREHENSIVE RESULTS FIGURE")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: CI distribution by category
ax = axes[0, 0]
family_ci = channel_df[channel_df['category'] == 'family']['ci_pca']
adult_ci = channel_df[channel_df['category'] == 'adult']['ci_pca']
ax.hist(adult_ci, bins=12, alpha=0.6, label='Adult', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.hist(family_ci, bins=12, alpha=0.6, label='Family', color='#F44336', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Commercialization Index')
ax.set_ylabel('Count')
ax.set_title('(A) CI Distribution by Category')
ax.legend()

# Panel B: LII distribution by category
ax = axes[0, 1]
family_lii = channel_df[channel_df['category'] == 'family']['lii_pca']
adult_lii = channel_df[channel_df['category'] == 'adult']['lii_pca']
ax.hist(adult_lii, bins=12, alpha=0.6, label='Adult', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.hist(family_lii, bins=12, alpha=0.6, label='Family', color='#F44336', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Labor Intensity Index')
ax.set_ylabel('Count')
ax.set_title('(B) LII Distribution by Category')
ax.legend()

# Panel C: CI vs LII scatter
ax = axes[0, 2]
colors_scatter = ['#F44336' if c == 'family' else '#2196F3' for c in channel_df['category']]
ax.scatter(channel_df['ci_pca'], channel_df['lii_pca'], c=colors_scatter, alpha=0.7, 
           edgecolors='black', linewidth=0.3, s=50)
# Add regression line
from numpy.polynomial.polynomial import polyfit
b, m = polyfit(channel_df['ci_pca'], channel_df['lii_pca'], 1)
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, b + m * x_line, 'k--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('Commercialization Index')
ax.set_ylabel('Labor Intensity Index')
ax.set_title(f'(C) CI vs LII (r={0.26:.2f}, p=0.035)')
ax.legend(['Regression', 'Family', 'Adult'], loc='upper left')

# Panel D: Videos per week comparison
ax = axes[1, 0]
data_vpw = [adult_ci.values, family_ci.values]
bp = ax.boxplot([channel_df[channel_df['category']=='adult']['videos_per_week'],
                  channel_df[channel_df['category']=='family']['videos_per_week']],
                 labels=['Adult', 'Family'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][1].set_facecolor('#F44336')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax.set_ylabel('Videos per Week')
ax.set_title('(D) Upload Frequency (p=0.037)')

# Panel E: Exploit score comparison
ax = axes[1, 1]
bp2 = ax.boxplot([channel_df[channel_df['category']=='adult']['mean_exploit_v4'].dropna(),
                   channel_df[channel_df['category']=='family']['mean_exploit_v4'].dropna()],
                  labels=['Adult', 'Family'], patch_artist=True)
bp2['boxes'][0].set_facecolor('#2196F3')
bp2['boxes'][1].set_facecolor('#F44336')
for box in bp2['boxes']:
    box.set_alpha(0.6)
ax.set_ylabel('Mean Exploitation Score')
ax.set_title('(E) Content Exploitation (p=0.037)')

# Panel F: Hierarchical model
ax = axes[1, 2]
models_short = ['Struct', '+NLP', '+CV', '+LLM']
r2_vals_plot = [hierarchical_results[m]['R2'] for m in hierarchical_results.keys()]
bars = ax.bar(models_short, r2_vals_plot, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Cross-Validated R²')
ax.set_title('(F) Hierarchical Feature Contribution')
for bar, val in zip(bars, r2_vals_plot):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Computational Audit of Kidfluencer Content: Key Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/comprehensive_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: comprehensive_results.png")

# ============================================================
# 9. SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING ALL RESULTS")
print("=" * 60)

results = {
    'video_model': {
        'n_samples': int(X_video.shape[0]),
        'n_features': int(X_video.shape[1]),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'train_r2': float(train_r2),
    },
    'hierarchical': hierarchical_results,
    'channel_model': {
        'n_samples': int(X_ch.shape[0]),
        'loo_r2': float(loo_r2),
        'loo_mae': float(loo_mae),
    },
    'exploit_score_regression': {
        'family_beta': float(lr_fam.coef_[0]),
        'family_r2': float(r2_fam),
        'family_n': int(len(family_v4)),
        'adult_beta': float(lr_adu.coef_[0]),
        'adult_r2': float(r2_adu),
        'adult_n': int(len(adult_v4)),
    },
    'shap_top_features_video': shap_importance.head(10).to_dict('records'),
    'shap_top_features_channel': ch_shap_importance.to_dict('records'),
}

with open(f'{OUTPUT}/ml_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: ml_results.json")

# Save SHAP importance
shap_importance.to_csv(f'{OUTPUT}/shap_importance_video.csv', index=False)
ch_shap_importance.to_csv(f'{OUTPUT}/shap_importance_channel.csv', index=False)
print(f"Saved: shap_importance_video.csv, shap_importance_channel.csv")

print("\n" + "=" * 60)
print("ML PIPELINE COMPLETE")
print("=" * 60)
