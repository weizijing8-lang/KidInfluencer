"""
Combined Regression Analysis: NLP + CV + Metadata → Views
For Paper B: Kidfluencer Content Strategies
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = '/home/ubuntu/KidInfluencer/analysis_v3/combined'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load and merge all data
# ============================================================
print("Loading data...", flush=True)

videos = pd.read_csv('/home/ubuntu/KidInfluencer/data/combined_videos.csv')
cv_features = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/thumbnails/thumbnail_cv_features.csv')
nlp_features = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/nlp/title_nlp_features.csv')
channels = pd.read_csv('/home/ubuntu/KidInfluencer/data/combined_channels.csv')

# Merge
df = videos.merge(cv_features, on='video_id', how='inner')
df = df.merge(nlp_features, on='video_id', how='inner')
df = df.merge(channels[['channel_id', 'subscribers', 'title']].rename(columns={'title': 'channel_title'}), on='channel_id', how='left')

# Clean
df = df[df['views'] > 0].copy()
df['log_views'] = np.log1p(df['views'])
df['log_subs'] = np.log1p(df['subscribers'].fillna(0))
df['duration_minutes'] = df['length_seconds'] / 60.0

print(f"Combined dataset: {len(df)} videos, {df['channel_id'].nunique()} channels", flush=True)

# ============================================================
# Feature groups for analysis
# ============================================================

# NLP features
nlp_cols = ['caps_ratio', 'caps_word_count', 'exclamation_count', 'question_count',
            'emoji_count', 'sentiment_compound', 'sentiment_pos', 'sentiment_neg',
            'has_challenge', 'has_prank', 'has_surprise', 'has_emotional_word',
            'has_family_word', 'has_child_word', 'has_first_person', 'has_clickbait_phrase',
            'word_count', 'title_length', 'special_char_ratio']

# CV features
cv_cols = ['num_faces', 'max_face_ratio', 'has_open_mouth', 'has_text_overlay',
           'brightness', 'saturation', 'colorfulness', 'edge_density', 'contrast']

# Control variables
control_cols = ['log_subs', 'duration_minutes']

# ============================================================
# 1. Descriptive Statistics
# ============================================================
print("\n--- DESCRIPTIVE STATISTICS ---", flush=True)

desc_stats = {
    'Metric': [],
    'Mean': [],
    'Std': [],
    'Min': [],
    'Max': [],
}

key_vars = ['views', 'duration_minutes', 'num_faces', 'caps_ratio', 
            'exclamation_count', 'sentiment_compound', 'brightness', 'colorfulness']
for col in key_vars:
    desc_stats['Metric'].append(col)
    desc_stats['Mean'].append(f"{df[col].mean():.2f}")
    desc_stats['Std'].append(f"{df[col].std():.2f}")
    desc_stats['Min'].append(f"{df[col].min():.2f}")
    desc_stats['Max'].append(f"{df[col].max():.2f}")

desc_df = pd.DataFrame(desc_stats)
print(desc_df.to_string(index=False), flush=True)

# ============================================================
# 2. Correlation Analysis
# ============================================================
print("\n--- TOP CORRELATIONS WITH LOG_VIEWS ---", flush=True)

all_features = nlp_cols + cv_cols + control_cols
corr_with_views = df[all_features + ['log_views']].corr()['log_views'].drop('log_views').sort_values(key=abs, ascending=False)
print(corr_with_views.head(15).to_string(), flush=True)

# Correlation heatmap (top features)
top_features = corr_with_views.head(12).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[top_features + ['log_views']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Matrix: Top Features vs Log Views')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_correlation_heatmap.png')
plt.close()

# ============================================================
# 3. OLS Regression Models (Hierarchical)
# ============================================================
print("\n--- HIERARCHICAL REGRESSION ---", flush=True)

# Model 1: Controls only
X1 = sm.add_constant(df[control_cols].fillna(0))
model1 = sm.OLS(df['log_views'], X1).fit()
print(f"\nModel 1 (Controls only): R² = {model1.rsquared:.4f}, Adj R² = {model1.rsquared_adj:.4f}", flush=True)

# Convert boolean columns to int
bool_cols = df[nlp_cols + cv_cols].select_dtypes(include='bool').columns.tolist()
for col in bool_cols:
    df[col] = df[col].astype(int)

# Model 2: Controls + NLP
X2 = sm.add_constant(df[control_cols + nlp_cols].fillna(0).astype(float))
model2 = sm.OLS(df['log_views'], X2).fit()
print(f"Model 2 (+ NLP): R² = {model2.rsquared:.4f}, Adj R² = {model2.rsquared_adj:.4f}", flush=True)
print(f"  ΔR² from NLP: {model2.rsquared - model1.rsquared:.4f}", flush=True)

# Model 3: Controls + NLP + CV
X3 = sm.add_constant(df[control_cols + nlp_cols + cv_cols].fillna(0).astype(float))
model3 = sm.OLS(df['log_views'], X3).fit()
print(f"Model 3 (+ NLP + CV): R² = {model3.rsquared:.4f}, Adj R² = {model3.rsquared_adj:.4f}", flush=True)
print(f"  ΔR² from CV: {model3.rsquared - model2.rsquared:.4f}", flush=True)

# Print significant coefficients from full model
print(f"\nFull Model (Model 3) - Significant coefficients (p < 0.05):", flush=True)
sig_params = model3.params[model3.pvalues < 0.05].drop('const', errors='ignore')
sig_pvals = model3.pvalues[model3.pvalues < 0.05].drop('const', errors='ignore')
for param in sig_params.index:
    stars = '***' if model3.pvalues[param] < 0.001 else '**' if model3.pvalues[param] < 0.01 else '*'
    print(f"  {param:25s}: β = {model3.params[param]:+.4f}  p = {model3.pvalues[param]:.4f} {stars}", flush=True)

# ============================================================
# 4. Feature Importance (Random Forest)
# ============================================================
print("\n--- FEATURE IMPORTANCE (Random Forest) ---", flush=True)

all_X = df[control_cols + nlp_cols + cv_cols].fillna(0)
y = df['log_views']

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(all_X, y)

# Cross-validation
cv_scores = cross_val_score(rf, all_X, y, cv=5, scoring='r2')
print(f"Random Forest CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", flush=True)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=all_X.columns).sort_values(ascending=False)
print(f"\nTop 15 features:", flush=True)
print(importances.head(15).to_string(), flush=True)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
importances.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance for Predicting Video Views (Random Forest)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_feature_importance.png')
plt.close()

# ============================================================
# 5. Key Comparisons (visual clickbait indicators)
# ============================================================
print("\n--- KEY COMPARISONS ---", flush=True)

# Open mouth vs no open mouth
mouth_views = df[df['has_open_mouth'] == True]['views']
no_mouth_views = df[df['has_open_mouth'] == False]['views']
t_stat, p_val = stats.mannwhitneyu(mouth_views, no_mouth_views, alternative='greater')
print(f"\nOpen mouth thumbnails:", flush=True)
print(f"  With open mouth: median views = {mouth_views.median():.0f} (n={len(mouth_views)})", flush=True)
print(f"  Without: median views = {no_mouth_views.median():.0f} (n={len(no_mouth_views)})", flush=True)
print(f"  Mann-Whitney U p-value: {p_val:.6f}", flush=True)

# Text overlay vs no text
text_views = df[df['has_text_overlay'] == True]['views']
no_text_views = df[df['has_text_overlay'] == False]['views']
t_stat2, p_val2 = stats.mannwhitneyu(text_views, no_text_views, alternative='greater')
print(f"\nText overlay thumbnails:", flush=True)
print(f"  With text: median views = {text_views.median():.0f} (n={len(text_views)})", flush=True)
print(f"  Without: median views = {no_text_views.median():.0f} (n={len(no_text_views)})", flush=True)
print(f"  Mann-Whitney U p-value: {p_val2:.6f}", flush=True)

# ALL CAPS words
caps_views = df[df['caps_word_count'] > 0]['views']
no_caps_views = df[df['caps_word_count'] == 0]['views']
t_stat3, p_val3 = stats.mannwhitneyu(caps_views, no_caps_views, alternative='greater')
print(f"\nALL CAPS in title:", flush=True)
print(f"  With CAPS: median views = {caps_views.median():.0f} (n={len(caps_views)})", flush=True)
print(f"  Without: median views = {no_caps_views.median():.0f} (n={len(no_caps_views)})", flush=True)
print(f"  Mann-Whitney U p-value: {p_val3:.6f}", flush=True)

# Exclamation marks
excl_views = df[df['exclamation_count'] > 0]['views']
no_excl_views = df[df['exclamation_count'] == 0]['views']
t_stat4, p_val4 = stats.mannwhitneyu(excl_views, no_excl_views, alternative='greater')
print(f"\nExclamation marks in title:", flush=True)
print(f"  With !: median views = {excl_views.median():.0f} (n={len(excl_views)})", flush=True)
print(f"  Without: median views = {no_excl_views.median():.0f} (n={len(no_excl_views)})", flush=True)
print(f"  Mann-Whitney U p-value: {p_val4:.6f}", flush=True)

# ============================================================
# 6. Visualization: Clickbait strategies comparison
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Open mouth
strategies = ['Open Mouth\nThumbnail', 'Text Overlay\nThumbnail', 'ALL CAPS\nin Title', 'Exclamation!\nin Title']
medians_with = [mouth_views.median(), text_views.median(), caps_views.median(), excl_views.median()]
medians_without = [no_mouth_views.median(), no_text_views.median(), no_caps_views.median(), no_excl_views.median()]

ax = axes[0, 0]
data_mouth = [df[df['has_open_mouth']==False]['log_views'], df[df['has_open_mouth']==True]['log_views']]
ax.boxplot(data_mouth, labels=['No', 'Yes'])
ax.set_title('Open Mouth in Thumbnail')
ax.set_ylabel('Log Views')

ax = axes[0, 1]
data_text = [df[df['has_text_overlay']==False]['log_views'], df[df['has_text_overlay']==True]['log_views']]
ax.boxplot(data_text, labels=['No', 'Yes'])
ax.set_title('Text Overlay in Thumbnail')

ax = axes[1, 0]
data_caps = [df[df['caps_word_count']==0]['log_views'], df[df['caps_word_count']>0]['log_views']]
ax.boxplot(data_caps, labels=['No', 'Yes'])
ax.set_title('ALL CAPS Words in Title')
ax.set_ylabel('Log Views')

ax = axes[1, 1]
data_excl = [df[df['exclamation_count']==0]['log_views'], df[df['exclamation_count']>0]['log_views']]
ax.boxplot(data_excl, labels=['No', 'Yes'])
ax.set_title('Exclamation Marks in Title')

plt.suptitle('Content Strategy Indicators vs Video Views', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_strategy_comparisons.png')
plt.close()

# ============================================================
# 7. Regression coefficients visualization
# ============================================================

# Plot significant coefficients from Model 3
sig_mask = model3.pvalues < 0.05
sig_coefs = model3.params[sig_mask].drop('const', errors='ignore')
sig_ci = model3.conf_int().loc[sig_coefs.index]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(sig_coefs))
ax.barh(y_pos, sig_coefs.values, color=['green' if v > 0 else 'red' for v in sig_coefs.values], alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_coefs.index)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient (effect on log views)')
ax.set_title('Significant Predictors of Video Views (OLS, p < 0.05)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_regression_coefficients.png')
plt.close()

# ============================================================
# 8. Channel-level aggregation
# ============================================================
print("\n--- CHANNEL-LEVEL ANALYSIS ---", flush=True)

channel_agg = df.groupby(['channel_id', 'channel_title']).agg({
    'views': 'median',
    'caps_ratio': 'mean',
    'exclamation_count': 'mean',
    'has_open_mouth': 'mean',
    'has_text_overlay': 'mean',
    'num_faces': 'mean',
    'sentiment_compound': 'mean',
    'colorfulness': 'mean',
    'has_challenge': 'mean',
    'has_emotional_word': 'mean',
    'has_clickbait_phrase': 'mean',
    'emoji_count': 'mean',
    'subscribers': 'first',
}).reset_index()

# Clickbait intensity score (composite)
channel_agg['clickbait_intensity'] = (
    channel_agg['caps_ratio'] * 0.2 +
    (channel_agg['exclamation_count'] / channel_agg['exclamation_count'].max()) * 0.2 +
    channel_agg['has_open_mouth'] * 0.2 +
    channel_agg['has_text_overlay'] * 0.2 +
    channel_agg['has_clickbait_phrase'] * 0.2
)

# Top clickbait channels
top_clickbait = channel_agg.nlargest(15, 'clickbait_intensity')
print("\nTop 15 channels by clickbait intensity:", flush=True)
for _, row in top_clickbait.iterrows():
    print(f"  {row['channel_title']:30s} | intensity={row['clickbait_intensity']:.3f} | subs={row['subscribers']:.0f}", flush=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_clickbait)), top_clickbait['clickbait_intensity'].values, color='coral')
ax.set_yticks(range(len(top_clickbait)))
ax.set_yticklabels(top_clickbait['channel_title'].values)
ax.set_xlabel('Clickbait Intensity Score')
ax.set_title('Top 15 Channels by Clickbait Intensity')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_top_clickbait_channels.png')
plt.close()

# ============================================================
# 9. Save summary
# ============================================================

summary = {
    'dataset': {
        'n_videos': len(df),
        'n_channels': df['channel_id'].nunique(),
    },
    'model_performance': {
        'model1_controls_R2': round(model1.rsquared, 4),
        'model2_nlp_R2': round(model2.rsquared, 4),
        'model3_full_R2': round(model3.rsquared, 4),
        'delta_R2_nlp': round(model2.rsquared - model1.rsquared, 4),
        'delta_R2_cv': round(model3.rsquared - model2.rsquared, 4),
        'random_forest_cv_R2': round(cv_scores.mean(), 4),
    },
    'key_findings': {
        'open_mouth_effect': f"median views {mouth_views.median():.0f} vs {no_mouth_views.median():.0f}, p={p_val:.6f}",
        'text_overlay_effect': f"median views {text_views.median():.0f} vs {no_text_views.median():.0f}, p={p_val2:.6f}",
        'caps_effect': f"median views {caps_views.median():.0f} vs {no_caps_views.median():.0f}, p={p_val3:.6f}",
        'exclamation_effect': f"median views {excl_views.median():.0f} vs {no_excl_views.median():.0f}, p={p_val4:.6f}",
    }
}

import json
with open(f'{OUTPUT_DIR}/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save full model summary
with open(f'{OUTPUT_DIR}/model3_summary.txt', 'w') as f:
    f.write(model3.summary().as_text())

channel_agg.to_csv(f'{OUTPUT_DIR}/channel_clickbait_scores.csv', index=False)

print(f"\n--- ALL DONE ---", flush=True)
print(f"Outputs saved to {OUTPUT_DIR}/", flush=True)
