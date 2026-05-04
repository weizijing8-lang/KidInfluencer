"""
Behavioral Fingerprinting Pipeline for Kidfluencer Detection
=============================================================
Extracts temporal and metadata-based behavioral features from YouTube channels
to test whether kidfluencer channels can be distinguished from adult channels
purely from behavioral patterns (no content analysis needed).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/home/ubuntu/KidInfluencer/analysis_behavioral', exist_ok=True)
os.makedirs('/home/ubuntu/KidInfluencer/analysis_behavioral/figures', exist_ok=True)

print("=" * 60)
print("BEHAVIORAL FINGERPRINTING PIPELINE")
print("=" * 60)

# ============================================================
# STEP 1: Load and prepare data
# ============================================================
print("\n[1/5] Loading data...")

# Load V4 full results (98K videos with timestamps)
v4 = pd.read_csv('/home/ubuntu/KidInfluencer/data/results_v4/full_results_v4.csv',
                 usecols=['id', 'title', 'publishedAt', 'channel_short_name', 
                          'channel_category', 'viewCount', 'likeCount', 
                          'commentCount', 'exploit_score_v4'])

# Parse timestamps
v4['publishedAt'] = pd.to_datetime(v4['publishedAt'], errors='coerce')
v4 = v4.dropna(subset=['publishedAt'])

# Clean categories (remove noise rows)
v4 = v4[v4['channel_category'].isin(['adult', 'family'])]
print(f"  Videos after cleaning: {len(v4)}")
print(f"  Channels: {v4['channel_short_name'].nunique()}")
print(f"  Date range: {v4['publishedAt'].min()} to {v4['publishedAt'].max()}")

# ============================================================
# STEP 2: Extract Behavioral Features per Channel
# ============================================================
print("\n[2/5] Extracting behavioral features...")

def extract_behavioral_features(channel_df):
    """Extract comprehensive behavioral fingerprint for a channel."""
    features = {}
    
    # Sort by date
    df = channel_df.sort_values('publishedAt')
    
    # --- TEMPORAL PATTERNS ---
    # Upload frequency
    if len(df) > 1:
        date_range = (df['publishedAt'].max() - df['publishedAt'].min()).days
        if date_range > 0:
            features['uploads_per_week'] = len(df) / (date_range / 7)
            features['uploads_per_month'] = len(df) / (date_range / 30)
        else:
            features['uploads_per_week'] = 0
            features['uploads_per_month'] = 0
    else:
        features['uploads_per_week'] = 0
        features['uploads_per_month'] = 0
    
    # Inter-upload intervals
    intervals = df['publishedAt'].diff().dt.total_seconds() / 3600  # hours
    intervals = intervals.dropna()
    if len(intervals) > 0:
        features['interval_mean_hours'] = intervals.mean()
        features['interval_median_hours'] = intervals.median()
        features['interval_std_hours'] = intervals.std()
        features['interval_cv'] = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
        features['interval_min_hours'] = intervals.min()
        features['interval_max_hours'] = intervals.max()
        # Burstiness: ratio of very short intervals
        features['burst_ratio'] = (intervals < 24).mean()  # uploads within 24h of each other
    else:
        for k in ['interval_mean_hours', 'interval_median_hours', 'interval_std_hours',
                  'interval_cv', 'interval_min_hours', 'interval_max_hours', 'burst_ratio']:
            features[k] = 0
    
    # Day-of-week distribution
    dow = df['publishedAt'].dt.dayofweek  # 0=Monday, 6=Sunday
    features['weekend_ratio'] = ((dow >= 5).sum()) / len(dow) if len(dow) > 0 else 0
    features['dow_entropy'] = stats.entropy(dow.value_counts(normalize=True).values) if len(dow) > 0 else 0
    
    # Hour-of-day distribution
    hour = df['publishedAt'].dt.hour
    features['hour_mean'] = hour.mean()
    features['hour_std'] = hour.std()
    features['evening_ratio'] = ((hour >= 18) | (hour <= 6)).mean()  # evening/night uploads
    features['hour_entropy'] = stats.entropy(hour.value_counts(normalize=True).values) if len(hour) > 0 else 0
    
    # Monthly upload trend (is frequency increasing?)
    df_monthly = df.set_index('publishedAt').resample('M').size()
    if len(df_monthly) > 3:
        x = np.arange(len(df_monthly))
        slope, _, r_value, _, _ = stats.linregress(x, df_monthly.values)
        features['upload_trend_slope'] = slope
        features['upload_trend_r2'] = r_value ** 2
    else:
        features['upload_trend_slope'] = 0
        features['upload_trend_r2'] = 0
    
    # Seasonality: holiday uploads (Dec 24-Jan 1)
    month_day = df['publishedAt'].apply(lambda x: (x.month, x.day))
    holiday_mask = month_day.apply(lambda x: (x[0] == 12 and x[1] >= 24) or (x[0] == 1 and x[1] <= 1))
    features['holiday_upload_ratio'] = holiday_mask.mean() if len(df) > 0 else 0
    
    # --- ENGAGEMENT PATTERNS ---
    features['views_mean'] = df['viewCount'].mean()
    features['views_median'] = df['viewCount'].median()
    features['views_std'] = df['viewCount'].std()
    features['views_cv'] = df['viewCount'].std() / df['viewCount'].mean() if df['viewCount'].mean() > 0 else 0
    features['views_skew'] = df['viewCount'].skew()
    features['views_max_ratio'] = df['viewCount'].max() / df['viewCount'].mean() if df['viewCount'].mean() > 0 else 0
    
    # Like ratio
    features['like_view_ratio_mean'] = (df['likeCount'] / df['viewCount'].replace(0, np.nan)).mean()
    
    # Comment ratio
    features['comment_view_ratio_mean'] = (df['commentCount'] / df['viewCount'].replace(0, np.nan)).mean()
    
    # Engagement trend (are views growing?)
    if len(df) > 10:
        recent_half = df.iloc[len(df)//2:]
        early_half = df.iloc[:len(df)//2]
        features['views_growth_ratio'] = recent_half['viewCount'].mean() / early_half['viewCount'].mean() if early_half['viewCount'].mean() > 0 else 1
    else:
        features['views_growth_ratio'] = 1
    
    # --- TITLE LINGUISTICS (metadata only, no content) ---
    titles = df['title'].dropna().astype(str)
    if len(titles) > 0:
        features['title_length_mean'] = titles.str.len().mean()
        features['title_length_std'] = titles.str.len().std()
        features['title_caps_ratio'] = titles.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)).mean()
        features['title_exclaim_ratio'] = titles.str.contains('!').mean()
        features['title_question_ratio'] = titles.str.contains(r'\?').mean()
        features['title_emoji_ratio'] = titles.apply(lambda x: sum(1 for c in x if ord(c) > 127) / max(len(x), 1)).mean()
        features['title_allcaps_ratio'] = titles.apply(lambda x: x == x.upper() and len(x) > 3).mean()
        features['title_number_ratio'] = titles.str.contains(r'\d').mean()
        # Emotional words (simple proxy)
        emotional_words = ['amazing', 'incredible', 'shocking', 'crazy', 'insane', 'epic',
                          'worst', 'best', 'never', 'always', 'challenge', 'prank', 'surprise',
                          'emotional', 'crying', 'screaming', 'hilarious', 'unbelievable']
        features['title_emotional_ratio'] = titles.apply(
            lambda x: any(w in x.lower() for w in emotional_words)).mean()
        # Word count
        features['title_word_count_mean'] = titles.str.split().str.len().mean()
    else:
        for k in ['title_length_mean', 'title_length_std', 'title_caps_ratio',
                  'title_exclaim_ratio', 'title_question_ratio', 'title_emoji_ratio',
                  'title_allcaps_ratio', 'title_number_ratio', 'title_emotional_ratio',
                  'title_word_count_mean']:
            features[k] = 0
    
    # --- EXPLOITATION SCORE PATTERNS ---
    features['exploit_score_mean'] = df['exploit_score_v4'].mean()
    features['exploit_score_std'] = df['exploit_score_v4'].std()
    features['exploit_score_max'] = df['exploit_score_v4'].max()
    features['high_exploit_ratio'] = (df['exploit_score_v4'] > 0.1).mean()
    
    # --- SCALE FEATURES ---
    features['total_videos'] = len(df)
    features['channel_age_days'] = date_range if len(df) > 1 else 0
    
    return features


# Extract features for all channels
channel_features = {}
for channel, group in v4.groupby('channel_short_name'):
    if len(group) >= 20:  # Need at least 20 videos for meaningful patterns
        features = extract_behavioral_features(group)
        features['category'] = group['channel_category'].iloc[0]
        channel_features[channel] = features

feature_df = pd.DataFrame(channel_features).T
feature_df.index.name = 'channel'
print(f"  Extracted features for {len(feature_df)} channels")
print(f"  Feature dimensions: {feature_df.shape[1] - 1}")  # minus category
print(f"  Family channels: {(feature_df['category'] == 'family').sum()}")
print(f"  Adult channels: {(feature_df['category'] == 'adult').sum()}")

# Save features
feature_df.to_csv('/home/ubuntu/KidInfluencer/analysis_behavioral/channel_behavioral_features.csv')

# ============================================================
# STEP 3: Unsupervised Clustering
# ============================================================
print("\n[3/5] Unsupervised clustering...")

# Prepare feature matrix (exclude category and non-numeric)
numeric_cols = [c for c in feature_df.columns if c != 'category']
X = feature_df[numeric_cols].fillna(0).values
categories = feature_df['category'].values
channel_names = feature_df.index.values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=min(10, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA explained variance (first 5): {pca.explained_variance_ratio_[:5].round(3)}")
print(f"  Total variance explained (10 PCs): {pca.explained_variance_ratio_.sum():.3f}")

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(X_scaled)-1))
X_tsne = tsne.fit_transform(X_scaled)

# K-Means clustering (k=2, 3, 4)
best_silhouette = -1
best_k = 2
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  K={k}: silhouette={sil:.3f}")
    if sil > best_silhouette:
        best_silhouette = sil
        best_k = k

# Final clustering with best k
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Check cluster-category alignment
print(f"\n  Best K={best_k}, silhouette={best_silhouette:.3f}")
print(f"  Cluster-Category contingency table:")
contingency = pd.crosstab(
    pd.Series(cluster_labels, name='Cluster'),
    pd.Series(categories, name='Category')
)
print(contingency)

# Chi-squared test for independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"  Chi-squared: {chi2:.2f}, p={p_value:.4f}")

# ============================================================
# STEP 4: Supervised Classification
# ============================================================
print("\n[4/5] Supervised classification (behavioral features only)...")

y = (categories == 'family').astype(int)

# Random Forest with cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
cv_scores = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')
print(f"  Random Forest 5-fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

cv_acc = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy')
print(f"  Random Forest 5-fold CV Accuracy: {cv_acc.mean():.3f} (+/- {cv_acc.std():.3f})")

cv_auc = cross_val_score(rf, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
print(f"  Random Forest 5-fold CV AUC: {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
cv_scores_gb = cross_val_score(gb, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1')
print(f"  Gradient Boosting 5-fold CV F1: {cv_scores_gb.mean():.3f} (+/- {cv_scores_gb.std():.3f})")

cv_auc_gb = cross_val_score(gb, X_scaled, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
print(f"  Gradient Boosting 5-fold CV AUC: {cv_auc_gb.mean():.3f} (+/- {cv_auc_gb.std():.3f})")

# Feature importance
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
print(f"\n  Top 15 most discriminative features:")
for feat, imp in importances.head(15).items():
    print(f"    {feat}: {imp:.4f}")

# Leave-One-Out for small sample validation
loo = LeaveOneOut()
loo_correct = 0
loo_total = 0
for train_idx, test_idx in loo.split(X_scaled):
    rf_loo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_loo.fit(X_scaled[train_idx], y[train_idx])
    pred = rf_loo.predict(X_scaled[test_idx])
    loo_correct += (pred == y[test_idx]).sum()
    loo_total += 1
print(f"\n  Leave-One-Out Accuracy: {loo_correct/loo_total:.3f} ({loo_correct}/{loo_total})")

# ============================================================
# STEP 5: Visualization
# ============================================================
print("\n[5/5] Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Behavioral Fingerprinting: Kidfluencer vs Adult Channels', fontsize=14, fontweight='bold')

# Plot 1: t-SNE colored by category
ax = axes[0, 0]
colors = ['#e74c3c' if c == 'family' else '#3498db' for c in categories]
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
ax.set_title('t-SNE: Behavioral Space\n(Red=Family, Blue=Adult)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# Plot 2: t-SNE colored by cluster
ax = axes[0, 1]
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
ax.set_title(f't-SNE: K-Means Clusters (K={best_k})\nSilhouette={best_silhouette:.3f}')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax)

# Plot 3: Feature importance
ax = axes[0, 2]
top_features = importances.head(12)
colors_fi = ['#e74c3c' if 'exploit' in f or 'emotional' in f else '#3498db' for f in top_features.index]
ax.barh(range(len(top_features)), top_features.values, color=colors_fi)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features.index, fontsize=8)
ax.set_xlabel('Feature Importance')
ax.set_title('Top Discriminative Features\n(RF Importance)')
ax.invert_yaxis()

# Plot 4: Upload frequency comparison
ax = axes[1, 0]
family_mask = categories == 'family'
data_upload = [feature_df.loc[family_mask, 'uploads_per_week'].astype(float).dropna().values,
               feature_df.loc[~family_mask, 'uploads_per_week'].astype(float).dropna().values]
bp = ax.boxplot(data_upload, labels=['Family/Kid', 'Adult'], patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')
ax.set_ylabel('Uploads per Week')
ax.set_title('Upload Frequency')
# Add Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(data_upload[0], data_upload[1], alternative='two-sided')
ax.text(0.5, 0.95, f'Mann-Whitney p={u_p:.4f}', transform=ax.transAxes, ha='center', fontsize=9)

# Plot 5: Weekend ratio comparison
ax = axes[1, 1]
data_weekend = [feature_df.loc[family_mask, 'weekend_ratio'].astype(float).dropna().values,
                feature_df.loc[~family_mask, 'weekend_ratio'].astype(float).dropna().values]
bp = ax.boxplot(data_weekend, labels=['Family/Kid', 'Adult'], patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')
ax.set_ylabel('Weekend Upload Ratio')
ax.set_title('Weekend Upload Pattern')
ax.axhline(y=2/7, color='gray', linestyle='--', alpha=0.5, label='Expected (2/7)')
ax.legend()
u_stat2, u_p2 = stats.mannwhitneyu(data_weekend[0], data_weekend[1], alternative='two-sided')
ax.text(0.5, 0.95, f'Mann-Whitney p={u_p2:.4f}', transform=ax.transAxes, ha='center', fontsize=9)

# Plot 6: Title emotional ratio comparison
ax = axes[1, 2]
data_emotional = [feature_df.loc[family_mask, 'title_emotional_ratio'].astype(float).dropna().values,
                  feature_df.loc[~family_mask, 'title_emotional_ratio'].astype(float).dropna().values]
bp = ax.boxplot(data_emotional, labels=['Family/Kid', 'Adult'], patch_artist=True)
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')
ax.set_ylabel('Emotional Title Ratio')
ax.set_title('Emotional Language in Titles')
u_stat3, u_p3 = stats.mannwhitneyu(data_emotional[0], data_emotional[1], alternative='two-sided')
ax.text(0.5, 0.95, f'Mann-Whitney p={u_p3:.4f}', transform=ax.transAxes, ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/KidInfluencer/analysis_behavioral/figures/behavioral_fingerprint_overview.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional: Radar chart of behavioral differences
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Select key features for radar
radar_features = ['uploads_per_week', 'weekend_ratio', 'burst_ratio', 
                  'title_emotional_ratio', 'title_exclaim_ratio', 'title_caps_ratio',
                  'exploit_score_mean', 'views_cv', 'interval_cv', 'holiday_upload_ratio']

# Normalize each feature to 0-1 range
family_means = feature_df.loc[family_mask, radar_features].mean()
adult_means = feature_df.loc[~family_mask, radar_features].mean()

# Min-max normalize
all_vals = pd.concat([family_means, adult_means], axis=1)
min_vals = all_vals.min(axis=1)
max_vals = all_vals.max(axis=1)
range_vals = max_vals - min_vals
range_vals = range_vals.replace(0, 1)

family_norm = ((family_means - min_vals) / range_vals).values
adult_norm = ((adult_means - min_vals) / range_vals).values

# Plot radar
angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]
family_norm = np.append(family_norm, family_norm[0])
adult_norm = np.append(adult_norm, adult_norm[0])

ax.plot(angles, family_norm, 'o-', linewidth=2, label='Family/Kid Channels', color='#e74c3c')
ax.fill(angles, family_norm, alpha=0.15, color='#e74c3c')
ax.plot(angles, adult_norm, 'o-', linewidth=2, label='Adult Channels', color='#3498db')
ax.fill(angles, adult_norm, alpha=0.15, color='#3498db')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_features, fontsize=9)
ax.set_title('Behavioral Fingerprint Radar\nFamily/Kid vs Adult Channels', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.savefig('/home/ubuntu/KidInfluencer/analysis_behavioral/figures/behavioral_radar.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nDataset: {len(v4)} videos across {len(feature_df)} channels")
print(f"  Family/Kid channels: {family_mask.sum()}")
print(f"  Adult channels: {(~family_mask).sum()}")
print(f"\nBehavioral features extracted: {len(numeric_cols)}")

print(f"\n--- Unsupervised Clustering ---")
print(f"  Best K: {best_k}, Silhouette: {best_silhouette:.3f}")
print(f"  Chi-squared (cluster vs category): {chi2:.2f}, p={p_value:.4f}")

print(f"\n--- Supervised Classification (metadata-only) ---")
print(f"  Random Forest CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"  Random Forest CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
print(f"  Gradient Boosting CV F1: {cv_scores_gb.mean():.3f} +/- {cv_scores_gb.std():.3f}")
print(f"  Gradient Boosting CV AUC: {cv_auc_gb.mean():.3f} +/- {cv_auc_gb.std():.3f}")
print(f"  Leave-One-Out Accuracy: {loo_correct/loo_total:.3f}")

print(f"\n--- Key Behavioral Differences ---")
key_comparisons = ['uploads_per_week', 'weekend_ratio', 'burst_ratio', 
                   'title_emotional_ratio', 'exploit_score_mean', 'interval_mean_hours']
for feat in key_comparisons:
    fam_val = feature_df.loc[family_mask, feat].median()
    adult_val = feature_df.loc[~family_mask, feat].median()
    u, p = stats.mannwhitneyu(
        feature_df.loc[family_mask, feat].astype(float).dropna().values,
        feature_df.loc[~family_mask, feat].astype(float).dropna().values,
        alternative='two-sided'
    )
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {feat:30s}: Family={fam_val:.4f}, Adult={adult_val:.4f}, p={p:.4f} {sig}")

print("\nDone! Figures saved to analysis_behavioral/figures/")
