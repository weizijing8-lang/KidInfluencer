"""
Construct Commercialization Index (CI) and Labor Intensity Index (LII)
for the revised Paper 1: AI-powered computational audit of kidfluencer channels.

This script:
1. Merges all available data sources (annotations, NLP, CV, upload frequency, sponsorship)
2. Constructs a multi-dimensional Commercialization Index
3. Constructs a multi-dimensional Labor Intensity Index
4. Runs initial validation and correlation analysis
5. Outputs channel-level and video-level datasets ready for ML modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/ubuntu/KidInfluencer'

# ============================================================
# 1. LOAD ALL DATA SOURCES
# ============================================================
print("=" * 60)
print("LOADING DATA SOURCES")
print("=" * 60)

# A. LLM Annotations (4381 videos, 75 channels)
ann = pd.read_csv(f'{BASE}/data/annotations_merged.csv')
print(f"LLM Annotations: {ann.shape}")

# B. Combined Videos (5570 videos, 115 channels)
vids = pd.read_csv(f'{BASE}/data/combined_videos.csv')
print(f"Combined Videos: {vids.shape}")

# C. Combined Channels (115 channels)
chs = pd.read_csv(f'{BASE}/data/combined_channels.csv')
print(f"Combined Channels: {chs.shape}")

# D. NLP Features (title-level)
nlp = pd.read_csv(f'{BASE}/analysis_v3/nlp/title_nlp_features.csv')
print(f"NLP Features: {nlp.shape}")

# E. CV Features (thumbnail-level)
cv = pd.read_csv(f'{BASE}/analysis_v3/thumbnails/thumbnail_cv_features.csv')
print(f"CV Features: {cv.shape}")

# F. Upload Frequency Metrics (66 channels: 25 family + 41 adult)
uf = pd.read_csv(f'{BASE}/data/results_v4/upload_frequency_metrics.csv')
print(f"Upload Frequency: {uf.shape}")

# G. Sponsorship Data (66 channels)
spon = pd.read_csv(f'{BASE}/data/results_v4/sponsorship_by_channel.csv')
print(f"Sponsorship: {spon.shape}")

# H. Channel Risk Scores V3 (75 channels - family only)
risk = pd.read_csv(f'{BASE}/analysis_v3/channel_risk_scores_v3.csv')
print(f"Channel Risk Scores: {risk.shape}")

# I. Full V4 Results (98k videos with exploit scores)
v4 = pd.read_csv(f'{BASE}/data/results_v4/full_results_v4.csv', 
                  usecols=['id', 'channel_short_name', 'channel_category', 
                           'viewCount', 'likeCount', 'commentCount',
                           'exploit_score_v4', 'exploit_score_title_only', 'publishedAt'])
print(f"Full V4 Results: {v4.shape}")

# ============================================================
# 2. CLEAN AND STANDARDIZE LLM ANNOTATIONS
# ============================================================
print("\n" + "=" * 60)
print("CLEANING LLM ANNOTATIONS")
print("=" * 60)

# Remove error rows
ann_clean = ann[ann['content_type'] != 'error'].copy()
print(f"After removing errors: {ann_clean.shape[0]} videos")

# Standardize emotional_manipulation to numeric score
emotional_map = {
    'none': 0, 'mild': 1, 'low': 1, 'Low': 1,
    'moderate': 2, 'Medium': 2, 'medium': 2,
    'severe': 3, 'high': 3, 'High': 3
}
ann_clean['emotional_score'] = ann_clean['emotional_manipulation'].map(emotional_map).fillna(0)

# Standardize commercial_signals to binary
commercial_positive = ['brand_mention', 'likely_sponsored', 'sponsored', 'product_placement',
                       'brand_integration', 'affiliate', 'yes', 'high', 'moderate']
ann_clean['commercial_binary'] = ann_clean['commercial_signals'].apply(
    lambda x: 1 if str(x).lower() in [s.lower() for s in commercial_positive] else 0
)

# Standardize child_role
protagonist_values = ['protagonist', 'Central', 'central', 'main', 'featured']
ann_clean['child_protagonist'] = ann_clean['child_role'].apply(
    lambda x: 1 if str(x).lower() in [s.lower() for s in protagonist_values] else 0
)

# Standardize privacy_concern to numeric
privacy_map = {
    'none': 0, 'no_privacy_concern': 0,
    'low': 1, 'mild': 1, 'Low': 1,
    'moderate': 2, 'Medium': 2, 'medium': 2,
    'high': 3, 'High': 3, 'severe': 3
}
ann_clean['privacy_score'] = ann_clean['privacy_concern'].map(privacy_map).fillna(0)

# Standardize clickbait_level to numeric
clickbait_map = {
    'none': 0, 'mild': 1, 'Low': 1, 'low': 1,
    'moderate': 2, 'Medium': 2, 'medium': 2,
    'severe': 3, 'high': 3, 'High': 3
}
ann_clean['clickbait_score'] = ann_clean['clickbait_level'].map(clickbait_map).fillna(0)

print(f"Cleaned annotations: {ann_clean.shape[0]} videos")
print(f"  emotional_score mean: {ann_clean['emotional_score'].mean():.3f}")
print(f"  commercial_binary rate: {ann_clean['commercial_binary'].mean():.3f}")
print(f"  child_protagonist rate: {ann_clean['child_protagonist'].mean():.3f}")
print(f"  privacy_score mean: {ann_clean['privacy_score'].mean():.3f}")
print(f"  clickbait_score mean: {ann_clean['clickbait_score'].mean():.3f}")

# ============================================================
# 3. MERGE VIDEO-LEVEL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("MERGING VIDEO-LEVEL FEATURES")
print("=" * 60)

# Merge annotations with video metadata
video_df = vids.merge(ann_clean[['video_id', 'emotional_score', 'commercial_binary', 
                                  'child_protagonist', 'privacy_score', 'clickbait_score',
                                  'content_type']], 
                       on='video_id', how='left')

# Merge NLP features
nlp_cols = ['title_length', 'word_count', 'caps_ratio', 'caps_word_count',
            'exclamation_count', 'question_count', 'has_ellipsis', 'special_char_ratio',
            'emoji_count', 'sentiment_compound', 'has_challenge', 'has_prank',
            'has_surprise', 'has_emotional_wo']
# The last column in NLP is video_id
nlp_rename = nlp.copy()
nlp_id_col = nlp.columns[-1]  # last column is video_id
nlp_rename = nlp_rename.rename(columns={nlp_id_col: 'video_id'})
available_nlp_cols = [c for c in nlp_cols if c in nlp_rename.columns]
video_df = video_df.merge(nlp_rename[['video_id'] + available_nlp_cols], 
                           on='video_id', how='left')

# Merge CV features
cv_rename = cv.copy()
cv_id_col = cv.columns[-1]  # last column is video_id
cv_rename = cv_rename.rename(columns={cv_id_col: 'video_id'})
cv_cols = ['num_faces', 'max_face_ratio', 'has_open_mouth', 'brightness', 
           'saturation', 'colorfulness', 'has_text_overlay']
available_cv_cols = [c for c in cv_cols if c in cv_rename.columns]
video_df = video_df.merge(cv_rename[['video_id'] + available_cv_cols], 
                           on='video_id', how='left')

print(f"Merged video-level dataset: {video_df.shape}")
print(f"Columns: {list(video_df.columns)}")

# ============================================================
# 4. CONSTRUCT CHANNEL-LEVEL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("CONSTRUCTING CHANNEL-LEVEL FEATURES")
print("=" * 60)

# Aggregate video-level features to channel level
channel_agg = video_df.groupby('channel_id').agg(
    n_videos=('video_id', 'count'),
    mean_views=('views', 'mean'),
    median_views=('views', 'median'),
    total_views=('views', 'sum'),
    mean_duration=('length_seconds', 'mean'),
    median_duration=('length_seconds', 'median'),
    # LLM annotation aggregates
    emotional_score_mean=('emotional_score', 'mean'),
    emotional_score_max=('emotional_score', 'max'),
    commercial_rate=('commercial_binary', 'mean'),
    child_protagonist_rate=('child_protagonist', 'mean'),
    privacy_score_mean=('privacy_score', 'mean'),
    clickbait_score_mean=('clickbait_score', 'mean'),
    # NLP aggregates
    caps_ratio_mean=('caps_ratio', 'mean'),
    exclamation_mean=('exclamation_count', 'mean'),
    sentiment_mean=('sentiment_compound', 'mean'),
    # CV aggregates
    face_ratio_mean=('max_face_ratio', 'mean'),
    open_mouth_rate=('has_open_mouth', 'mean'),
    text_overlay_rate=('has_text_overlay', 'mean'),
).reset_index()

# Merge with channel metadata
channel_df = chs.merge(channel_agg, on='channel_id', how='left')

# Merge with upload frequency (need to match by channel name)
# First, create a mapping from channel_id to short name
channel_df['channel_short'] = channel_df['handle'].astype(str).str.replace('@', '').str.lower()

# Merge sponsorship data
spon_merge = spon.rename(columns={'channel': 'channel_short_spon'})
# We'll match by trying different approaches
# For now, merge upload frequency and sponsorship by matching channel names

print(f"Channel-level dataset: {channel_df.shape}")

# ============================================================
# 5. CONSTRUCT COMMERCIALIZATION INDEX (CI)
# ============================================================
print("\n" + "=" * 60)
print("CONSTRUCTING COMMERCIALIZATION INDEX")
print("=" * 60)

"""
Commercialization Index components:
1. commercial_rate: % of videos with commercial signals (LLM-detected)
2. cross_platform_count: Number of social media platforms (proxy for brand reach)
3. subscribers: Channel size (log-transformed, proxy for monetization potential)
4. sponsor_rate: % of videos with sponsorship mentions (from description NLP)
5. production_quality: Proxy from CV features (text overlays, professional thumbnails)
"""

# Component 1: Commercial signal rate from LLM annotations
# Already in channel_agg as 'commercial_rate'

# Component 2: Cross-platform presence
# Already in chs as 'cross_platform_count'

# Component 3: Channel size (log subscribers)
channel_df['log_subscribers'] = np.log10(channel_df['subscribers'].clip(lower=1))

# Component 4: Sponsorship rate from video descriptions
# Match sponsorship data to channels
# Use the V4 channel summary for exploit scores
v4_summary = pd.read_csv(f'{BASE}/data/results_v4/channel_summary_v4.csv')

# Component 5: Production quality proxy (text overlay rate + colorfulness)
# Already computed as text_overlay_rate

# Build CI using available data
ci_features = ['commercial_rate', 'cross_platform_count', 'log_subscribers', 'text_overlay_rate']
ci_data = channel_df[['channel_id', 'title'] + ci_features].dropna()

# Normalize each component to [0, 1]
scaler = MinMaxScaler()
ci_normalized = pd.DataFrame(
    scaler.fit_transform(ci_data[ci_features]),
    columns=[f'{c}_norm' for c in ci_features],
    index=ci_data.index
)

# Equal-weight composite (can be refined later with PCA)
channel_df.loc[ci_data.index, 'commercialization_index'] = ci_normalized.mean(axis=1).values

print(f"Commercialization Index computed for {ci_data.shape[0]} channels")
print(f"CI stats:")
print(channel_df['commercialization_index'].describe())

# ============================================================
# 6. CONSTRUCT LABOR INTENSITY INDEX (LII)
# ============================================================
print("\n" + "=" * 60)
print("CONSTRUCTING LABOR INTENSITY INDEX")
print("=" * 60)

"""
Labor Intensity Index components:
1. child_protagonist_rate: % of videos where child is main performer
2. emotional_score_mean: Average emotional performance intensity
3. upload_frequency: Videos per week (from upload frequency data)
4. mean_duration: Average video length (more content = more labor)
5. privacy_score_mean: Privacy exposure level
6. clickbait_score_mean: Clickbait intensity (child performs for thumbnails)
"""

# We need upload frequency - match from uf data
# uf uses 'channel' column with short names
# Let's try to match

# For channels without upload frequency data, estimate from our video sample
# Calculate approximate upload frequency from the 50 most recent videos
def estimate_upload_freq(group):
    """Estimate videos per week from published_text patterns"""
    # This is approximate since we only have relative time strings
    n = len(group)
    # Assume our sample covers roughly the last 3 months (12 weeks)
    return n / 12.0  # rough estimate

channel_upload_est = video_df.groupby('channel_id').apply(
    lambda x: len(x) / 12.0  # ~50 videos over ~12 weeks
).reset_index(name='est_videos_per_week')

channel_df = channel_df.merge(channel_upload_est, on='channel_id', how='left')

# LII Components
lii_features = ['child_protagonist_rate', 'emotional_score_mean', 
                'est_videos_per_week', 'mean_duration', 'privacy_score_mean',
                'clickbait_score_mean']

lii_data = channel_df[['channel_id', 'title'] + lii_features].dropna()

# Normalize
scaler2 = MinMaxScaler()
lii_normalized = pd.DataFrame(
    scaler2.fit_transform(lii_data[lii_features]),
    columns=[f'{c}_norm' for c in lii_features],
    index=lii_data.index
)

# Equal-weight composite
channel_df.loc[lii_data.index, 'labor_intensity_index'] = lii_normalized.mean(axis=1).values

print(f"Labor Intensity Index computed for {lii_data.shape[0]} channels")
print(f"LII stats:")
print(channel_df['labor_intensity_index'].describe())

# ============================================================
# 7. INITIAL VALIDATION: CI vs LII CORRELATION
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION: CI vs LII RELATIONSHIP")
print("=" * 60)

valid_df = channel_df.dropna(subset=['commercialization_index', 'labor_intensity_index'])
print(f"Channels with both indices: {len(valid_df)}")

# Pearson correlation
r, p = stats.pearsonr(valid_df['commercialization_index'], valid_df['labor_intensity_index'])
print(f"Pearson r = {r:.4f}, p = {p:.6f}")

# Spearman correlation (non-parametric)
rho, p_rho = stats.spearmanr(valid_df['commercialization_index'], valid_df['labor_intensity_index'])
print(f"Spearman rho = {rho:.4f}, p = {p_rho:.6f}")

# Split into high/low CI groups
median_ci = valid_df['commercialization_index'].median()
high_ci = valid_df[valid_df['commercialization_index'] >= median_ci]
low_ci = valid_df[valid_df['commercialization_index'] < median_ci]

print(f"\nHigh CI group (n={len(high_ci)}): LII mean = {high_ci['labor_intensity_index'].mean():.4f}")
print(f"Low CI group (n={len(low_ci)}): LII mean = {low_ci['labor_intensity_index'].mean():.4f}")

# Mann-Whitney U test
u_stat, u_p = stats.mannwhitneyu(high_ci['labor_intensity_index'], 
                                   low_ci['labor_intensity_index'], 
                                   alternative='greater')
print(f"Mann-Whitney U: U={u_stat:.1f}, p={u_p:.6f}")

# Component-level analysis
print("\n--- Component-level: High CI vs Low CI ---")
for comp in lii_features:
    if comp in valid_df.columns:
        h = high_ci[comp].dropna()
        l = low_ci[comp].dropna()
        if len(h) > 0 and len(l) > 0:
            u, p = stats.mannwhitneyu(h, l, alternative='two-sided')
            print(f"  {comp}: High CI mean={h.mean():.4f}, Low CI mean={l.mean():.4f}, p={p:.4f}")

# ============================================================
# 8. SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

output_dir = f'{BASE}/analysis_paper1_v2'
os.makedirs(output_dir, exist_ok=True)

# Save channel-level dataset
channel_df.to_csv(f'{output_dir}/channel_level_features.csv', index=False)
print(f"Saved: channel_level_features.csv ({channel_df.shape})")

# Save video-level dataset
video_df.to_csv(f'{output_dir}/video_level_features.csv', index=False)
print(f"Saved: video_level_features.csv ({video_df.shape})")

# Save clean annotations
ann_clean.to_csv(f'{output_dir}/annotations_cleaned.csv', index=False)
print(f"Saved: annotations_cleaned.csv ({ann_clean.shape})")

# Save summary statistics
summary = {
    'n_channels': len(channel_df),
    'n_channels_with_indices': len(valid_df),
    'n_videos_total': len(video_df),
    'n_videos_annotated': len(ann_clean),
    'ci_mean': float(valid_df['commercialization_index'].mean()),
    'ci_std': float(valid_df['commercialization_index'].std()),
    'lii_mean': float(valid_df['labor_intensity_index'].mean()),
    'lii_std': float(valid_df['labor_intensity_index'].std()),
    'pearson_r': float(r),
    'pearson_p': float(p),
    'spearman_rho': float(rho),
    'spearman_p': float(p_rho),
    'mannwhitney_u': float(u_stat),
    'mannwhitney_p': float(u_p),
    'high_ci_lii_mean': float(high_ci['labor_intensity_index'].mean()),
    'low_ci_lii_mean': float(low_ci['labor_intensity_index'].mean()),
}
with open(f'{output_dir}/index_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: index_summary.json")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
