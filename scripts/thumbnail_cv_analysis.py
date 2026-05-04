"""
Thumbnail Computer Vision Analysis:
1. Face detection (are faces present?)
2. Age estimation (are children present? age < 18)
3. Emotion recognition (dominant emotion, focus on surprise/fear/sadness)

Uses DeepFace with OpenCV backend for speed.
"""
import pandas as pd
import numpy as np
import os, json, warnings, glob, sys
from collections import Counter
warnings.filterwarnings('ignore')

# Import CV libraries
from deepface import DeepFace
import cv2

# Load data
print("Loading data...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
clusters = pd.read_csv('analysis_discovery/videos_with_clusters.csv')
df['cluster'] = clusters['cluster'].values

# Get thumbnails that match our dataset
thumb_dir = 'data/thumbnails'
thumb_files = glob.glob(f'{thumb_dir}/*.jpg')
thumb_ids = set(os.path.splitext(os.path.basename(f))[0] for f in thumb_files)

df_with_thumb = df[df['id'].isin(thumb_ids)].copy()
print(f"Videos with thumbnails: {len(df_with_thumb)}")
print(f"Cluster coverage: {df_with_thumb['cluster'].nunique()}/15")

# Process thumbnails
results = []
total = len(df_with_thumb)
errors = 0

print(f"\nAnalyzing {total} thumbnails...")
for i, (idx, row) in enumerate(df_with_thumb.iterrows()):
    if (i+1) % 100 == 0:
        print(f"  Progress: {i+1}/{total} (errors: {errors})")
    
    video_id = row['id']
    img_path = os.path.join(thumb_dir, f"{video_id}.jpg")
    
    if not os.path.exists(img_path):
        continue
    
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            errors += 1
            continue
        
        # Analyze with DeepFace
        analyses = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        
        # Handle single or multiple faces
        if not isinstance(analyses, list):
            analyses = [analyses]
        
        n_faces = len(analyses)
        has_child = False
        ages = []
        emotions = []
        dominant_emotions = []
        
        for face in analyses:
            age = face.get('age', None)
            if age is not None:
                ages.append(age)
                if age < 18:
                    has_child = True
            
            emotion = face.get('dominant_emotion', None)
            if emotion:
                dominant_emotions.append(emotion)
            
            # Get emotion scores
            emotion_scores = face.get('emotion', {})
            emotions.append(emotion_scores)
        
        # Aggregate emotion scores across all faces
        avg_emotions = {}
        if emotions:
            for emo_key in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                vals = [e.get(emo_key, 0) for e in emotions if e]
                avg_emotions[emo_key] = np.mean(vals) if vals else 0
        
        # Compute manipulation-relevant emotion score
        # High surprise + fear + sad = potential emotional manipulation
        manip_emotion_score = avg_emotions.get('surprise', 0) + avg_emotions.get('fear', 0) + avg_emotions.get('sad', 0)
        
        result = {
            'video_id': video_id,
            'cluster': row['cluster'],
            'channel': row['channel_short_name'],
            'viewCount': row['viewCount'],
            'n_faces': n_faces,
            'has_child': has_child,
            'min_age': min(ages) if ages else None,
            'max_age': max(ages) if ages else None,
            'mean_age': np.mean(ages) if ages else None,
            'dominant_emotion': dominant_emotions[0] if dominant_emotions else None,
            'all_emotions': ','.join(dominant_emotions),
            'emotion_surprise': avg_emotions.get('surprise', 0),
            'emotion_fear': avg_emotions.get('fear', 0),
            'emotion_sad': avg_emotions.get('sad', 0),
            'emotion_happy': avg_emotions.get('happy', 0),
            'emotion_angry': avg_emotions.get('angry', 0),
            'emotion_neutral': avg_emotions.get('neutral', 0),
            'manip_emotion_score': manip_emotion_score,
        }
        results.append(result)
        
    except Exception as e:
        errors += 1
        results.append({
            'video_id': video_id,
            'cluster': row['cluster'],
            'channel': row['channel_short_name'],
            'viewCount': row['viewCount'],
            'n_faces': 0,
            'has_child': False,
            'min_age': None,
            'max_age': None,
            'mean_age': None,
            'dominant_emotion': None,
            'all_emotions': '',
            'emotion_surprise': 0,
            'emotion_fear': 0,
            'emotion_sad': 0,
            'emotion_happy': 0,
            'emotion_angry': 0,
            'emotion_neutral': 0,
            'manip_emotion_score': 0,
        })

print(f"\nCompleted: {len(results)} analyzed, {errors} errors")

# Save results
results_df = pd.DataFrame(results)
output_path = 'analysis_discovery/thumbnail_cv_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# ============ SUMMARY STATISTICS ============
print(f"\n{'='*60}")
print("THUMBNAIL CV ANALYSIS SUMMARY")
print(f"{'='*60}")

print(f"\nFace Detection:")
print(f"  Thumbnails with faces: {(results_df['n_faces'] > 0).sum()} ({(results_df['n_faces'] > 0).mean():.1%})")
print(f"  Mean faces per thumbnail: {results_df['n_faces'].mean():.2f}")

print(f"\nChild Detection (age < 18):")
print(f"  Thumbnails with children: {results_df['has_child'].sum()} ({results_df['has_child'].mean():.1%})")

print(f"\nDominant Emotions:")
print(results_df['dominant_emotion'].value_counts())

print(f"\nManipulation Emotion Score (surprise + fear + sad):")
print(f"  Mean: {results_df['manip_emotion_score'].mean():.2f}")
print(f"  Median: {results_df['manip_emotion_score'].median():.2f}")

# ============ PER-CLUSTER ANALYSIS ============
print(f"\n{'='*60}")
print("PER-CLUSTER THUMBNAIL ANALYSIS")
print(f"{'='*60}")
print(f"\n{'Cluster':>8} {'N':>5} {'Faces%':>7} {'Child%':>7} {'ManipEmo':>9} {'DomEmo':<12} {'ViewBoost':>10}")
print("-" * 70)

cluster_labels = json.load(open('analysis_discovery/cluster_labels_llm.json'))
label_map = {c['cluster_id']: c['category_name'] for c in cluster_labels}

for k in range(15):
    cluster_data = results_df[results_df['cluster'] == k]
    if len(cluster_data) == 0:
        continue
    n = len(cluster_data)
    face_pct = (cluster_data['n_faces'] > 0).mean()
    child_pct = cluster_data['has_child'].mean()
    manip_emo = cluster_data['manip_emotion_score'].mean()
    dom_emo = cluster_data['dominant_emotion'].mode().iloc[0] if len(cluster_data['dominant_emotion'].dropna()) > 0 else 'N/A'
    
    # Get view boost from cluster info
    boost = [c for c in json.load(open('analysis_discovery/cluster_info.json')) if c['cluster'] == k][0]['view_boost']
    
    print(f"{k:>8} {n:>5} {face_pct:>6.1%} {child_pct:>6.1%} {manip_emo:>9.1f} {dom_emo:<12} {boost*100:>+9.0f}%")

print(f"\n{'='*60}")
print("KEY FINDING: Correlation between manipulation emotions and views")
print(f"{'='*60}")

# Correlation: manip_emotion_score vs log(views)
from scipy import stats
valid = results_df[(results_df['n_faces'] > 0) & (results_df['viewCount'] > 0)]
log_views = np.log10(valid['viewCount'])
r, p = stats.pearsonr(valid['manip_emotion_score'], log_views)
print(f"  Pearson r (manip_emotion vs log_views): {r:.4f}, p={p:.4f}")

# Compare high vs low manip emotion
median_manip = valid['manip_emotion_score'].median()
high_manip = valid[valid['manip_emotion_score'] > median_manip]['viewCount']
low_manip = valid[valid['manip_emotion_score'] <= median_manip]['viewCount']
t, p2 = stats.mannwhitneyu(high_manip, low_manip, alternative='greater')
print(f"  High vs Low manip emotion median views: {high_manip.median():,.0f} vs {low_manip.median():,.0f}")
print(f"  Mann-Whitney U test: U={t:.0f}, p={p2:.4f}")

# Child presence and views
child_views = valid[valid['has_child']]['viewCount']
no_child_views = valid[~valid['has_child']]['viewCount']
if len(child_views) > 0 and len(no_child_views) > 0:
    t2, p3 = stats.mannwhitneyu(child_views, no_child_views, alternative='greater')
    print(f"\n  Child present vs absent median views: {child_views.median():,.0f} vs {no_child_views.median():,.0f}")
    print(f"  Mann-Whitney U test: U={t2:.0f}, p={p3:.4f}")
