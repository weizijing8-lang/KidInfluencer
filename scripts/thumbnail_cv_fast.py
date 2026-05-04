"""
Fast Thumbnail CV Analysis using:
1. OpenCV Haar Cascades for face detection
2. DeepFace with opencv backend + batch processing for age/emotion
   (process in batches to avoid memory issues)
"""
import pandas as pd
import numpy as np
import os, json, warnings, glob, sys, time
from collections import Counter
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2

# Load data
print("Loading data...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
clusters = pd.read_csv('analysis_discovery/videos_with_clusters.csv')
df['cluster'] = clusters['cluster'].values

# Get thumbnails
thumb_dir = 'data/thumbnails'
thumb_ids = set(os.path.splitext(f)[0] for f in os.listdir(thumb_dir) if f.endswith('.jpg'))
df_with_thumb = df[df['id'].isin(thumb_ids)].copy()
print(f"Videos with thumbnails: {len(df_with_thumb)}")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Phase 1: Fast face detection with OpenCV only
print("\n=== Phase 1: Face Detection (OpenCV Haar) ===")
face_results = []
start = time.time()

for i, (idx, row) in enumerate(df_with_thumb.iterrows()):
    if (i+1) % 200 == 0:
        elapsed = time.time() - start
        rate = (i+1) / elapsed
        print(f"  {i+1}/{len(df_with_thumb)} ({rate:.0f} img/s)")
    
    video_id = row['id']
    img_path = os.path.join(thumb_dir, f"{video_id}.jpg")
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            face_results.append({'video_id': video_id, 'n_faces': 0, 'face_areas': []})
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_areas = []
        for (x, y, w, h) in faces:
            face_areas.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
        
        face_results.append({
            'video_id': video_id,
            'n_faces': len(faces),
            'face_areas': face_areas,
            'img_h': img.shape[0],
            'img_w': img.shape[1]
        })
    except Exception as e:
        face_results.append({'video_id': video_id, 'n_faces': 0, 'face_areas': []})

elapsed = time.time() - start
print(f"  Phase 1 complete: {len(face_results)} images in {elapsed:.1f}s ({len(face_results)/elapsed:.0f} img/s)")

# Phase 2: DeepFace age + emotion on detected faces only (subsample for speed)
print("\n=== Phase 2: Age + Emotion Analysis (DeepFace on faces only) ===")

# Only process thumbnails that have faces (saves time)
face_df = pd.DataFrame([{'video_id': r['video_id'], 'n_faces': r['n_faces']} for r in face_results])
has_faces = face_df[face_df['n_faces'] > 0]
print(f"  Thumbnails with faces: {len(has_faces)} ({len(has_faces)/len(face_df):.1%})")

# For speed, sample up to 1000 face-containing thumbnails
MAX_DEEPFACE = 1000
if len(has_faces) > MAX_DEEPFACE:
    sample_ids = has_faces.sample(n=MAX_DEEPFACE, random_state=42)['video_id'].tolist()
else:
    sample_ids = has_faces['video_id'].tolist()

print(f"  Processing {len(sample_ids)} thumbnails with DeepFace...")

# Import DeepFace
from deepface import DeepFace

deepface_results = []
errors = 0
start2 = time.time()

for i, video_id in enumerate(sample_ids):
    if (i+1) % 50 == 0:
        elapsed2 = time.time() - start2
        rate = (i+1) / elapsed2
        eta = (len(sample_ids) - i - 1) / rate
        print(f"  {i+1}/{len(sample_ids)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")
    
    img_path = os.path.join(thumb_dir, f"{video_id}.jpg")
    
    try:
        analyses = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        
        if not isinstance(analyses, list):
            analyses = [analyses]
        
        ages = []
        emotions_list = []
        dominant_emotions = []
        
        for face in analyses:
            age = face.get('age', None)
            if age is not None:
                ages.append(age)
            emotion = face.get('dominant_emotion', None)
            if emotion:
                dominant_emotions.append(emotion)
            emotion_scores = face.get('emotion', {})
            emotions_list.append(emotion_scores)
        
        # Aggregate
        avg_emotions = {}
        if emotions_list:
            for emo_key in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                vals = [e.get(emo_key, 0) for e in emotions_list if e]
                avg_emotions[emo_key] = float(np.mean(vals)) if vals else 0.0
        
        has_child = any(a < 18 for a in ages) if ages else False
        manip_score = avg_emotions.get('surprise', 0) + avg_emotions.get('fear', 0) + avg_emotions.get('sad', 0)
        
        deepface_results.append({
            'video_id': video_id,
            'n_faces_df': len(analyses),
            'has_child': has_child,
            'min_age': min(ages) if ages else None,
            'mean_age': float(np.mean(ages)) if ages else None,
            'dominant_emotion': dominant_emotions[0] if dominant_emotions else None,
            'emotion_surprise': avg_emotions.get('surprise', 0),
            'emotion_fear': avg_emotions.get('fear', 0),
            'emotion_sad': avg_emotions.get('sad', 0),
            'emotion_happy': avg_emotions.get('happy', 0),
            'emotion_angry': avg_emotions.get('angry', 0),
            'emotion_neutral': avg_emotions.get('neutral', 0),
            'manip_emotion_score': manip_score,
        })
    except Exception as e:
        errors += 1
        deepface_results.append({
            'video_id': video_id,
            'n_faces_df': 0,
            'has_child': False,
            'min_age': None,
            'mean_age': None,
            'dominant_emotion': None,
            'emotion_surprise': 0,
            'emotion_fear': 0,
            'emotion_sad': 0,
            'emotion_happy': 0,
            'emotion_angry': 0,
            'emotion_neutral': 0,
            'manip_emotion_score': 0,
        })

elapsed2 = time.time() - start2
print(f"  Phase 2 complete: {len(deepface_results)} in {elapsed2:.1f}s, {errors} errors")

# ============ MERGE AND SAVE ============
print("\n=== Merging results ===")

# Create face detection dataframe
face_det_df = pd.DataFrame([{
    'video_id': r['video_id'],
    'n_faces_haar': r['n_faces'],
    'has_face': r['n_faces'] > 0,
    'face_area_ratio': sum(f['w']*f['h'] for f in r['face_areas']) / (r.get('img_h', 480) * r.get('img_w', 640)) if r['face_areas'] else 0
} for r in face_results])

# Create DeepFace dataframe
deepface_df = pd.DataFrame(deepface_results)

# Merge with video metadata
merged = df_with_thumb[['id', 'title', 'channel_short_name', 'viewCount', 'cluster']].copy()
merged = merged.rename(columns={'id': 'video_id'})
merged = merged.merge(face_det_df, on='video_id', how='left')
merged = merged.merge(deepface_df, on='video_id', how='left')

# Save
output_path = 'analysis_discovery/thumbnail_cv_results.csv'
merged.to_csv(output_path, index=False)
print(f"Saved: {output_path} ({len(merged)} rows)")

# ============ ANALYSIS ============
print(f"\n{'='*60}")
print("THUMBNAIL CV ANALYSIS RESULTS")
print(f"{'='*60}")

print(f"\n--- Face Detection (Haar Cascade, all {len(merged)} thumbnails) ---")
print(f"  Thumbnails with faces: {merged['has_face'].sum()} ({merged['has_face'].mean():.1%})")
print(f"  Mean faces per thumbnail: {merged['n_faces_haar'].mean():.2f}")

# DeepFace subset
df_deep = merged[merged['dominant_emotion'].notna()]
print(f"\n--- Age & Emotion (DeepFace, {len(df_deep)} thumbnails) ---")
print(f"  Thumbnails with children (age<18): {df_deep['has_child'].sum()} ({df_deep['has_child'].mean():.1%})")
print(f"  Mean estimated age: {df_deep['mean_age'].mean():.1f}")
print(f"\n  Dominant emotion distribution:")
print(df_deep['dominant_emotion'].value_counts().to_string())

print(f"\n  Manipulation emotion score (surprise+fear+sad):")
print(f"    Mean: {df_deep['manip_emotion_score'].mean():.2f}")
print(f"    Median: {df_deep['manip_emotion_score'].median():.2f}")

# Per-cluster analysis
print(f"\n{'='*60}")
print("PER-CLUSTER THUMBNAIL ANALYSIS")
print(f"{'='*60}")

cluster_labels = json.load(open('analysis_discovery/cluster_labels_llm.json'))
label_map = {c['cluster_id']: c['category_name'] for c in cluster_labels}
cluster_info = json.load(open('analysis_discovery/cluster_info.json'))
boost_map = {c['cluster']: c['view_boost'] for c in cluster_info}

print(f"\n{'Cl':>3} {'Category':<30} {'Face%':>6} {'Child%':>7} {'ManipEmo':>9} {'DomEmo':<10} {'Boost':>8}")
print("-" * 80)

cluster_summary = []
for k in range(15):
    cl_data = merged[merged['cluster'] == k]
    cl_deep = df_deep[df_deep['cluster'] == k]
    
    face_pct = cl_data['has_face'].mean() if len(cl_data) > 0 else 0
    child_pct = cl_deep['has_child'].mean() if len(cl_deep) > 0 else 0
    manip_emo = cl_deep['manip_emotion_score'].mean() if len(cl_deep) > 0 else 0
    dom_emo = cl_deep['dominant_emotion'].mode().iloc[0] if len(cl_deep) > 0 and len(cl_deep['dominant_emotion'].dropna()) > 0 else 'N/A'
    boost = boost_map.get(k, 0)
    
    row_data = {
        'cluster': k,
        'category': label_map.get(k, f'C{k}'),
        'face_pct': face_pct,
        'child_pct': child_pct,
        'manip_emotion': manip_emo,
        'dominant_emotion': dom_emo,
        'view_boost': boost
    }
    cluster_summary.append(row_data)
    
    print(f"{k:>3} {label_map.get(k, 'Unknown'):<30} {face_pct:>5.1%} {child_pct:>6.1%} {manip_emo:>9.1f} {dom_emo:<10} {boost*100:>+7.0f}%")

# Save cluster summary
pd.DataFrame(cluster_summary).to_csv('analysis_discovery/thumbnail_cluster_summary.csv', index=False)

# Statistical tests
print(f"\n{'='*60}")
print("STATISTICAL TESTS")
print(f"{'='*60}")

from scipy import stats

# Test: Do high-view clusters have more emotional thumbnails?
valid = df_deep[df_deep['viewCount'] > 0].copy()
valid['log_views'] = np.log10(valid['viewCount'])

r, p = stats.pearsonr(valid['manip_emotion_score'], valid['log_views'])
print(f"\n  Pearson r (manip_emotion vs log_views): r={r:.4f}, p={p:.6f}")

# Spearman
rs, ps = stats.spearmanr(valid['manip_emotion_score'], valid['log_views'])
print(f"  Spearman rho (manip_emotion vs log_views): rho={rs:.4f}, p={ps:.6f}")

# High vs low manip emotion
median_manip = valid['manip_emotion_score'].median()
high = valid[valid['manip_emotion_score'] > median_manip]['viewCount']
low = valid[valid['manip_emotion_score'] <= median_manip]['viewCount']
u, pu = stats.mannwhitneyu(high, low, alternative='greater')
print(f"\n  High manip emotion median views: {high.median():,.0f}")
print(f"  Low manip emotion median views: {low.median():,.0f}")
print(f"  Mann-Whitney U: U={u:.0f}, p={pu:.6f}")

# Child presence
if valid['has_child'].sum() > 10:
    child_v = valid[valid['has_child']]['viewCount']
    nochild_v = valid[~valid['has_child']]['viewCount']
    u2, p2 = stats.mannwhitneyu(child_v, nochild_v, alternative='two-sided')
    print(f"\n  Child present median views: {child_v.median():,.0f}")
    print(f"  No child median views: {nochild_v.median():,.0f}")
    print(f"  Mann-Whitney U: U={u2:.0f}, p={p2:.6f}")

# Face presence
face_v = merged[merged['has_face']]['viewCount']
noface_v = merged[~merged['has_face']]['viewCount']
u3, p3 = stats.mannwhitneyu(face_v, noface_v, alternative='two-sided')
print(f"\n  Face present median views: {face_v.median():,.0f}")
print(f"  No face median views: {noface_v.median():,.0f}")
print(f"  Mann-Whitney U: U={u3:.0f}, p={p3:.6f}")

print("\n=== DONE ===")
