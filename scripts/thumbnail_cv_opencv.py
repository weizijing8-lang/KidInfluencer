"""
Thumbnail CV Analysis - OpenCV approach:
1. Haar cascade face detection
2. Face size analysis (large face = close-up, common in clickbait)
3. Smile detection as proxy for emotional expression
4. Color/saturation analysis (bright/saturated = more designed)
5. Text density (edge detection proxy)
"""
import pandas as pd
import numpy as np
import os, json, warnings, glob, time
from collections import Counter
warnings.filterwarnings('ignore')
import cv2

# Load data
print("Loading data...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
clusters = pd.read_csv('analysis_discovery/videos_with_clusters.csv')
df['cluster'] = clusters['cluster'].values

# Get thumbnails that match our kidfluencer dataset
thumb_dir = 'data/thumbnails'
all_thumbs = set(os.path.splitext(f)[0] for f in os.listdir(thumb_dir) if f.endswith('.jpg'))
kid_ids = set(df['id'].values)
matching_ids = kid_ids & all_thumbs
df_with_thumb = df[df['id'].isin(matching_ids)].copy()
print(f"Videos with thumbnails: {len(df_with_thumb)}")
print(f"Cluster coverage: {df_with_thumb['cluster'].nunique()}/15")

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

if face_cascade.empty():
    print("ERROR: Face cascade failed to load!")
    exit(1)

print(f"\nAnalyzing {len(df_with_thumb)} thumbnails...")
results = []
start = time.time()

for i, (idx, row) in enumerate(df_with_thumb.iterrows()):
    if (i+1) % 300 == 0:
        elapsed = time.time() - start
        rate = (i+1) / elapsed
        eta = (len(df_with_thumb) - i - 1) / rate
        print(f"  {i+1}/{len(df_with_thumb)} ({rate:.0f} img/s, ETA: {eta:.0f}s)")
    
    video_id = row['id']
    img_path = os.path.join(thumb_dir, f"{video_id}.jpg")
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            results.append({
                'video_id': video_id, 'n_faces': 0, 'face_coverage': 0,
                'max_face_ratio': 0, 'has_large_face': False, 'has_smile': False,
                'n_smiles': 0, 'mean_saturation': 0, 'mean_brightness': 0,
                'text_density': 0, 'color_std': 0
            })
            continue
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        n_faces = len(faces)
        
        # Face metrics
        total_face_area = 0
        max_face_size = 0
        has_large_face = False
        has_smile = False
        n_smiles = 0
        
        for (fx, fy, fw, fh) in faces:
            face_area = fw * fh
            total_face_area += face_area
            max_face_size = max(max_face_size, face_area)
            
            # Large face = close-up shot
            if face_area > (h * w * 0.08):
                has_large_face = True
            
            # Smile detection in face region
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            smiles = smile_cascade.detectMultiScale(
                face_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
            )
            if len(smiles) > 0:
                has_smile = True
                n_smiles += 1
        
        face_coverage = total_face_area / (h * w) if (h * w) > 0 else 0
        max_face_ratio = max_face_size / (h * w) if (h * w) > 0 else 0
        
        # Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = float(hsv[:,:,1].mean())
        mean_brightness = float(hsv[:,:,2].mean())
        
        # Text/edge density
        edges = cv2.Canny(gray, 100, 200)
        text_density = float(edges.mean() / 255.0)
        
        # Color diversity
        color_std = float(np.std(img, axis=(0,1)).mean())
        
        results.append({
            'video_id': video_id,
            'n_faces': n_faces,
            'face_coverage': face_coverage,
            'max_face_ratio': max_face_ratio,
            'has_large_face': has_large_face,
            'has_smile': has_smile,
            'n_smiles': n_smiles,
            'mean_saturation': mean_saturation,
            'mean_brightness': mean_brightness,
            'text_density': text_density,
            'color_std': color_std,
        })
        
    except Exception as e:
        results.append({
            'video_id': video_id, 'n_faces': 0, 'face_coverage': 0,
            'max_face_ratio': 0, 'has_large_face': False, 'has_smile': False,
            'n_smiles': 0, 'mean_saturation': 0, 'mean_brightness': 0,
            'text_density': 0, 'color_std': 0
        })

elapsed = time.time() - start
print(f"\nComplete: {len(results)} images in {elapsed:.1f}s ({len(results)/elapsed:.0f} img/s)")

# Create results dataframe
results_df = pd.DataFrame(results)

# Merge with video metadata
merged = df_with_thumb[['id', 'title', 'channel_short_name', 'viewCount', 'cluster']].copy()
merged = merged.rename(columns={'id': 'video_id'})
merged = merged.merge(results_df, on='video_id', how='left')

# Save
output_path = 'analysis_discovery/thumbnail_cv_results.csv'
merged.to_csv(output_path, index=False)
print(f"Saved: {output_path}")

# ============ ANALYSIS ============
print(f"\n{'='*60}")
print("THUMBNAIL CV ANALYSIS RESULTS")
print(f"{'='*60}")

print(f"\nOverall Statistics (n={len(merged)}):")
print(f"  Thumbnails with faces: {(merged['n_faces'] > 0).sum()} ({(merged['n_faces'] > 0).mean():.1%})")
print(f"  Mean faces per thumbnail: {merged['n_faces'].mean():.2f}")
print(f"  Thumbnails with large face (close-up): {merged['has_large_face'].sum()} ({merged['has_large_face'].mean():.1%})")
print(f"  Thumbnails with smile: {merged['has_smile'].sum()} ({merged['has_smile'].mean():.1%})")
print(f"  Mean saturation: {merged['mean_saturation'].mean():.1f}")
print(f"  Mean text density: {merged['text_density'].mean():.4f}")

# Per-cluster
print(f"\n{'='*60}")
print("PER-CLUSTER ANALYSIS")
print(f"{'='*60}")

cluster_labels = json.load(open('analysis_discovery/cluster_labels_llm.json'))
label_map = {c['cluster_id']: c['category_name'] for c in cluster_labels}
cluster_info = json.load(open('analysis_discovery/cluster_info.json'))
boost_map = {c['cluster']: c['view_boost'] for c in cluster_info}

print(f"\n{'Cl':>3} {'Category':<28} {'N':>5} {'Face%':>6} {'LgFace%':>8} {'Smile%':>7} {'Satur':>6} {'TxtDen':>7} {'Boost':>8}")
print("-" * 90)

cluster_summary = []
for k in range(15):
    cl = merged[merged['cluster'] == k]
    if len(cl) == 0:
        continue
    
    face_pct = (cl['n_faces'] > 0).mean()
    lg_face_pct = cl['has_large_face'].mean()
    smile_pct = cl['has_smile'].mean()
    satur = cl['mean_saturation'].mean()
    txt_den = cl['text_density'].mean()
    boost = boost_map.get(k, 0)
    
    row_data = {
        'cluster': k,
        'category': label_map.get(k, f'C{k}'),
        'n_thumbnails': len(cl),
        'face_pct': face_pct,
        'large_face_pct': lg_face_pct,
        'smile_pct': smile_pct,
        'mean_saturation': satur,
        'text_density': txt_den,
        'view_boost': boost
    }
    cluster_summary.append(row_data)
    
    print(f"{k:>3} {label_map.get(k, 'Unknown'):<28} {len(cl):>5} {face_pct:>5.1%} {lg_face_pct:>7.1%} {smile_pct:>6.1%} {satur:>6.1f} {txt_den:>7.4f} {boost*100:>+7.0f}%")

# Save cluster summary
summary_df = pd.DataFrame(cluster_summary)
summary_df.to_csv('analysis_discovery/thumbnail_cluster_summary.csv', index=False)

# ============ STATISTICAL TESTS ============
print(f"\n{'='*60}")
print("STATISTICAL TESTS")
print(f"{'='*60}")
from scipy import stats

valid = merged[(merged['viewCount'] > 0)].copy()
valid['log_views'] = np.log10(valid['viewCount'])
valid_faces = valid[valid['n_faces'] > 0]

# Face presence vs views
print("\n--- Face Presence vs Views ---")
face_views = valid[valid['n_faces'] > 0]['viewCount']
noface_views = valid[valid['n_faces'] == 0]['viewCount']
if len(face_views) > 5 and len(noface_views) > 5:
    u, p = stats.mannwhitneyu(face_views, noface_views, alternative='two-sided')
    print(f"  Face present: median={face_views.median():,.0f} (n={len(face_views)})")
    print(f"  No face: median={noface_views.median():,.0f} (n={len(noface_views)})")
    print(f"  Mann-Whitney U={u:.0f}, p={p:.6f}")

# Large face (close-up) vs views
print("\n--- Large Face (close-up) vs Views ---")
lg_views = valid[valid['has_large_face'] == True]['viewCount']
no_lg_views = valid[valid['has_large_face'] == False]['viewCount']
if len(lg_views) > 5:
    u2, p2 = stats.mannwhitneyu(lg_views, no_lg_views, alternative='greater')
    print(f"  Large face: median={lg_views.median():,.0f} (n={len(lg_views)})")
    print(f"  No large face: median={no_lg_views.median():,.0f} (n={len(no_lg_views)})")
    print(f"  Mann-Whitney U={u2:.0f}, p={p2:.6f}")

# Smile vs views
print("\n--- Smile vs Views ---")
smile_views = valid[valid['has_smile'] == True]['viewCount']
no_smile_views = valid[valid['has_smile'] == False]['viewCount']
if len(smile_views) > 5:
    u3, p3 = stats.mannwhitneyu(smile_views, no_smile_views, alternative='two-sided')
    print(f"  Smile: median={smile_views.median():,.0f} (n={len(smile_views)})")
    print(f"  No smile: median={no_smile_views.median():,.0f} (n={len(no_smile_views)})")
    print(f"  Mann-Whitney U={u3:.0f}, p={p3:.6f}")

# Correlations
print("\n--- Correlations with log(Views) ---")
for col in ['face_coverage', 'max_face_ratio', 'mean_saturation', 'text_density', 'color_std', 'n_faces']:
    r, p = stats.spearmanr(valid[col], valid['log_views'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {col:<20}: rho={r:+.4f}, p={p:.6f} {sig}")

# Cluster-level correlations
print(f"\n--- Cluster-level Correlations (n=15 clusters) ---")
cs = summary_df.copy()
for col in ['face_pct', 'large_face_pct', 'smile_pct', 'mean_saturation', 'text_density']:
    r, p = stats.spearmanr(cs[col], cs['view_boost'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {col:<20} vs view_boost: rho={r:+.4f}, p={p:.4f} {sig}")

print("\n=== DONE ===")
