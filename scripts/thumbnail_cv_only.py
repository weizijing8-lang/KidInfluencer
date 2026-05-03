"""
Thumbnail CV Analysis (fast version, no OCR)
- Face detection (Haar cascade)
- Open mouth / exaggerated expression detection
- Visual features: brightness, saturation, contrast, colorfulness, edge density
- Text overlay detection via edge density heuristic
"""

import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path

def analyze_thumbnail(filepath):
    """Analyze a single thumbnail using OpenCV only."""
    result = {
        'num_faces': 0,
        'max_face_ratio': 0.0,
        'has_open_mouth': False,
        'brightness': 0.0,
        'saturation': 0.0,
        'contrast': 0.0,
        'colorfulness': 0.0,
        'edge_density': 0.0,
        'top_edge_density': 0.0,
        'bottom_edge_density': 0.0,
        'has_text_overlay': False,
    }
    
    if filepath is None or not os.path.exists(filepath):
        return result
    
    try:
        img = cv2.imread(filepath)
        if img is None:
            return result
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- Face Detection ---
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        result['num_faces'] = len(faces)
        
        if len(faces) > 0:
            # Max face ratio (largest face area / image area)
            face_areas = [(fw * fh) / (w * h) for (x, y, fw, fh) in faces]
            result['max_face_ratio'] = max(face_areas)
            
            # Open mouth detection (proxy for exaggerated expression)
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            for (x, y, fw, fh) in faces:
                face_roi = gray[y:y+fh, x:x+fw]
                lower_face = face_roi[fh//2:, :]
                if lower_face.shape[0] > 0 and lower_face.shape[1] > 0:
                    mouths = mouth_cascade.detectMultiScale(lower_face, scaleFactor=1.5, minNeighbors=15)
                    if len(mouths) > 0:
                        result['has_open_mouth'] = True
                        break
        
        # --- Visual Features ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result['brightness'] = float(np.mean(hsv[:, :, 2]))
        result['saturation'] = float(np.mean(hsv[:, :, 1]))
        result['contrast'] = float(np.std(gray))
        
        # Colorfulness
        B, G, R = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        result['colorfulness'] = float(np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.3 * np.sqrt(np.std(rg)**2 + np.std(yb)**2))
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        result['edge_density'] = float(np.sum(edges > 0) / (h * w))
        
        # Text overlay heuristic (high edge density in top/bottom regions)
        top_region = edges[:int(h*0.2), :]
        bottom_region = edges[int(h*0.8):, :]
        result['top_edge_density'] = float(np.sum(top_region > 0) / (top_region.shape[0] * w)) if top_region.shape[0] > 0 else 0
        result['bottom_edge_density'] = float(np.sum(bottom_region > 0) / (bottom_region.shape[0] * w)) if bottom_region.shape[0] > 0 else 0
        result['has_text_overlay'] = bool(result['top_edge_density'] > 0.15 or result['bottom_edge_density'] > 0.15)
        
    except Exception as e:
        pass
    
    return result


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("THUMBNAIL CV ANALYSIS (FAST VERSION)", flush=True)
    print("=" * 60, flush=True)
    
    df = pd.read_csv('/home/ubuntu/KidInfluencer/data/combined_videos.csv')
    video_ids = df['video_id'].dropna().unique().tolist()
    thumb_dir = '/home/ubuntu/KidInfluencer/data/thumbnails'
    output_dir = '/home/ubuntu/KidInfluencer/analysis_v3/thumbnails'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(video_ids)} thumbnails...", flush=True)
    
    results = []
    for i, vid in enumerate(video_ids):
        filepath = os.path.join(thumb_dir, f"{vid}.jpg")
        analysis = analyze_thumbnail(filepath)
        analysis['video_id'] = vid
        results.append(analysis)
        
        if (i + 1) % 500 == 0:
            print(f"  Analyzed {i+1}/{len(video_ids)}...", flush=True)
    
    cv_df = pd.DataFrame(results)
    
    # Stats
    print(f"\n--- RESULTS ---", flush=True)
    print(f"Thumbnails with faces: {(cv_df['num_faces'] > 0).sum()} ({(cv_df['num_faces'] > 0).mean()*100:.1f}%)", flush=True)
    print(f"Avg faces per thumbnail: {cv_df['num_faces'].mean():.2f}", flush=True)
    print(f"Avg max face ratio: {cv_df[cv_df['num_faces']>0]['max_face_ratio'].mean():.3f}", flush=True)
    print(f"Open mouth (exaggerated): {cv_df['has_open_mouth'].sum()} ({cv_df['has_open_mouth'].mean()*100:.1f}%)", flush=True)
    print(f"Text overlay detected: {cv_df['has_text_overlay'].sum()} ({cv_df['has_text_overlay'].mean()*100:.1f}%)", flush=True)
    print(f"Avg brightness: {cv_df['brightness'].mean():.1f}", flush=True)
    print(f"Avg saturation: {cv_df['saturation'].mean():.1f}", flush=True)
    print(f"Avg colorfulness: {cv_df['colorfulness'].mean():.1f}", flush=True)
    print(f"Avg edge density: {cv_df['edge_density'].mean():.4f}", flush=True)
    
    # Save
    cv_df.to_csv(f'{output_dir}/thumbnail_cv_features.csv', index=False)
    
    # Merge with video data
    merged = df.merge(cv_df, on='video_id', how='left')
    merged.to_csv(f'{output_dir}/videos_with_thumbnail_features.csv', index=False)
    
    print(f"\nSaved to {output_dir}/thumbnail_cv_features.csv", flush=True)
    print("DONE.", flush=True)
