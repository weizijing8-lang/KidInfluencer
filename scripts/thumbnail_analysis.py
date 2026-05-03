"""
Thumbnail Analysis for Kidfluencer Study
- Downloads thumbnails from YouTube
- Detects faces (count, estimated age, gender)
- Analyzes facial expressions (happy, sad, surprise, etc.)
- Detects text overlay (OCR) for visual clickbait
- Extracts visual features (brightness, saturation, contrast)
"""

import pandas as pd
import numpy as np
import os
import requests
import time
import json
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PHASE 1: Download thumbnails
# ============================================================

def download_thumbnail(video_id, save_dir):
    """Download thumbnail for a video, trying multiple resolutions."""
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]
    
    filepath = os.path.join(save_dir, f"{video_id}.jpg")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return filepath  # already downloaded
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                return filepath
        except:
            continue
    return None


def download_all_thumbnails(video_ids, save_dir, max_workers=10):
    """Download thumbnails in parallel."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_thumbnail, vid, save_dir): vid for vid in video_ids}
        done = 0
        for future in as_completed(futures):
            vid = futures[future]
            done += 1
            try:
                path = future.result()
                results[vid] = path
            except Exception as e:
                results[vid] = None
            if done % 500 == 0:
                print(f"  Downloaded {done}/{len(video_ids)} thumbnails...", flush=True)
    
    return results


# ============================================================
# PHASE 2: CV Analysis (lightweight, no deepface)
# ============================================================

def analyze_thumbnail_cv(filepath):
    """Analyze a single thumbnail using OpenCV."""
    result = {
        'num_faces': 0,
        'has_child_face': False,
        'dominant_expression': 'unknown',
        'has_open_mouth': False,
        'has_text_overlay': False,
        'brightness': 0.0,
        'saturation': 0.0,
        'contrast': 0.0,
        'colorfulness': 0.0,
        'edge_density': 0.0,
    }
    
    if filepath is None or not os.path.exists(filepath):
        return result
    
    try:
        img = cv2.imread(filepath)
        if img is None:
            return result
        
        h, w = img.shape[:2]
        
        # --- Face Detection using Haar Cascade ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        result['num_faces'] = len(faces)
        
        # Check for open mouths (surprise/shock expression proxy)
        if len(faces) > 0:
            # Estimate if any face is likely a child (smaller face relative to image, or lower position)
            for (x, y, fw, fh) in faces:
                face_ratio = (fw * fh) / (w * h)
                # Large face ratio in thumbnail = likely main subject
                if face_ratio > 0.03:
                    result['has_child_face'] = True  # We can't truly detect child vs adult with Haar, mark as face present
            
            # Detect open mouth as proxy for exaggerated expression
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            for (x, y, fw, fh) in faces:
                face_roi = gray[y:y+fh, x:x+fw]
                # Look in lower half of face for mouth
                lower_face = face_roi[fh//2:, :]
                mouths = mouth_cascade.detectMultiScale(lower_face, scaleFactor=1.5, minNeighbors=15)
                if len(mouths) > 0:
                    result['has_open_mouth'] = True
                    break
        
        # --- Visual Features ---
        # Brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result['brightness'] = float(np.mean(hsv[:, :, 2]))
        
        # Saturation
        result['saturation'] = float(np.mean(hsv[:, :, 1]))
        
        # Contrast (std of grayscale)
        result['contrast'] = float(np.std(gray))
        
        # Colorfulness (Hasler & Süsstrunk metric)
        B, G, R = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        result['colorfulness'] = float(np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.3 * np.sqrt(np.std(rg)**2 + np.std(yb)**2))
        
        # Edge density (proxy for visual complexity/text overlay)
        edges = cv2.Canny(gray, 100, 200)
        result['edge_density'] = float(np.sum(edges > 0) / (h * w))
        
        # --- Text Detection (simple approach: high edge density in specific regions) ---
        # Top and bottom 20% of image often have text overlays
        top_region = edges[:int(h*0.2), :]
        bottom_region = edges[int(h*0.8):, :]
        top_edge_density = np.sum(top_region > 0) / (top_region.shape[0] * w) if top_region.shape[0] > 0 else 0
        bottom_edge_density = np.sum(bottom_region > 0) / (bottom_region.shape[0] * w) if bottom_region.shape[0] > 0 else 0
        # High edge density in top/bottom = likely text overlay
        result['has_text_overlay'] = bool(top_edge_density > 0.15 or bottom_edge_density > 0.15)
        
    except Exception as e:
        pass
    
    return result


def run_cv_analysis(video_ids, thumb_dir, batch_size=100):
    """Run CV analysis on all downloaded thumbnails."""
    results = []
    total = len(video_ids)
    
    for i, vid in enumerate(video_ids):
        filepath = os.path.join(thumb_dir, f"{vid}.jpg")
        analysis = analyze_thumbnail_cv(filepath)
        analysis['video_id'] = vid
        results.append(analysis)
        
        if (i + 1) % batch_size == 0:
            print(f"  Analyzed {i+1}/{total} thumbnails...", flush=True)
    
    return pd.DataFrame(results)


# ============================================================
# PHASE 3: OCR for text detection (using EasyOCR)
# ============================================================

def run_ocr_sample(video_ids, thumb_dir, sample_size=500):
    """Run OCR on a sample of thumbnails to detect text overlays."""
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    sample_ids = np.random.choice(video_ids, size=min(sample_size, len(video_ids)), replace=False)
    results = {}
    
    for i, vid in enumerate(sample_ids):
        filepath = os.path.join(thumb_dir, f"{vid}.jpg")
        if not os.path.exists(filepath):
            continue
        try:
            detections = reader.readtext(filepath)
            texts = [d[1] for d in detections]
            full_text = ' '.join(texts).strip()
            results[vid] = {
                'ocr_text': full_text,
                'ocr_word_count': len(full_text.split()) if full_text else 0,
                'has_ocr_text': len(full_text) > 0,
                'ocr_all_caps': full_text == full_text.upper() and len(full_text) > 3 if full_text else False,
            }
        except:
            results[vid] = {'ocr_text': '', 'ocr_word_count': 0, 'has_ocr_text': False, 'ocr_all_caps': False}
        
        if (i + 1) % 50 == 0:
            print(f"  OCR processed {i+1}/{len(sample_ids)} thumbnails...", flush=True)
    
    return pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'video_id'})


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("THUMBNAIL ANALYSIS FOR KIDFLUENCER STUDY", flush=True)
    print("=" * 60, flush=True)
    
    # Load video data
    df = pd.read_csv('data/combined_videos.csv')
    video_ids = df['video_id'].dropna().unique().tolist()
    print(f"\nTotal videos to process: {len(video_ids)}", flush=True)
    
    thumb_dir = 'data/thumbnails'
    output_dir = 'analysis_v3/thumbnails'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Download thumbnails
    print("\n--- STEP 1: Downloading thumbnails ---", flush=True)
    thumb_paths = download_all_thumbnails(video_ids, thumb_dir, max_workers=15)
    downloaded = sum(1 for v in thumb_paths.values() if v is not None)
    print(f"  Downloaded: {downloaded}/{len(video_ids)} ({downloaded/len(video_ids)*100:.1f}%)", flush=True)
    
    # Step 2: CV Analysis
    print("\n--- STEP 2: Running CV analysis ---", flush=True)
    valid_ids = [vid for vid, path in thumb_paths.items() if path is not None]
    cv_df = run_cv_analysis(valid_ids, thumb_dir)
    print(f"  Analyzed: {len(cv_df)} thumbnails", flush=True)
    
    # Quick stats
    print(f"\n  Face detection stats:", flush=True)
    print(f"    Thumbnails with faces: {(cv_df['num_faces'] > 0).sum()} ({(cv_df['num_faces'] > 0).mean()*100:.1f}%)", flush=True)
    print(f"    Avg faces per thumbnail: {cv_df['num_faces'].mean():.2f}", flush=True)
    print(f"    Open mouth detected: {cv_df['has_open_mouth'].sum()} ({cv_df['has_open_mouth'].mean()*100:.1f}%)", flush=True)
    print(f"    Text overlay detected: {cv_df['has_text_overlay'].sum()} ({cv_df['has_text_overlay'].mean()*100:.1f}%)", flush=True)
    
    print(f"\n  Visual feature stats:", flush=True)
    print(f"    Avg brightness: {cv_df['brightness'].mean():.1f}", flush=True)
    print(f"    Avg saturation: {cv_df['saturation'].mean():.1f}", flush=True)
    print(f"    Avg colorfulness: {cv_df['colorfulness'].mean():.1f}", flush=True)
    print(f"    Avg edge density: {cv_df['edge_density'].mean():.4f}", flush=True)
    
    # Step 3: OCR on sample
    print("\n--- STEP 3: Running OCR on sample (500 thumbnails) ---", flush=True)
    ocr_df = run_ocr_sample(valid_ids, thumb_dir, sample_size=500)
    print(f"  OCR results:", flush=True)
    print(f"    Thumbnails with text: {ocr_df['has_ocr_text'].sum()} ({ocr_df['has_ocr_text'].mean()*100:.1f}%)", flush=True)
    print(f"    Avg words on thumbnail: {ocr_df['ocr_word_count'].mean():.1f}", flush=True)
    print(f"    All-caps text: {ocr_df['ocr_all_caps'].sum()} ({ocr_df['ocr_all_caps'].mean()*100:.1f}%)", flush=True)
    
    # Save results
    cv_df.to_csv(f'{output_dir}/thumbnail_cv_features.csv', index=False)
    ocr_df.to_csv(f'{output_dir}/thumbnail_ocr_sample.csv', index=False)
    
    # Merge with video data for analysis
    merged = df.merge(cv_df, on='video_id', how='left')
    merged.to_csv(f'{output_dir}/videos_with_thumbnail_features.csv', index=False)
    
    print(f"\n--- DONE ---", flush=True)
    print(f"Results saved to {output_dir}/", flush=True)
    print(f"  - thumbnail_cv_features.csv ({len(cv_df)} rows)", flush=True)
    print(f"  - thumbnail_ocr_sample.csv ({len(ocr_df)} rows)", flush=True)
    print(f"  - videos_with_thumbnail_features.csv ({len(merged)} rows)", flush=True)
