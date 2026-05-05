#!/usr/bin/env python3
"""
CV analysis of thumbnails for the 2306 sample videos.
Uses OpenCV for face detection + color analysis, and GPT-4.1-mini Vision for 
emotion detection on a subset.
"""
import sys
import os
import json
import csv
import time
import base64
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI()

THUMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'thumbnails_sample')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_discovery')
CV_OUTPUT = os.path.join(OUTPUT_DIR, 'thumbnail_cv_v2.csv')
VISION_OUTPUT = os.path.join(OUTPUT_DIR, 'thumbnail_vision_v2.csv')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def analyze_thumbnail_cv(img_path):
    """Analyze a single thumbnail with OpenCV."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    num_faces = len(faces)
    
    # Smile detection on each face
    num_smiles = 0
    for (x, y, fw, fh) in faces:
        roi_gray = gray[y:y+fh, x:x+fw]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            num_smiles += 1
    
    # Color analysis
    saturation = hsv[:, :, 1].mean()
    brightness = hsv[:, :, 2].mean()
    
    # Color variance (visual complexity)
    color_std = img.std()
    
    # Red/yellow dominance (attention-grabbing colors)
    red_mask = ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 100)
    yellow_mask = ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 35)) & (hsv[:, :, 1] > 100)
    red_ratio = red_mask.sum() / (h * w)
    yellow_ratio = yellow_mask.sum() / (h * w)
    
    # Text-like regions (high contrast areas that might be overlaid text)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / (h * w * 255)
    
    # Face area ratio (how much of the thumbnail is faces)
    face_area_ratio = sum(fw * fh for (x, y, fw, fh) in faces) / (h * w) if num_faces > 0 else 0
    
    return {
        'num_faces': num_faces,
        'num_smiles': num_smiles,
        'smile_ratio': num_smiles / num_faces if num_faces > 0 else 0,
        'saturation': round(saturation, 1),
        'brightness': round(brightness, 1),
        'color_std': round(color_std, 1),
        'red_ratio': round(red_ratio, 4),
        'yellow_ratio': round(yellow_ratio, 4),
        'edge_density': round(edge_density, 4),
        'face_area_ratio': round(face_area_ratio, 4),
    }


def analyze_thumbnail_vision(img_path):
    """Analyze a thumbnail with GPT-4.1-mini Vision for emotion detection."""
    with open(img_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Analyze this YouTube video thumbnail. Answer these questions in JSON format:
1. "child_present": true/false - Is there a child (under 16) visible?
2. "child_emotion": one of ["happy", "excited", "surprised", "scared", "sad", "crying", "angry", "neutral", "not_visible"] - What emotion is the child displaying?
3. "emotion_appears_genuine": true/false - Does the emotion appear genuine or exaggerated/performed for camera?
4. "visual_clickbait_elements": list of any present: ["arrows", "circles", "emoji_overlay", "text_overlay", "bright_border", "shocked_face", "split_screen", "before_after", "none"]
5. "exploitation_concern": 0-3 scale (0=none, 1=mild, 2=moderate, 3=high) - Does this thumbnail raise child exploitation concerns?
6. "concern_reason": brief explanation if exploitation_concern > 0, else ""

Return ONLY valid JSON."""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
    except Exception as e:
        print(f"  Vision API error: {e}", flush=True)
        return None


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting thumbnail CV analysis")
    
    # Get list of thumbnails
    thumb_files = list(Path(THUMB_DIR).glob('*.jpg')) + list(Path(THUMB_DIR).glob('*.webp'))
    print(f"Found {len(thumb_files)} thumbnails")
    
    # Phase 1: OpenCV analysis on ALL thumbnails
    print(f"\n=== Phase 1: OpenCV analysis on {len(thumb_files)} thumbnails ===")
    cv_results = []
    for i, thumb_path in enumerate(thumb_files):
        video_id = thumb_path.stem
        result = analyze_thumbnail_cv(str(thumb_path))
        if result:
            result['video_id'] = video_id
            cv_results.append(result)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(thumb_files)}", flush=True)
    
    # Save CV results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(CV_OUTPUT, index=False)
    print(f"\nCV results saved: {len(cv_results)} thumbnails -> {CV_OUTPUT}")
    
    # Stats
    print(f"\n=== CV STATS ===")
    print(f"  Face detected: {(cv_df['num_faces'] > 0).sum()} ({(cv_df['num_faces'] > 0).mean()*100:.1f}%)")
    print(f"  Avg faces per thumbnail: {cv_df['num_faces'].mean():.2f}")
    print(f"  Smile detected: {(cv_df['num_smiles'] > 0).sum()} ({(cv_df['num_smiles'] > 0).mean()*100:.1f}%)")
    print(f"  Avg saturation: {cv_df['saturation'].mean():.1f}")
    print(f"  Avg brightness: {cv_df['brightness'].mean():.1f}")
    print(f"  Avg red ratio: {cv_df['red_ratio'].mean():.4f}")
    print(f"  Avg edge density: {cv_df['edge_density'].mean():.4f}")
    
    # Phase 2: Vision API on a stratified subset (5 per channel, ~240 total)
    print(f"\n=== Phase 2: Vision API analysis (stratified subset) ===")
    
    # Load the 6dim classification to get channel info
    cls_file = os.path.join(OUTPUT_DIR, 'classification_6dim_sample.csv')
    if os.path.exists(cls_file):
        cls_df = pd.read_csv(cls_file)
        # Get video IDs that have thumbnails
        thumb_ids = set(cv_df['video_id'].tolist())
        cls_with_thumb = cls_df[cls_df['id'].isin(thumb_ids)]
        
        # Stratified sample: 5 per channel for vision analysis
        vision_sample = []
        for ch, grp in cls_with_thumb.groupby('channel_short_name'):
            n = min(5, len(grp))
            vision_sample.append(grp.sample(n, random_state=42))
        vision_sample = pd.concat(vision_sample, ignore_index=True)
        print(f"Vision API sample: {len(vision_sample)} videos from {vision_sample['channel_short_name'].nunique()} channels")
    else:
        # Fallback: random sample of 200
        sample_ids = cv_df.sample(min(200, len(cv_df)), random_state=42)['video_id'].tolist()
        vision_sample = pd.DataFrame({'id': sample_ids})
        print(f"Vision API sample: {len(vision_sample)} videos (random)")
    
    vision_results = []
    for i, row in vision_sample.iterrows():
        vid_id = row['id']
        # Find thumbnail file
        thumb_path = None
        for ext in ['.jpg', '.webp']:
            p = os.path.join(THUMB_DIR, f"{vid_id}{ext}")
            if os.path.exists(p):
                thumb_path = p
                break
        
        if not thumb_path:
            continue
        
        result = analyze_thumbnail_vision(thumb_path)
        if result:
            result['video_id'] = vid_id
            result['channel'] = row.get('channel_short_name', '')
            vision_results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Vision: {len(vision_results)}/{len(vision_sample)} done", flush=True)
        
        time.sleep(0.3)
    
    # Save vision results
    vision_df = pd.DataFrame(vision_results)
    vision_df.to_csv(VISION_OUTPUT, index=False)
    print(f"\nVision results saved: {len(vision_results)} thumbnails -> {VISION_OUTPUT}")
    
    # Vision stats
    if len(vision_results) > 0:
        print(f"\n=== VISION STATS ===")
        print(f"  Child present: {vision_df['child_present'].sum()} ({vision_df['child_present'].mean()*100:.1f}%)")
        if 'child_emotion' in vision_df.columns:
            print(f"  Emotion distribution:")
            print(vision_df['child_emotion'].value_counts().to_string())
        if 'exploitation_concern' in vision_df.columns:
            print(f"  Exploitation concern distribution:")
            print(vision_df['exploitation_concern'].value_counts().sort_index().to_string())
        if 'emotion_appears_genuine' in vision_df.columns:
            genuine = vision_df[vision_df['child_present']==True]['emotion_appears_genuine']
            print(f"  Genuine emotion (where child present): {genuine.sum()}/{len(genuine)} ({genuine.mean()*100:.1f}%)")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE!")


if __name__ == '__main__':
    main()
