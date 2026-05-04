"""
CV analysis of kidfluencer video thumbnails using:
1. OpenCV Haar Cascades for face detection (fast, local)
2. OpenAI Vision API (gpt-4.1-mini) for emotion/context analysis on a sample

For each thumbnail:
- Number of faces detected
- Face sizes (child vs adult proxy)
- Smile detection
- Color saturation, brightness
- For a sample: LLM vision analysis of emotional context
"""
import pandas as pd
import numpy as np
import cv2
import os, sys, json, glob, time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

print("Loading data and thumbnails...", flush=True)

# Find all thumbnails
thumb_dir = 'data/thumbnails'
thumb_files = glob.glob(f'{thumb_dir}/*.jpg') + glob.glob(f'{thumb_dir}/*.png') + glob.glob(f'{thumb_dir}/*.webp')
print(f"Found {len(thumb_files)} thumbnail files", flush=True)

# Map to video IDs
thumb_map = {}
for f in thumb_files:
    vid_id = Path(f).stem
    thumb_map[vid_id] = f

# Load labeled data
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
df_with_thumb = df[df['id'].isin(thumb_map)].copy()
print(f"Videos with thumbnails: {len(df_with_thumb)}", flush=True)

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Process all thumbnails with OpenCV
print(f"\nProcessing {len(df_with_thumb)} thumbnails with OpenCV...", flush=True)
results = []

for idx, (_, row) in enumerate(df_with_thumb.iterrows()):
    vid_id = row['id']
    thumb_path = thumb_map[vid_id]
    
    try:
        img = cv2.imread(thumb_path)
        if img is None:
            results.append({'id': vid_id, 'error': True})
            continue
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        n_faces = len(faces)
        
        # Face metrics
        has_large_face = False
        total_face_area_ratio = 0
        n_smiles = 0
        
        for (fx, fy, fw, fh) in faces:
            face_area_ratio = (fw * fh) / (w * h)
            total_face_area_ratio += face_area_ratio
            if face_area_ratio > 0.05:
                has_large_face = True
            
            # Smile detection within face region
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            if len(smiles) > 0:
                n_smiles += 1
        
        # Color analysis
        mean_saturation = float(hsv[:,:,1].mean())
        mean_brightness = float(hsv[:,:,2].mean())
        
        # Edge density (visual complexity)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.mean())
        
        # Color variance (how colorful)
        color_std = float(hsv[:,:,0].std())
        
        results.append({
            'id': vid_id,
            'n_faces': n_faces,
            'has_face': n_faces > 0,
            'has_large_face': has_large_face,
            'total_face_area_ratio': total_face_area_ratio,
            'n_smiles': n_smiles,
            'has_smile': n_smiles > 0,
            'smile_ratio': n_smiles / n_faces if n_faces > 0 else 0,
            'mean_saturation': mean_saturation,
            'mean_brightness': mean_brightness,
            'edge_density': edge_density,
            'color_std': color_std,
            'error': False
        })
        
    except Exception as e:
        results.append({'id': vid_id, 'error': True})
    
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/{len(df_with_thumb)} thumbnails processed", flush=True)

cv_df = pd.DataFrame(results)
cv_df = cv_df[~cv_df['error']].drop('error', axis=1)
cv_df.to_csv('analysis_discovery/thumbnail_cv_results.csv', index=False)

print(f"\nOpenCV results: {len(cv_df)} thumbnails", flush=True)
print(f"  Face detection rate: {cv_df['has_face'].mean()*100:.1f}%", flush=True)
print(f"  Large face rate: {cv_df['has_large_face'].mean()*100:.1f}%", flush=True)
print(f"  Smile rate (among faces): {cv_df[cv_df['has_face']]['has_smile'].mean()*100:.1f}%", flush=True)
print(f"  Mean saturation: {cv_df['mean_saturation'].mean():.1f}", flush=True)
print(f"  Mean brightness: {cv_df['mean_brightness'].mean():.1f}", flush=True)

# ============================================================
# LLM Vision Analysis on a stratified sample
# ============================================================
print(f"\n{'='*60}", flush=True)
print("LLM Vision Analysis (stratified sample)...", flush=True)

from openai import OpenAI
import base64

client = OpenAI()

# Sample: take 300 thumbnails stratified by channel
sample_per_channel = 12
sample_ids = []
for ch in df_with_thumb['channel_short_name'].unique():
    ch_ids = df_with_thumb[df_with_thumb['channel_short_name'] == ch]['id'].tolist()
    np.random.seed(42)
    n = min(sample_per_channel, len(ch_ids))
    sample_ids.extend(np.random.choice(ch_ids, n, replace=False).tolist())

print(f"  LLM sample: {len(sample_ids)} thumbnails", flush=True)

vision_results = []
for i, vid_id in enumerate(sample_ids):
    thumb_path = thumb_map[vid_id]
    title = df_with_thumb[df_with_thumb['id'] == vid_id]['title'].iloc[0]
    channel = df_with_thumb[df_with_thumb['id'] == vid_id]['channel_short_name'].iloc[0]
    
    # Encode image
    with open(thumb_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Determine mime type
    ext = Path(thumb_path).suffix.lower()
    mime = {'jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(ext, 'image/jpeg')
    
    prompt = f"""Analyze this YouTube thumbnail from a kidfluencer channel.
Title: "{title}" | Channel: {channel}

Answer in JSON:
{{
  "child_visible": true/false,
  "child_count": 0-5,
  "child_emotion": "happy"|"sad"|"scared"|"crying"|"surprised"|"neutral"|"excited"|"distressed"|"none",
  "adult_visible": true/false,
  "adult_emotion": "happy"|"sad"|"scared"|"neutral"|"excited"|"none",
  "scene_type": "indoor"|"outdoor"|"studio"|"animated"|"mixed",
  "is_animated": true/false,
  "has_text_overlay": true/false,
  "emotional_tone": "positive"|"negative"|"neutral"|"dramatic"|"exciting",
  "exploitation_concern": 0-3 (0=none, 1=mild, 2=moderate, 3=high),
  "brief_description": "one sentence describing what's shown"
}}"""

    try:
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_data}', 'detail': 'low'}}
                ]
            }],
            temperature=0,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        result['id'] = vid_id
        result['title'] = title
        result['channel'] = channel
        vision_results.append(result)
        
    except Exception as e:
        print(f"  Error on {vid_id}: {e}", flush=True)
        vision_results.append({'id': vid_id, 'title': title, 'channel': channel, 'error': str(e)})
    
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(sample_ids)} vision analyses done", flush=True)
        # Save checkpoint
        pd.DataFrame(vision_results).to_csv('analysis_discovery/thumbnail_vision_results.csv', index=False)

vision_df = pd.DataFrame(vision_results)
vision_df.to_csv('analysis_discovery/thumbnail_vision_results.csv', index=False)

print(f"\nVision analysis complete: {len(vision_df)} thumbnails", flush=True)
print(f"\nChild visible: {vision_df.get('child_visible', pd.Series()).sum()}", flush=True)
print(f"Child emotions:", flush=True)
if 'child_emotion' in vision_df.columns:
    print(vision_df['child_emotion'].value_counts().to_string(), flush=True)
print(f"\nEmotional tone:", flush=True)
if 'emotional_tone' in vision_df.columns:
    print(vision_df['emotional_tone'].value_counts().to_string(), flush=True)
print(f"\nExploitation concern:", flush=True)
if 'exploitation_concern' in vision_df.columns:
    print(vision_df['exploitation_concern'].value_counts().to_string(), flush=True)

print("\nAll saved to analysis_discovery/", flush=True)
