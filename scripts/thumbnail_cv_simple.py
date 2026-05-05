#!/usr/bin/env python3
"""Simple CV analysis - saves incrementally to avoid OOM crashes."""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import csv, os, time, gc

THUMB_DIR = '/home/ubuntu/KidInfluencer/data/thumbnails_sample'
OUTPUT_DIR = '/home/ubuntu/KidInfluencer/analysis_discovery'
CV_OUTPUT = os.path.join(OUTPUT_DIR, 'thumbnail_cv_v2.csv')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

thumb_files = list(Path(THUMB_DIR).glob('*.jpg')) + list(Path(THUMB_DIR).glob('*.webp'))
print(f'Found {len(thumb_files)} thumbnails', flush=True)

fieldnames = ['video_id', 'num_faces', 'num_smiles', 'smile_ratio', 'saturation', 'brightness', 'color_std', 'red_ratio', 'yellow_ratio', 'edge_density', 'face_area_ratio']

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CV_OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    for i, thumb_path in enumerate(thumb_files):
        video_id = thumb_path.stem
        img = cv2.imread(str(thumb_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        num_faces = len(faces)
        
        num_smiles = 0
        for (x, y, fw, fh) in faces:
            roi_gray = gray[y:y+fh, x:x+fw]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            if len(smiles) > 0:
                num_smiles += 1
        
        saturation = float(hsv[:, :, 1].mean())
        brightness = float(hsv[:, :, 2].mean())
        color_std = float(img.std())
        
        red_mask = ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 100)
        yellow_mask = ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 35)) & (hsv[:, :, 1] > 100)
        red_ratio = float(red_mask.sum()) / (h * w)
        yellow_ratio = float(yellow_mask.sum()) / (h * w)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.sum()) / (h * w * 255)
        
        face_area_ratio = sum(fw * fh for (x, y, fw, fh) in faces) / (h * w) if num_faces > 0 else 0
        
        writer.writerow({
            'video_id': video_id,
            'num_faces': num_faces,
            'num_smiles': num_smiles,
            'smile_ratio': round(num_smiles / num_faces, 3) if num_faces > 0 else 0,
            'saturation': round(saturation, 1),
            'brightness': round(brightness, 1),
            'color_std': round(color_std, 1),
            'red_ratio': round(red_ratio, 4),
            'yellow_ratio': round(yellow_ratio, 4),
            'edge_density': round(edge_density, 4),
            'face_area_ratio': round(face_area_ratio, 4),
        })
        
        del img, gray, hsv, edges
        if (i + 1) % 200 == 0:
            gc.collect()
            print(f'  Processed {i+1}/{len(thumb_files)}', flush=True)

print(f'\nDone! Saved to {CV_OUTPUT}', flush=True)

# Quick stats
df = pd.read_csv(CV_OUTPUT)
print(f'Total: {len(df)} thumbnails')
print(f'Face detected: {(df["num_faces"] > 0).sum()} ({(df["num_faces"] > 0).mean()*100:.1f}%)')
print(f'Avg faces: {df["num_faces"].mean():.2f}')
print(f'Smile detected: {(df["num_smiles"] > 0).sum()} ({(df["num_smiles"] > 0).mean()*100:.1f}%)')
print(f'Avg saturation: {df["saturation"].mean():.1f}')
print(f'Avg brightness: {df["brightness"].mean():.1f}')
print(f'Avg red_ratio: {df["red_ratio"].mean():.4f}')
print(f'Avg edge_density: {df["edge_density"].mean():.4f}')
