#!/usr/bin/env python3
"""Vision API analysis using GPT-4.1-mini on a subset of thumbnails (5/channel = 236).
Analyzes: child presence, emotion, visual clickbait, exploitation concern level.
Saves incrementally to avoid data loss."""

import os
import sys
import csv
import json
import time
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI()

THUMB_DIR = Path('/home/ubuntu/KidInfluencer/data/thumbnails_sample')
SAMPLE_FILE = '/home/ubuntu/KidInfluencer/analysis_discovery/vision_sample_ids.csv'
OUTPUT_FILE = '/home/ubuntu/KidInfluencer/analysis_discovery/thumbnail_vision_v2.csv'

VISION_PROMPT = """Analyze this YouTube video thumbnail from a child/family influencer channel. 
Assess the following dimensions:

1. child_present: Is a child (under 18) visible? (yes/no)
2. child_emotion: What emotion does the child appear to display? (happy/sad/surprised/scared/angry/neutral/crying/distressed/none)
3. emotion_appears_genuine: Does the child's emotion appear genuine or performed/exaggerated for camera? (genuine/performed/unclear/na)
4. visual_clickbait_elements: List any clickbait visual elements (e.g., arrows, circles, exaggerated text, bright colors, shocked faces). Comma-separated or "none".
5. exploitation_concern: Rate exploitation concern level 0-3:
   0 = No concern (normal family content)
   1 = Low concern (mild performativity)
   2 = Moderate concern (child appears uncomfortable, exaggerated emotions, or privacy issues)
   3 = High concern (child distress, inappropriate content, or clear exploitation signals)
6. concern_reason: Brief explanation for the exploitation concern rating (max 20 words).

Respond ONLY in this exact JSON format:
{"child_present": "yes/no", "child_emotion": "...", "emotion_appears_genuine": "...", "visual_clickbait_elements": "...", "exploitation_concern": 0-3, "concern_reason": "..."}"""

def encode_image(image_path):
    """Encode image to base64 for the API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_path(video_id):
    """Find the thumbnail file for a video ID."""
    jpg = THUMB_DIR / f"{video_id}.jpg"
    webp = THUMB_DIR / f"{video_id}.webp"
    if jpg.exists():
        return jpg
    elif webp.exists():
        return webp
    return None

def analyze_thumbnail(image_path):
    """Call Vision API to analyze a single thumbnail."""
    base64_image = encode_image(image_path)
    ext = image_path.suffix.lower()
    media_type = "image/jpeg" if ext == ".jpg" else "image/webp"
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=300,
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    # Parse JSON from response
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {
                "child_present": "unknown",
                "child_emotion": "unknown",
                "emotion_appears_genuine": "unknown",
                "visual_clickbait_elements": "unknown",
                "exploitation_concern": -1,
                "concern_reason": "parse_error"
            }
    
    return result

def main():
    import pandas as pd
    
    sample = pd.read_csv(SAMPLE_FILE)
    video_ids = sample['id'].tolist()
    
    print(f"Vision API analysis on {len(video_ids)} thumbnails", flush=True)
    
    # Check for existing progress
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(existing['video_id'].tolist())
        print(f"  Resuming: {len(processed_ids)} already processed", flush=True)
    
    fieldnames = ['video_id', 'child_present', 'child_emotion', 'emotion_appears_genuine', 
                  'visual_clickbait_elements', 'exploitation_concern', 'concern_reason']
    
    # Open in append mode if resuming
    mode = 'a' if processed_ids else 'w'
    with open(OUTPUT_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not processed_ids:
            writer.writeheader()
        
        success = 0
        errors = 0
        
        for i, video_id in enumerate(video_ids):
            if video_id in processed_ids:
                continue
            
            image_path = get_image_path(video_id)
            if image_path is None:
                continue
            
            try:
                result = analyze_thumbnail(image_path)
                row = {
                    'video_id': video_id,
                    'child_present': result.get('child_present', 'unknown'),
                    'child_emotion': result.get('child_emotion', 'unknown'),
                    'emotion_appears_genuine': result.get('emotion_appears_genuine', 'unknown'),
                    'visual_clickbait_elements': result.get('visual_clickbait_elements', 'unknown'),
                    'exploitation_concern': result.get('exploitation_concern', -1),
                    'concern_reason': result.get('concern_reason', ''),
                }
                writer.writerow(row)
                f.flush()
                success += 1
                
            except Exception as e:
                errors += 1
                print(f"  Error on {video_id}: {e}", flush=True)
                # Write error row
                writer.writerow({
                    'video_id': video_id,
                    'child_present': 'error',
                    'child_emotion': 'error',
                    'emotion_appears_genuine': 'error',
                    'visual_clickbait_elements': 'error',
                    'exploitation_concern': -1,
                    'concern_reason': str(e)[:50],
                })
                f.flush()
                time.sleep(2)  # Back off on error
            
            if (success + errors) % 20 == 0:
                print(f"  Processed {success + errors + len(processed_ids)}/{len(video_ids)} (success: {success}, errors: {errors})", flush=True)
            
            # Rate limiting: ~0.5s between calls
            time.sleep(0.3)
    
    print(f"\nDone! Total: {success} success, {errors} errors", flush=True)
    
    # Summary stats
    df = pd.read_csv(OUTPUT_FILE)
    print(f"\n=== Vision API Results Summary ===")
    print(f"Total analyzed: {len(df)}")
    print(f"Child present: {(df['child_present'] == 'yes').sum()} ({(df['child_present'] == 'yes').mean()*100:.1f}%)")
    print(f"Exploitation concern distribution:")
    print(df['exploitation_concern'].value_counts().sort_index())
    print(f"\nEmotion distribution:")
    print(df['child_emotion'].value_counts())
    print(f"\nGenuineness:")
    print(df['emotion_appears_genuine'].value_counts())

if __name__ == "__main__":
    main()
