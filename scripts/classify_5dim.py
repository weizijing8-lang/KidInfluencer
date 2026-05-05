#!/usr/bin/env python3
"""
5-Dimension LLM Classification for Kidfluencer Videos
======================================================
Classifies each video title along 5 dimensions:
1. performative (vs organic) - Is the child performing/working for the video?
2. emotional_bait - Does the title use emotional hooks/conflict to attract clicks?
3. narrative_conflict - Is there a scripted story conflict?
4. multi_child_collab - Are multiple non-family children involved?
5. sexualization_age - Does the title contain gender/objectification/age-inappropriate language?

Uses GPT-4.1-nano for batch classification.
"""

import sys
import os
import json
import csv
import time
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-nano"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_discovery')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'classification_5dim.csv')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'classification_5dim_checkpoint.csv')

BATCH_SIZE = 30  # titles per API call (smaller to avoid JSON truncation)
DELAY = 0.2

SYSTEM_PROMPT = """You are an expert in children's media and child exploitation research.
You will classify YouTube video titles from kidfluencer/family vlog channels along 5 dimensions.

For each title, output a JSON object with these fields:

- "performative": 1 if the child is clearly performing/working FOR the video (challenges, roleplay, scripted skits, dance routines, unboxing, reviews, pranks, games designed for content). 0 if organic/natural (birthday, vacation, daily life that would happen without a camera). -1 if ambiguous or clearly no child involved.

- "emotional_bait": 1 if the title uses exaggerated emotional language or manufactured drama to attract clicks. This includes: ALL CAPS shouting, excessive punctuation (!!!), fake emergencies, exaggerated reactions ("SHE CRIED", "I CAN'T BELIEVE", "EMOTIONAL"), surprise reveals ("SURPRISING..."), manufactured urgency, or sensationalized everyday events. Basically any title that amplifies emotions beyond what the content likely warrants. 0 if the title is calm, descriptive, or matter-of-fact.

- "narrative_conflict": 1 if the title implies a story with interpersonal conflict, mystery, or dramatic tension (theft: "WHO STOLE...", confrontation: "CONFRONTING...", betrayal, punishment, secrets revealed, someone getting caught, villain/hero dynamics, "gone wrong"). 0 if no narrative tension or conflict.

- "challenge_format": 1 if the title indicates a challenge, competition, or game format (e.g., "24 HOURS...", "LAST TO LEAVE...", "...VS...", "WHO CAN...", "$10,000 CHALLENGE", "TRY NOT TO LAUGH", dares, races, contests, any structured competitive activity designed for content). 0 if not a challenge/competition format.

- "commercial_content": 1 if the title references specific brands, products, stores, or commercial activities (e.g., "UNBOXING NEW iPHONE", "TESTING SLIME FROM AMAZON", "TARGET SHOPPING SPREE", "TRYING [brand] PRODUCTS", toy names, app names, store names, haul videos, sponsored content indicators). 0 if no brand/product/commercial reference.

IMPORTANT RULES:
- Only output a JSON array of objects, one per title, in the same order as input
- "emotional_bait": YES for exaggerated clickbait style (ALL CAPS + !!! + emotional words). A simple "Valentine's Day Surprise" = 0, but "SURPRISING MY GIRLS FOR VALENTINE'S DAY!!!" = 1 because of the exaggerated formatting.
- "narrative_conflict": Focus on INTERPERSONAL conflict or mystery, not just any activity.
- "challenge_format": Mark 1 for ANY structured game/challenge/competition format, even if it seems harmless (e.g., "WHO KNOWS ME BETTER" = 1).
- "commercial_content": Mark 1 if ANY brand name, product name, store name, or commercial activity is mentioned. Generic words like "toy" or "game" without a specific brand = 0.
- "performative" should be 1 for ANY content clearly produced for YouTube (challenges, games, skits, tutorials, reviews, pranks).
"""


def classify_batch(titles):
    """Classify a batch of titles using GPT-4.1-nano."""
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    
    user_msg = f"""Classify these {len(titles)} video titles. Return ONLY a JSON array of {len(titles)} objects.

{titles_text}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=8000,
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up response
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        
        results = json.loads(content)
        
        if len(results) != len(titles):
            print(f"  WARNING: Expected {len(titles)} results, got {len(results)}")
            # Pad or truncate
            while len(results) < len(titles):
                results.append({"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "multi_child_collab": 0, "sexualization_age": 0})
            results = results[:len(titles)]
        
        return results
    
    except json.JSONDecodeError as e:
        print(f"  JSON ERROR: {e}")
        print(f"  Raw response: {content[:200]}")
        return [{"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "multi_child_collab": 0, "sexualization_age": 0}] * len(titles)
    except Exception as e:
        print(f"  API ERROR: {e}")
        return [{"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "multi_child_collab": 0, "sexualization_age": 0}] * len(titles)


def main():
    import pandas as pd
    
    print(f"[{datetime.now()}] Starting 5-dimension classification")
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'combined_family_videos.csv'))
    print(f"Total videos: {len(df)}")
    print(f"Channels: {df['channel_short_name'].nunique()}")
    
    # Check for existing checkpoint
    start_idx = 0
    existing_results = []
    if os.path.exists(CHECKPOINT_FILE):
        existing = pd.read_csv(CHECKPOINT_FILE)
        start_idx = len(existing)
        existing_results = existing.to_dict('records')
        print(f"Resuming from checkpoint: {start_idx} already classified")
    
    # Process in batches
    titles = df['title'].fillna('').tolist()
    ids = df['id'].tolist()
    channels = df['channel_short_name'].tolist()
    views = df['viewCount'].tolist()
    
    total_batches = (len(titles) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Batches remaining: {total_batches}")
    print()
    
    results = existing_results.copy()
    
    for batch_num in range(total_batches):
        batch_start = start_idx + batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(titles))
        batch_titles = titles[batch_start:batch_end]
        
        classifications = classify_batch(batch_titles)
        
        for i, cls in enumerate(classifications):
            idx = batch_start + i
            results.append({
                'id': ids[idx],
                'title': titles[idx],
                'channel_short_name': channels[idx],
                'viewCount': views[idx],
                'performative': cls.get('performative', -1),
                'emotional_bait': cls.get('emotional_bait', 0),
                'narrative_conflict': cls.get('narrative_conflict', 0),
                'challenge_format': cls.get('challenge_format', 0),
                'commercial_content': cls.get('commercial_content', 0),
            })
        
        # Progress
        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            pct = (batch_end / len(titles)) * 100
            print(f"  Batch {batch_num+1}/{total_batches} ({pct:.1f}%) - {batch_end}/{len(titles)} videos", flush=True)
        
        # Checkpoint every 50 batches
        if (batch_num + 1) % 50 == 0:
            save_checkpoint(results)
            print(f"  [CHECKPOINT] Saved {len(results)} results", flush=True)
        
        time.sleep(DELAY)
    
    # Final save
    save_results(results)
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION COMPLETE: {len(results)} videos classified")
    print(f"{'='*60}")
    
    # Quick stats
    result_df = pd.DataFrame(results)
    print(f"\nPerformative: {(result_df['performative']==1).sum()} ({(result_df['performative']==1).mean()*100:.1f}%)")
    print(f"Emotional bait: {(result_df['emotional_bait']==1).sum()} ({(result_df['emotional_bait']==1).mean()*100:.1f}%)")
    print(f"Narrative conflict: {(result_df['narrative_conflict']==1).sum()} ({(result_df['narrative_conflict']==1).mean()*100:.1f}%)")
    print(f"Challenge format: {(result_df['challenge_format']==1).sum()} ({(result_df['challenge_format']==1).mean()*100:.1f}%)")
    print(f"Commercial content: {(result_df['commercial_content']==1).sum()} ({(result_df['commercial_content']==1).mean()*100:.1f}%)")


def save_checkpoint(results):
    """Save checkpoint to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ['id', 'title', 'channel_short_name', 'viewCount', 
                  'performative', 'emotional_bait', 'narrative_conflict', 
                  'challenge_format', 'commercial_content']
    with open(CHECKPOINT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def save_results(results):
    """Save final results to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ['id', 'title', 'channel_short_name', 'viewCount', 
                  'performative', 'emotional_bait', 'narrative_conflict', 
                  'challenge_format', 'commercial_content']
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    # Also save checkpoint
    save_checkpoint(results)


if __name__ == '__main__':
    main()
