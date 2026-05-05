#!/usr/bin/env python3
"""
Classify a stratified sample (50 videos per channel) using gpt-4.1-mini.
This gives us ~2400 videos for initial analysis before committing to full run.
"""
import sys
import os
import json
import csv
import time
from datetime import datetime

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-mini"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_discovery')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'classification_5dim_sample.csv')

BATCH_SIZE = 25
DELAY = 0.3

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
    """Classify a batch of titles using gpt-4.1-mini."""
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    user_msg = f"Classify these {len(titles)} video titles. Return ONLY a JSON array of {len(titles)} objects.\n\n{titles_text}"

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
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        
        results = json.loads(content)
        if len(results) != len(titles):
            print(f"  WARNING: Expected {len(titles)}, got {len(results)}", flush=True)
            while len(results) < len(titles):
                results.append({"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0})
            results = results[:len(titles)]
        return results
    except json.JSONDecodeError as e:
        print(f"  JSON ERROR: {e}", flush=True)
        return [{"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0}] * len(titles)
    except Exception as e:
        print(f"  API ERROR: {e}", flush=True)
        time.sleep(5)
        return [{"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0}] * len(titles)


def main():
    print(f"[{datetime.now()}] Starting stratified sample classification with {MODEL}")
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'combined_family_videos.csv'))
    print(f"Full dataset: {len(df)} videos, {df['channel_short_name'].nunique()} channels")
    
    # Stratified sample: 50 per channel
    samples = []
    for ch, grp in df.groupby('channel_short_name'):
        n = min(50, len(grp))
        samples.append(grp.sample(n, random_state=42))
    sample = pd.concat(samples, ignore_index=True)
    print(f"Stratified sample: {len(sample)} videos, {sample['channel_short_name'].nunique()} channels")
    print()
    
    # Process
    titles = sample['title'].fillna('').tolist()
    ids = sample['id'].tolist()
    channels = sample['channel_short_name'].tolist()
    views = sample['viewCount'].tolist()
    
    total_batches = (len(titles) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches: {total_batches}")
    
    results = []
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
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
        
        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            pct = batch_end / len(titles) * 100
            print(f"  Batch {batch_num+1}/{total_batches} ({pct:.1f}%) - {batch_end}/{len(titles)}", flush=True)
        
        time.sleep(DELAY)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ['id', 'title', 'channel_short_name', 'viewCount',
                  'performative', 'emotional_bait', 'narrative_conflict',
                  'challenge_format', 'commercial_content']
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} videos classified -> {OUTPUT_FILE}")
    print(f"{'='*60}")
    
    # Quick stats
    rdf = pd.DataFrame(results)
    rdf['viewCount'] = pd.to_numeric(rdf['viewCount'], errors='coerce')
    print(f"\n=== DISTRIBUTIONS ===")
    print(f"Performative:     {(rdf['performative']==1).sum():4d} ({(rdf['performative']==1).mean()*100:.1f}%)")
    print(f"Emotional bait:   {(rdf['emotional_bait']==1).sum():4d} ({(rdf['emotional_bait']==1).mean()*100:.1f}%)")
    print(f"Narrative conflict:{(rdf['narrative_conflict']==1).sum():4d} ({(rdf['narrative_conflict']==1).mean()*100:.1f}%)")
    print(f"Challenge format: {(rdf['challenge_format']==1).sum():4d} ({(rdf['challenge_format']==1).mean()*100:.1f}%)")
    print(f"Commercial:       {(rdf['commercial_content']==1).sum():4d} ({(rdf['commercial_content']==1).mean()*100:.1f}%)")
    
    print(f"\n=== VIEW BOOST (median) ===")
    for dim in ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content']:
        if dim == 'performative':
            g1 = rdf[rdf[dim]==1]['viewCount'].median()
            g0 = rdf[rdf[dim]==0]['viewCount'].median()
        else:
            g1 = rdf[rdf[dim]==1]['viewCount'].median()
            g0 = rdf[rdf[dim]==0]['viewCount'].median()
        if g0 and g0 > 0:
            boost = (g1 - g0) / g0 * 100
            print(f"  {dim:20s}: YES={g1:>12,.0f} | NO={g0:>12,.0f} | boost={boost:+.1f}%")
    
    # Channel-level breakdown
    print(f"\n=== TOP CHANNELS BY PERFORMATIVE RATE ===")
    ch_stats = rdf.groupby('channel_short_name').agg(
        n=('id', 'count'),
        perf_rate=('performative', lambda x: (x==1).mean()),
        bait_rate=('emotional_bait', lambda x: (x==1).mean()),
        conflict_rate=('narrative_conflict', lambda x: (x==1).mean()),
        challenge_rate=('challenge_format', lambda x: (x==1).mean()),
        commercial_rate=('commercial_content', lambda x: (x==1).mean()),
        median_views=('viewCount', 'median'),
    ).sort_values('perf_rate', ascending=False)
    print(ch_stats.head(15).to_string())


if __name__ == '__main__':
    main()
