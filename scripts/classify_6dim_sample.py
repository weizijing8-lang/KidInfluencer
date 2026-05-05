#!/usr/bin/env python3
"""
Classify a stratified sample (50 videos per channel) using gpt-4.1-mini.
6 dimensions based on UNCRC framework + child exploitation literature.
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'classification_6dim_sample.csv')

BATCH_SIZE = 20  # Smaller batch for 6 dimensions to avoid truncation
DELAY = 0.3

SYSTEM_PROMPT = """You are an expert in children's media and child exploitation research.
You will classify YouTube video titles from kidfluencer/family vlog channels along 6 dimensions.
These dimensions are grounded in the UN Convention on the Rights of the Child (UNCRC) and academic literature on kidfluencer exploitation.

For each title, output a JSON object with these fields:

1. "performative": (UNCRC Art. 32 - Economic Protection; Freitas 2024 "playbour")
   1 = The child is clearly performing/working FOR the video. The activity would not happen without a camera. Examples: challenges, roleplay, scripted skits, dance routines, unboxing, reviews, pranks, games designed for content, tutorials, "day in the life" that is clearly staged.
   0 = Organic/natural activity that would happen regardless of filming. Examples: birthday celebrations, vacations, genuine family outings, daily life documentation.
   -1 = Ambiguous or clearly no child involved.

2. "emotional_bait": (UNCRC Art. 19 - Freedom from Harm; Clark 2025 "manufactured emotional scenarios")
   1 = The title uses exaggerated emotional language or manufactured drama to attract clicks. Indicators: ALL CAPS shouting, excessive punctuation (!!!), fake emergencies ("RUSHED TO HOSPITAL"), exaggerated reactions ("SHE CRIED", "I CAN'T BELIEVE"), manufactured urgency, sensationalized everyday events, emotional manipulation of viewer ("YOU WON'T BELIEVE").
   0 = The title is calm, descriptive, or matter-of-fact.

3. "narrative_conflict": (UNCRC Art. 19 - Freedom from Harm; Clark 2025 "scripted conflict")
   1 = The title implies interpersonal conflict, mystery, or dramatic tension involving the child. Examples: theft ("WHO STOLE..."), confrontation ("CONFRONTING..."), betrayal, punishment, secrets revealed, someone getting caught, villain/hero dynamics, "gone wrong", revenge, breaking rules.
   0 = No narrative tension or interpersonal conflict.

4. "challenge_format": (UNCRC Art. 32 - Economic Protection; ILO Convention 182)
   1 = The title indicates a challenge, competition, or structured game format. Examples: "24 HOURS...", "LAST TO LEAVE...", "...VS...", "WHO CAN...", "$10,000 CHALLENGE", "TRY NOT TO LAUGH", dares, races, contests, any structured competitive activity designed for content.
   0 = Not a challenge/competition format.

5. "commercial_content": (UNCRC Art. 32 - Economic Protection; Hudders & Beuckels 2024)
   1 = The title references specific brands, products, stores, or is clearly a sponsored/commercial activity. Examples: "UNBOXING NEW iPHONE", "TESTING SLIME FROM AMAZON", "TARGET SHOPPING SPREE", specific toy brand names (Hot Wheels, Barbie, Roblox), app names, store names, haul videos.
   0 = No specific brand/product/commercial reference. Generic words like "toy" or "game" = 0.

6. "privacy_violation": (UNCRC Art. 16 - Right to Privacy; Clark 2025 "privacy")
   1 = The title suggests sharing the child's private, embarrassing, or intimate moments that a reasonable child would not want publicly shared. Examples: potty training, bathroom moments, tantrums/meltdowns, medical procedures, puberty/body changes, bedwetting, disciplinary moments, "caught doing [embarrassing thing]", naked/bath time, crying breakdowns filmed for content.
   0 = No obvious privacy concern - the content seems like something a child would be comfortable having shared publicly.

IMPORTANT RULES:
- Only output a JSON array of objects, one per title, in the same order as input.
- Be strict with "commercial_content": only mark 1 if a SPECIFIC brand/product/store is named.
- Be strict with "privacy_violation": only mark 1 if the title clearly indicates private/embarrassing content.
- "narrative_conflict" focuses on INTERPERSONAL conflict or mystery, not just any activity.
- "challenge_format": ANY structured game/challenge/competition format = 1.
- "performative": ANY content clearly produced for YouTube = 1.
"""


def classify_batch(titles):
    """Classify a batch of titles using gpt-4.1-mini."""
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    user_msg = f"Classify these {len(titles)} video titles. Return ONLY a JSON array of {len(titles)} objects.\n\n{titles_text}"

    default = {"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, 
               "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0}
    
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
                results.append(default.copy())
            results = results[:len(titles)]
        return results
    except json.JSONDecodeError as e:
        print(f"  JSON ERROR: {e}", flush=True)
        return [default.copy() for _ in titles]
    except Exception as e:
        print(f"  API ERROR: {e}", flush=True)
        time.sleep(5)
        return [default.copy() for _ in titles]


def main():
    print(f"[{datetime.now()}] Starting 6-dimension classification with {MODEL}")
    
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
                'privacy_violation': cls.get('privacy_violation', 0),
            })
        
        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            pct = batch_end / len(titles) * 100
            print(f"  Batch {batch_num+1}/{total_batches} ({pct:.1f}%) - {batch_end}/{len(titles)}", flush=True)
        
        time.sleep(DELAY)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dims = ['performative', 'emotional_bait', 'narrative_conflict',
            'challenge_format', 'commercial_content', 'privacy_violation']
    fieldnames = ['id', 'title', 'channel_short_name', 'viewCount'] + dims
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
    for dim in dims:
        if dim == 'performative':
            cnt = (rdf[dim]==1).sum()
            pct = (rdf[dim]==1).mean()*100
        else:
            cnt = (rdf[dim]==1).sum()
            pct = (rdf[dim]==1).mean()*100
        print(f"  {dim:20s}: {cnt:4d} ({pct:.1f}%)")
    
    print(f"\n=== VIEW BOOST (median) ===")
    for dim in dims:
        if dim == 'performative':
            g1 = rdf[rdf[dim]==1]['viewCount'].median()
            g0 = rdf[rdf[dim]==0]['viewCount'].median()
        else:
            g1 = rdf[rdf[dim]==1]['viewCount'].median()
            g0 = rdf[rdf[dim]==0]['viewCount'].median()
        if g0 and g0 > 0:
            boost = (g1 - g0) / g0 * 100
            print(f"  {dim:20s}: YES={g1:>12,.0f} | NO={g0:>12,.0f} | boost={boost:+.1f}%")
    
    # Sample titles for quality check
    print(f"\n=== SAMPLE TITLES (privacy_violation=1) ===")
    priv = rdf[rdf['privacy_violation']==1].head(10)
    for _, row in priv.iterrows():
        print(f"  [{row['channel_short_name']}] {row['title']}")
    
    print(f"\n=== SAMPLE TITLES (narrative_conflict=1, emotional_bait=1) ===")
    both = rdf[(rdf['narrative_conflict']==1) & (rdf['emotional_bait']==1)].head(10)
    for _, row in both.iterrows():
        print(f"  [{row['channel_short_name']}] {row['title']}")


if __name__ == '__main__':
    main()
