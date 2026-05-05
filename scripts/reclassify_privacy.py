#!/usr/bin/env python3
"""
Re-classify only the privacy_violation dimension with a tighter prompt.
Only marks 1 if the CHILD's privacy is being violated, not adults.
"""
import sys
import os
import json
import csv
import time

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-mini"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_discovery')
INPUT_FILE = os.path.join(OUTPUT_DIR, 'classification_6dim_sample.csv')

BATCH_SIZE = 40  # Larger batch since we only need one field
DELAY = 0.3

SYSTEM_PROMPT = """You are an expert in children's media and child privacy rights (UNCRC Article 16).

You will classify YouTube video titles from kidfluencer/family vlog channels on ONE dimension: whether the title suggests a CHILD's privacy is being violated.

"privacy_violation": 
1 = The title clearly suggests sharing a CHILD's (under 16) private, embarrassing, or intimate moments that the child would likely not want publicly shared when they grow up. 

MUST involve the CHILD specifically. Examples that ARE privacy violations:
- Potty training footage of the child
- Child having a tantrum/meltdown filmed for content
- Child's medical procedures or health conditions shared publicly
- Child caught doing something embarrassing ("CAUGHT MY 5 YEAR OLD DOING THIS")
- Child's body changes, puberty discussions
- Child crying/emotional breakdown filmed and uploaded
- Bathing/naked moments of children
- Child being disciplined/punished on camera

Examples that are NOT privacy violations (even if they seem private):
- Adult/parent pranks on each other ("I PUT PERIOD BLOOD ON MY HUSBAND'S FACE") - this is about ADULTS
- Parent's pregnancy test - this is about the ADULT
- Parent's medical appointment - this is about the ADULT
- Adult relationship drama - this is about ADULTS
- General family vacation/outing content
- Child doing a fun activity they'd likely enjoy sharing

KEY RULE: If the title is about ADULT content (parent pranks, parent health, parent relationships), mark 0 even if it seems private. We ONLY care about the CHILD's privacy.

For each title, output ONLY a JSON array of integers (1 or 0), one per title, in the same order.
"""


def classify_batch(titles):
    """Classify a batch of titles."""
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    user_msg = f"Classify these {len(titles)} titles for child privacy violation. Return ONLY a JSON array of {len(titles)} integers (1 or 0).\n\n{titles_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=2000,
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
                results.append(0)
            results = results[:len(titles)]
        return results
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        time.sleep(3)
        return [0] * len(titles)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Re-classifying privacy_violation (child-only)")
    
    # Load existing classification
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} videos")
    
    titles = df['title'].fillna('').tolist()
    total_batches = (len(titles) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches: {total_batches}")
    
    all_results = []
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(titles))
        batch_titles = titles[batch_start:batch_end]
        
        results = classify_batch(batch_titles)
        all_results.extend(results)
        
        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            print(f"  Batch {batch_num+1}/{total_batches} ({batch_end/len(titles)*100:.1f}%)", flush=True)
        
        time.sleep(DELAY)
    
    # Update the CSV
    old_count = (df['privacy_violation'] == 1).sum()
    df['privacy_violation'] = all_results
    new_count = (df['privacy_violation'] == 1).sum()
    
    df.to_csv(INPUT_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"DONE: privacy_violation updated")
    print(f"  Old count: {old_count} ({old_count/len(df)*100:.1f}%)")
    print(f"  New count: {new_count} ({new_count/len(df)*100:.1f}%)")
    print(f"{'='*60}")
    
    # Show samples
    priv_df = df[df['privacy_violation'] == 1]
    print(f"\n=== SAMPLE: privacy_violation=1 (child-specific) ===")
    for _, row in priv_df.head(15).iterrows():
        print(f"  [{row['channel_short_name']}] {row['title']}")
    
    # Show what was removed (was 1, now 0)
    if old_count > new_count:
        print(f"\n=== REMOVED (was 1, now 0): ===")
        # We can't easily show these without saving old values, but show new stats
        
    # View boost
    df['viewCount'] = pd.to_numeric(df['viewCount'], errors='coerce')
    g1 = df[df['privacy_violation']==1]['viewCount'].median()
    g0 = df[df['privacy_violation']==0]['viewCount'].median()
    if g0 and g0 > 0:
        boost = (g1 - g0) / g0 * 100
        print(f"\n  View boost: YES={g1:,.0f} | NO={g0:,.0f} | boost={boost:+.1f}%")


if __name__ == '__main__':
    main()
