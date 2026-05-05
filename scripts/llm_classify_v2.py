#!/usr/bin/env python3
"""
LLM Classification of 6 Exploitation Dimensions
=================================================
Classifies each video title into 6 binary dimensions using GPT-4.1-mini.
Processes in batches of 10 titles per API call to reduce total calls.
Saves incrementally to avoid data loss.

Input: data/stratified_sample_v2.csv
Output: data/llm_classifications_v2.csv
"""
import os
import sys
import csv
import json
import time
import pandas as pd
from datetime import datetime
from openai import OpenAI

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

client = OpenAI()
MODEL = "gpt-4.1-mini"

INPUT_CSV = '/home/ubuntu/KidInfluencer/data/stratified_sample_v2.csv'
OUTPUT_CSV = '/home/ubuntu/KidInfluencer/data/llm_classifications_v2.csv'
LOG_FILE = '/home/ubuntu/KidInfluencer/data/llm_classify_log.txt'
BATCH_SIZE = 25
DELAY = 0.3  # seconds between API calls

SYSTEM_PROMPT = """You are an expert in child welfare and media ethics. You analyze YouTube video titles from children's/family channels to detect potential exploitation dimensions.

For each video title, classify it on these 6 binary dimensions (1=present, 0=absent):

1. performative_labor: Child is performing structured labor for the camera (challenges, dares, pranks, skits, competitions, talent shows). NOT just appearing in a vlog.
2. emotional_bait: Title uses emotional manipulation or clickbait involving children's emotions (crying, screaming, heartbreak, surprise reactions, "emotional" moments used as hooks).
3. narrative_conflict: Title frames interpersonal conflict as entertainment (vs battles, fights, arguments, revenge, "caught", betrayal between family members).
4. challenge_format: Video uses a structured challenge format (24 hours, last to leave, extreme, impossible, dare wheel, would you rather with consequences).
5. commercial_content: Video is primarily product placement, unboxing, sponsored content, or shopping sprees.
6. privacy_violation: Title suggests exposing private/intimate moments of children (secret cameras, hidden recordings, embarrassing moments, bathroom/bedroom content, medical procedures).

Be conservative: only mark 1 if the title clearly indicates that dimension. Ambiguous cases should be 0."""

USER_PROMPT_TEMPLATE = """Classify these video titles. Return ONLY a JSON array with one object per title, each containing the 6 dimension scores (0 or 1).

Titles:
{titles}

Return format: [{{"performative_labor":0,"emotional_bait":0,"narrative_conflict":0,"challenge_format":0,"commercial_content":0,"privacy_violation":0}}, ...]"""


def classify_batch(titles):
    """Classify a batch of titles using GPT-4.1-mini."""
    numbered_titles = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    prompt = USER_PROMPT_TEMPLATE.format(titles=numbered_titles)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return None
        content = content.strip()
        parsed = json.loads(content)
        
        # Handle different response formats
        if isinstance(parsed, list):
            results = parsed
        elif isinstance(parsed, dict):
            # Sometimes wrapped in a key like "results" or "classifications"
            for key in ['results', 'classifications', 'data', 'titles']:
                if key in parsed and isinstance(parsed[key], list):
                    results = parsed[key]
                    break
            else:
                # Single result dict
                results = [parsed]
        else:
            return None
        
        # Validate
        dims = ['performative_labor', 'emotional_bait', 'narrative_conflict', 
                'challenge_format', 'commercial_content', 'privacy_violation']
        validated = []
        for r in results:
            row = {}
            for d in dims:
                val = r.get(d, 0)
                row[d] = 1 if val in [1, True, '1', 'true', 'True'] else 0
            validated.append(row)
        
        return validated
        
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None


def main():
    # Load sample
    df = pd.read_csv(INPUT_CSV)
    print(f"[{datetime.now()}] Loaded {len(df)} videos for classification")
    
    # Check for existing progress
    start_idx = 0
    existing_results = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        start_idx = len(existing)
        print(f"  Resuming from row {start_idx}")
    
    # Prepare output file
    dims = ['performative_labor', 'emotional_bait', 'narrative_conflict', 
            'challenge_format', 'commercial_content', 'privacy_violation']
    fieldnames = ['id', 'title', 'channel_short_name', 'viewCount'] + dims
    
    if start_idx == 0:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Process in batches
    total_batches = (len(df) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    processed = 0
    errors = 0
    
    for batch_start in range(start_idx, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        titles = batch_df['title'].tolist()
        
        # Classify
        results = classify_batch(titles)
        
        if results is None or len(results) != len(titles):
            # Retry once
            time.sleep(2)
            results = classify_batch(titles)
        
        if results is None or len(results) != len(titles):
            # Fall back to individual classification
            results = []
            for t in titles:
                r = classify_batch([t])
                if r and len(r) == 1:
                    results.append(r[0])
                else:
                    results.append({d: 0 for d in dims})
                    errors += 1
                time.sleep(DELAY)
        
        # Write results
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for i, (_, row) in enumerate(batch_df.iterrows()):
                if i < len(results):
                    out_row = {
                        'id': row['id'],
                        'title': row['title'],
                        'channel_short_name': row['channel_short_name'],
                        'viewCount': row['viewCount'],
                    }
                    out_row.update(results[i])
                    writer.writerow(out_row)
        
        processed += len(titles)
        batch_num = (batch_start - start_idx) // BATCH_SIZE + 1
        
        if batch_num % 20 == 0 or batch_num <= 3:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_num}/{total_batches}: "
                  f"{processed + start_idx}/{len(df)} done, {errors} errors")
        
        time.sleep(DELAY)
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total classified: {processed + start_idx}")
    print(f"Errors: {errors}")
    
    # Quick stats
    result_df = pd.read_csv(OUTPUT_CSV)
    print(f"\nDimension prevalence:")
    for d in dims:
        pct = result_df[d].mean() * 100
        print(f"  {d}: {pct:.1f}%")


if __name__ == '__main__':
    main()
