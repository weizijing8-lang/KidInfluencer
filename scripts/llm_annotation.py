"""
LLM Annotation for Robustness Check
=====================================
Use GPT-4.1-mini to annotate a stratified sample of video titles
for child exploitation indicators.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
RESULTS_V2 = DATA_DIR / "results_v2"
RESULTS_V3 = DATA_DIR / "results_v3"
RESULTS_V3.mkdir(parents=True, exist_ok=True)

client = OpenAI()

ANNOTATION_PROMPT = """You are an expert in child welfare and digital media ethics. 
Rate each YouTube video title for child exploitation indicators.

Score 0-5:
0 = Healthy/educational or unrelated to children
1 = Benign family content, mildly attention-seeking
2 = Mild exploitation (clickbait using children, not harmful)
3 = Moderate exploitation (privacy concerns, mild emotional manipulation)
4 = Significant exploitation (deliberate distress, dangerous situations)
5 = Severe exploitation (deliberate harm, extreme manipulation)

Dimensions: D1=emotional distress, D2=privacy violation, D3=dangerous/inappropriate, D4=clickbait exploitation, D5=commodification, D0=none

Respond ONLY with a JSON array. No other text."""


def sample_titles():
    """Sample 500 family + 200 adult titles."""
    print("Loading data...", flush=True)
    df = pd.read_csv(RESULTS_V2 / "full_results_v2.csv")
    
    family = df[df['channel_category'] == 'family']
    adult = df[df['channel_category'] == 'adult']
    
    # 500 family (stratified ~20 per channel)
    family_sample = family.groupby('channel_short_name').apply(
        lambda x: x.sample(n=min(20, len(x)), random_state=42),
        include_groups=False
    ).reset_index(drop=True)
    if len(family_sample) > 500:
        family_sample = family_sample.sample(n=500, random_state=42)
    
    # 200 adult (random)
    adult_sample = adult.sample(n=min(200, len(adult)), random_state=42)
    
    sample = pd.concat([family_sample, adult_sample], ignore_index=True)
    print(f"Sampled {len(sample)} titles: {len(family_sample)} family + {len(adult_sample)} adult", flush=True)
    return sample


def annotate_one_batch(batch_items):
    """Annotate a single batch of titles."""
    batch_text = "\n".join([
        f"{j+1}. [{item['category']}] {item['channel']}: \"{item['title']}\""
        for j, item in enumerate(batch_items)
    ])
    
    user_msg = f"""Rate these {len(batch_items)} titles. Return JSON array only.

{batch_text}

Format: [{{"score":0,"dims":["D0"],"conf":0.9}}, ...]"""
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": ANNOTATION_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.1,
        max_tokens=2000,
        timeout=30,
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    
    return json.loads(content)


def main():
    print("=" * 60, flush=True)
    print("LLM ANNOTATION FOR ROBUSTNESS CHECK", flush=True)
    print("=" * 60, flush=True)
    
    sample = sample_titles()
    
    # Prepare items
    items = []
    for _, row in sample.iterrows():
        items.append({
            'id': row.get('id', ''),
            'title': str(row['title'])[:200],  # Truncate long titles
            'channel': row['channel_short_name'],
            'category': row['channel_category'],
            'exploit_score_v2': row['exploit_score_v2'],
        })
    
    # Annotate in batches of 10
    BATCH_SIZE = 10
    all_results = []
    total = len(items)
    
    print(f"\nAnnotating {total} titles in batches of {BATCH_SIZE}...", flush=True)
    
    for i in range(0, total, BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        
        try:
            annotations = annotate_one_batch(batch)
            
            for j, item in enumerate(batch):
                if j < len(annotations):
                    ann = annotations[j]
                    all_results.append({
                        'id': item['id'],
                        'title': item['title'],
                        'channel': item['channel'],
                        'category': item['category'],
                        'exploit_score_v2': item['exploit_score_v2'],
                        'llm_score': ann.get('score', ann.get('exploitation_score', -1)),
                        'llm_dims': ','.join(ann.get('dims', ann.get('dimensions', []))),
                        'llm_conf': ann.get('conf', ann.get('confidence', 0)),
                    })
                else:
                    all_results.append({
                        'id': item['id'], 'title': item['title'],
                        'channel': item['channel'], 'category': item['category'],
                        'exploit_score_v2': item['exploit_score_v2'],
                        'llm_score': -1, 'llm_dims': 'MISSING', 'llm_conf': 0,
                    })
        except Exception as e:
            print(f"  Error batch {i//BATCH_SIZE}: {e}", flush=True)
            for item in batch:
                all_results.append({
                    'id': item['id'], 'title': item['title'],
                    'channel': item['channel'], 'category': item['category'],
                    'exploit_score_v2': item['exploit_score_v2'],
                    'llm_score': -1, 'llm_dims': 'ERROR', 'llm_conf': 0,
                })
        
        done = min(i + BATCH_SIZE, total)
        if done % 50 == 0 or done == total:
            print(f"  Progress: {done}/{total}", flush=True)
        
        # Save intermediate results every 100
        if done % 100 == 0:
            pd.DataFrame(all_results).to_csv(RESULTS_V3 / "llm_annotations_partial.csv", index=False)
        
        time.sleep(0.3)
    
    # Save final
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_V3 / "llm_annotations.csv", index=False)
    
    # Analysis
    valid = results_df[results_df['llm_score'] >= 0]
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {len(valid)}/{len(results_df)} valid annotations", flush=True)
    
    fam = valid[valid['category'] == 'family']
    adu = valid[valid['category'] == 'adult']
    print(f"  Family mean LLM score: {fam['llm_score'].mean():.2f}", flush=True)
    print(f"  Adult mean LLM score:  {adu['llm_score'].mean():.2f}", flush=True)
    
    from scipy import stats
    r, p = stats.pearsonr(valid['exploit_score_v2'], valid['llm_score'])
    print(f"  Correlation (embedding vs LLM): r={r:.4f}, p={'<0.001' if p<0.001 else f'{p:.4f}'}", flush=True)
    
    # Score distribution
    print(f"\n  Score distribution:", flush=True)
    for s in range(6):
        n = (valid['llm_score'] == s).sum()
        print(f"    Score {s}: {n} ({100*n/len(valid):.1f}%)", flush=True)
    
    print(f"\nDone! Saved to {RESULTS_V3 / 'llm_annotations.csv'}", flush=True)


if __name__ == "__main__":
    main()
