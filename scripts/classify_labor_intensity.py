"""
LLM-based classification of child labor intensity in kidfluencer videos.
Uses gpt-4.1-nano for speed (41K videos in batches of 50).
"""
import pandas as pd
import numpy as np
import json, os, sys, time
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("Loading data...", flush=True)
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
print(f"Total videos: {len(df)}", flush=True)

client = OpenAI()

# Check if partial results exist
partial_path = 'analysis_discovery/labor_classification_partial.csv'
start_from = 0
results = []
if os.path.exists(partial_path):
    partial = pd.read_csv(partial_path)
    results = partial.to_dict('records')
    start_from = len(results)
    print(f"Resuming from {start_from} (partial results found)", flush=True)

batch_size = 50
n_batches = (len(df) + batch_size - 1) // batch_size
start_batch = start_from // batch_size

print(f"Classifying {len(df)} videos in {n_batches} batches (batch_size={batch_size})...", flush=True)

for batch_idx in range(start_batch, n_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end]
    
    titles_text = "\n".join([f"{i+1}. [{row['channel_short_name']}] {row['title']}" 
                            for i, (_, row) in enumerate(batch.iterrows())])
    
    prompt = f"""Classify each video title from kidfluencer channels:
1. labor_type: "performative" (child working: challenges, roleplay, unboxing, pranks, games for content, toy reviews, singing/dancing for camera) | "organic" (natural: vacations, birthdays, routines, milestones) | "ambiguous" | "no_child"
2. emotional_exploitation: 1 if child distress/conflict/fear/crying/punishment/medical/embarrassment, else 0
3. content_format: challenge|roleplay|unboxing|prank|game|music_dance|toy_play|vlog|storytime|mukbang|tutorial|reaction|drama|milestone|travel|medical|announcement|other

Titles:
{titles_text}

Return JSON: {{"items": [{{labor_type, emotional_exploitation, content_format}}...]}} exactly {end-start} items."""

    try:
        response = client.chat.completions.create(
            model='gpt-4.1-nano',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        items = result.get('items', [])
        if not items:
            for v in result.values():
                if isinstance(v, list):
                    items = v
                    break
        
        while len(items) < (end - start):
            items.append({'labor_type': 'ambiguous', 'emotional_exploitation': 0, 'content_format': 'other'})
        items = items[:end-start]
        
        for item in items:
            results.append({
                'labor_type': item.get('labor_type', 'ambiguous'),
                'emotional_exploitation': int(item.get('emotional_exploitation', 0)),
                'content_format': item.get('content_format', 'other')
            })
            
    except Exception as e:
        print(f"  ERROR batch {batch_idx}: {e}", flush=True)
        for _ in range(end - start):
            results.append({'labor_type': 'ambiguous', 'emotional_exploitation': 0, 'content_format': 'other'})
    
    if (batch_idx + 1) % 20 == 0:
        print(f"  {batch_idx+1}/{n_batches} batches ({end}/{len(df)} videos)", flush=True)
        # Save checkpoint
        pd.DataFrame(results).to_csv(partial_path, index=False)

print(f"\nClassification complete: {len(results)} videos", flush=True)

# Save final results
results_df = pd.DataFrame(results)
df_combined = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
df_combined.to_csv('analysis_discovery/labor_classification_full.csv', index=False)

# Summary
print(f"\n{'='*60}", flush=True)
print("LABOR TYPE DISTRIBUTION:", flush=True)
print(results_df['labor_type'].value_counts().to_string(), flush=True)
print(f"\nEMOTIONAL EXPLOITATION:", flush=True)
print(f"  Yes: {results_df['emotional_exploitation'].sum()} ({results_df['emotional_exploitation'].mean()*100:.1f}%)", flush=True)
print(f"\nCONTENT FORMAT:", flush=True)
print(results_df['content_format'].value_counts().head(15).to_string(), flush=True)
print(f"\nCROSS-TAB: Labor Type x Emotional Exploitation", flush=True)
print(pd.crosstab(results_df['labor_type'], results_df['emotional_exploitation'], margins=True), flush=True)
print(f"\nSaved: analysis_discovery/labor_classification_full.csv", flush=True)
