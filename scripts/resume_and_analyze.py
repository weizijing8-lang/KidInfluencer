"""
Resume LLM classification for remaining videos, then do full analysis.
Since we already have 27K classified, resume from there.
"""
import pandas as pd
import numpy as np
import json, os, sys, time
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

print("Loading data...", flush=True)
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
partial = pd.read_csv('analysis_discovery/labor_classification_partial.csv')
print(f"Total: {len(df)}, Already classified: {len(partial)}", flush=True)

if len(partial) >= len(df):
    print("Already complete!", flush=True)
else:
    client = OpenAI()
    results = partial.to_dict('records')
    start_from = len(results)
    batch_size = 50
    n_batches = (len(df) + batch_size - 1) // batch_size
    start_batch = start_from // batch_size
    
    print(f"Resuming from batch {start_batch} ({start_from} videos)...", flush=True)
    
    for batch_idx in range(start_batch, n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        
        titles_text = "\n".join([f"{i+1}. [{row['channel_short_name']}] {row['title']}" 
                                for i, (_, row) in enumerate(batch.iterrows())])
        
        prompt = f"""Classify each video title. Return ONLY valid JSON.
1. labor_type: MUST be exactly one of: "performative", "organic", "ambiguous", "no_child"
2. emotional_exploitation: MUST be 0 or 1
3. content_format: MUST be one of: "challenge", "roleplay", "unboxing", "prank", "game", "music_dance", "toy_play", "vlog", "storytime", "mukbang", "tutorial", "reaction", "drama", "milestone", "travel", "medical", "announcement", "other"

Rules:
- "performative": child does challenges, roleplay, unboxing, pranks, games, toy reviews, singing/dancing FOR the camera
- "organic": child in natural family life - vacations, birthdays, routines, school
- emotional_exploitation=1: title suggests child distress, conflict, fear, crying, punishment, medical, embarrassment

Titles:
{titles_text}

Return: {{"items": [...]}} with exactly {end-start} items."""

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
            pd.DataFrame(results).to_csv('analysis_discovery/labor_classification_partial.csv', index=False)
    
    # Save final
    pd.DataFrame(results).to_csv('analysis_discovery/labor_classification_partial.csv', index=False)
    print(f"Classification complete: {len(results)} videos", flush=True)

# Now clean and save the full dataset
print("\nCleaning and saving full results...", flush=True)
partial = pd.read_csv('analysis_discovery/labor_classification_partial.csv')

# Clean labor_type
valid_types = {'performative', 'organic', 'ambiguous', 'no_child'}
labor_clean = partial['labor_type'].str.strip().str.lower()
labor_clean = labor_clean.map(lambda x: 'organic' if 'organ' in str(x) else x)
labor_clean = labor_clean.map(lambda x: 'performative' if 'perform' in str(x) else x)
labor_clean = labor_clean.map(lambda x: 'ambiguous' if 'ambigu' in str(x) else x)
labor_clean = labor_clean.map(lambda x: 'no_child' if 'no_child' in str(x) else x)
labor_clean = labor_clean.map(lambda x: x if x in valid_types else 'ambiguous')
partial['labor_type'] = labor_clean

# Clean content_format
valid_formats = {'challenge', 'roleplay', 'unboxing', 'prank', 'game', 'music_dance', 
                 'toy_play', 'vlog', 'storytime', 'mukbang', 'tutorial', 'reaction', 
                 'drama', 'milestone', 'travel', 'medical', 'announcement', 'other'}
partial['content_format'] = partial['content_format'].str.strip().str.lower()
partial['content_format'] = partial['content_format'].map(lambda x: x if x in valid_formats else 'other')

# Combine with original data (use only as many as we have labels for)
n = min(len(df), len(partial))
df_combined = pd.concat([df.iloc[:n].reset_index(drop=True), partial.iloc[:n].reset_index(drop=True)], axis=1)
df_combined.to_csv('analysis_discovery/labor_classification_full.csv', index=False)

print(f"\nFinal dataset: {len(df_combined)} videos", flush=True)
print(f"\nLABOR TYPE:", flush=True)
print(df_combined['labor_type'].value_counts().to_string(), flush=True)
print(f"\nEMOTIONAL EXPLOITATION: {df_combined['emotional_exploitation'].sum()} ({df_combined['emotional_exploitation'].mean()*100:.1f}%)", flush=True)
print(f"\nCONTENT FORMAT:", flush=True)
print(df_combined['content_format'].value_counts().head(12).to_string(), flush=True)
print("\nDone!", flush=True)
