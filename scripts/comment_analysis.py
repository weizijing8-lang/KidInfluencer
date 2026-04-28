"""
Comment Inappropriateness Detection
=====================================
Classify comments on family channel videos for:
- normal: Regular fan/viewer comment
- inappropriate: Sexualization, objectification, predatory behavior
- concern: Expressing worry about child welfare
- timestamp: Pointing to specific moments (predatory signal)

Use GPT-4.1-mini to classify in batches.
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
COMMENTS_DIR = DATA_DIR / "comments"
RESULTS_V4 = DATA_DIR / "results_v4"
RESULTS_V4.mkdir(parents=True, exist_ok=True)

client = OpenAI()

CLASSIFICATION_PROMPT = """You are a child safety expert analyzing YouTube comments on family/children's channels.

Classify each comment into ONE category:
- "normal": Regular fan comment, positive reaction, neutral observation
- "inappropriate": Sexualization, objectification, creepy/predatory language, adult content directed at minors
- "concern": Expressing worry about child welfare, criticism of parents exploiting children
- "timestamp": References specific time points in video (e.g., "2:34", "at 3 minutes") — often a predatory signal on children's content
- "toxic": General hate, bullying, or harassment (not specifically about child exploitation)

Also rate confidence 0-1.

Respond ONLY with a JSON array. No other text."""


def classify_batch(comments):
    """Classify a batch of comments."""
    batch_text = "\n".join([
        f'{i+1}. "{c[:150]}"' for i, c in enumerate(comments)
    ])
    
    user_msg = f"""Classify these {len(comments)} comments from a family YouTube channel:

{batch_text}

Return JSON array: [{{"cat":"normal","conf":0.9}}, ...]"""
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
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
    print("COMMENT INAPPROPRIATENESS DETECTION", flush=True)
    print("=" * 60, flush=True)
    
    # Load comments
    comments_df = pd.read_csv(COMMENTS_DIR / "sampled_comments.csv")
    print(f"Loaded {len(comments_df):,} comments from {comments_df['video_id'].nunique()} videos", flush=True)
    print(f"  High exploitation videos: {(comments_df['sample_type']=='high').sum()} comments", flush=True)
    print(f"  Low exploitation videos:  {(comments_df['sample_type']=='low').sum()} comments", flush=True)
    
    # Classify in batches of 20
    BATCH_SIZE = 20
    all_results = []
    total = len(comments_df)
    
    print(f"\nClassifying {total} comments in batches of {BATCH_SIZE}...", flush=True)
    
    for i in range(0, total, BATCH_SIZE):
        batch = comments_df.iloc[i:i+BATCH_SIZE]
        comment_texts = batch['comment_text'].fillna('').tolist()
        
        try:
            classifications = classify_batch(comment_texts)
            
            for j, (_, row) in enumerate(batch.iterrows()):
                if j < len(classifications):
                    cls = classifications[j]
                    all_results.append({
                        'video_id': row['video_id'],
                        'channel': row['channel'],
                        'exploit_score_v2': row['exploit_score_v2'],
                        'sample_type': row['sample_type'],
                        'comment_text': row['comment_text'],
                        'comment_likes': row['comment_likes'],
                        'category': cls.get('cat', 'unknown'),
                        'confidence': cls.get('conf', 0),
                    })
                else:
                    all_results.append({
                        'video_id': row['video_id'],
                        'channel': row['channel'],
                        'exploit_score_v2': row['exploit_score_v2'],
                        'sample_type': row['sample_type'],
                        'comment_text': row['comment_text'],
                        'comment_likes': row['comment_likes'],
                        'category': 'error',
                        'confidence': 0,
                    })
        except Exception as e:
            for _, row in batch.iterrows():
                all_results.append({
                    'video_id': row['video_id'],
                    'channel': row['channel'],
                    'exploit_score_v2': row['exploit_score_v2'],
                    'sample_type': row['sample_type'],
                    'comment_text': row['comment_text'],
                    'comment_likes': row['comment_likes'],
                    'category': 'error',
                    'confidence': 0,
                })
            if i % 200 == 0:
                print(f"  Error at batch {i//BATCH_SIZE}: {e}", flush=True)
        
        done = min(i + BATCH_SIZE, total)
        if done % 500 == 0 or done == total:
            print(f"  Progress: {done}/{total}", flush=True)
        
        # Save intermediate
        if done % 2000 == 0:
            pd.DataFrame(all_results).to_csv(RESULTS_V4 / "comment_classifications_partial.csv", index=False)
        
        time.sleep(0.2)
    
    # Save final
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_V4 / "comment_classifications.csv", index=False)
    
    # Analysis
    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    
    valid = results_df[results_df['category'] != 'error']
    print(f"Valid classifications: {len(valid)}/{len(results_df)}", flush=True)
    
    # Overall distribution
    print(f"\nOverall category distribution:", flush=True)
    for cat, count in valid['category'].value_counts().items():
        print(f"  {cat:15s}: {count:5d} ({100*count/len(valid):.1f}%)", flush=True)
    
    # High vs Low exploitation comparison
    print(f"\nHigh vs Low exploitation videos:", flush=True)
    for cat in ['inappropriate', 'concern', 'timestamp', 'toxic']:
        high_rate = (valid[valid['sample_type']=='high']['category'] == cat).mean()
        low_rate = (valid[valid['sample_type']=='low']['category'] == cat).mean()
        ratio = high_rate / low_rate if low_rate > 0 else float('inf')
        print(f"  {cat:15s}: high={100*high_rate:.2f}% vs low={100*low_rate:.2f}% (ratio={ratio:.2f}x)", flush=True)
    
    # By channel
    print(f"\nInappropriate comment rate by channel:", flush=True)
    ch_rates = valid.groupby('channel').apply(
        lambda x: (x['category'] == 'inappropriate').mean()
    ).sort_values(ascending=False)
    for ch, rate in ch_rates.items():
        print(f"  {ch:25s}: {100*rate:.2f}%", flush=True)
    
    print(f"\nDone! Saved to {RESULTS_V4 / 'comment_classifications.csv'}", flush=True)


if __name__ == "__main__":
    main()
