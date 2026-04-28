"""
Analyze content drift in YouTube family vlog channels.
1. Embed all video titles using sentence-transformers
2. Define an "exploitation direction" using anchor terms
3. Compute drift scores over time
4. Identify viral videos and test if they precede content shifts
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Step 1: Define exploitation direction anchors ===
# Positive anchors: titles that represent exploitative/clickbait family content
exploitation_anchors = [
    "TELLING MY KIDS THEY'RE ADOPTED PRANK",
    "MAKING MY DAUGHTER CRY ON CAMERA",
    "MY CHILD'S WORST DAY EVER emotional breakdown",
    "PUNISHING MY KIDS FOR 24 HOURS CHALLENGE",
    "MY 5 YEAR OLD GETS HER FIRST BOYFRIEND",
    "LEAVING MY KIDS ALONE FOR 24 HOURS",
    "EMBARRASSING MY DAUGHTER IN FRONT OF HER CRUSH",
    "MY BABY IS SICK RUSHING TO THE HOSPITAL EMERGENCY",
    "SURPRISING MY KID WITH $10000 SHOPPING SPREE",
    "DESTROYING MY KIDS TOYS PRANK GONE WRONG",
    "WE'RE GETTING A DIVORCE telling the kids",
    "MY DAUGHTER RAN AWAY FROM HOME",
    "CAUGHT MY KID DOING THIS ON CAMERA",
    "EXTREME PUNISHMENT FOR LYING challenge",
    "MAKING MY KIDS EAT ONLY ONE FOOD FOR 24 HOURS",
]

# Negative anchors: titles that represent healthy/educational family content
healthy_anchors = [
    "Fun family day at the park together",
    "Learning colors and shapes with kids educational",
    "Cooking healthy meals together as a family",
    "Reading bedtime stories to the children",
    "Family game night playing board games",
    "Teaching my kids to ride a bike",
    "Our family garden project planting flowers",
    "Arts and crafts time with the kids",
    "Family hiking adventure in nature",
    "Building a birdhouse together DIY project",
    "Learning about animals at the zoo",
    "Family music time singing and dancing",
    "Helping kids with homework study tips",
    "Family volunteering at the food bank",
    "Teaching kids about kindness and sharing",
]

print("Computing anchor embeddings...")
exploit_embs = model.encode(exploitation_anchors)
healthy_embs = model.encode(healthy_anchors)

# Direction vector: mean(exploitation) - mean(healthy)
exploit_center = exploit_embs.mean(axis=0)
healthy_center = healthy_embs.mean(axis=0)
direction_vector = exploit_center - healthy_center
direction_vector = direction_vector / np.linalg.norm(direction_vector)

print(f"Direction vector computed (dim={len(direction_vector)})")

# === Step 2: Load and process each channel ===
DATA_DIR = "/home/ubuntu/pilot/data"
OUTPUT_DIR = "/home/ubuntu/pilot/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

channels = {
    'acefamily': 'The ACE Family (Family)',
    'ryansworld': "Ryan's World (Family)",
    'familyfunpack': 'Family Fun Pack (Family)',
    'bratayley': 'Bratayley (Family)',
    'caseyneistat': 'Casey Neistat (Control)',
    'markwiens': 'Mark Wiens (Control)',
}

all_results = {}

for channel_key, channel_label in channels.items():
    filepath = os.path.join(DATA_DIR, f"{channel_key}.json")
    if not os.path.exists(filepath):
        print(f"Skipping {channel_key}: no data")
        continue
    
    with open(filepath) as f:
        videos = json.load(f)
    
    # Filter out videos without titles or dates
    valid_videos = [v for v in videos if v.get('title') and v.get('view_count') is not None]
    
    if not valid_videos:
        print(f"Skipping {channel_key}: no valid videos")
        continue
    
    print(f"\nProcessing {channel_key}: {len(valid_videos)} videos")
    
    # Embed all titles
    titles = [v['title'] for v in valid_videos]
    embeddings = model.encode(titles, show_progress_bar=True, batch_size=64)
    
    # Compute exploitation drift score for each video
    drift_scores = np.dot(embeddings, direction_vector)
    
    # Build dataframe
    df = pd.DataFrame(valid_videos)
    df['drift_score'] = drift_scores
    
    # Parse dates - playlist_index gives us chronological order (newest first)
    # Reverse to get oldest first
    if 'upload_date' in df.columns:
        df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d', errors='coerce')
    
    # Sort by playlist_index descending (oldest first)
    df = df.sort_values('playlist_index', ascending=False).reset_index(drop=True)
    df['video_order'] = range(len(df))
    
    # Compute rolling average of drift score (window=20)
    df['drift_rolling'] = df['drift_score'].rolling(window=20, min_periods=5).mean()
    
    # Identify viral videos (view_count > 2x channel median)
    median_views = df['view_count'].median()
    df['is_viral'] = df['view_count'] > (2 * median_views)
    df['viral_ratio'] = df['view_count'] / max(median_views, 1)
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"{channel_key}_drift.csv")
    df.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")
    print(f"  Mean drift score: {df['drift_score'].mean():.4f}")
    print(f"  Drift score std: {df['drift_score'].std():.4f}")
    print(f"  Median views: {median_views:,.0f}")
    print(f"  Viral videos (>2x median): {df['is_viral'].sum()}")
    
    all_results[channel_key] = {
        'label': channel_label,
        'n_videos': len(df),
        'mean_drift': df['drift_score'].mean(),
        'std_drift': df['drift_score'].std(),
        'median_views': median_views,
        'n_viral': int(df['is_viral'].sum()),
    }

# === Step 3: Summary comparison ===
print("\n" + "="*60)
print("SUMMARY: Mean Exploitation Drift Score by Channel")
print("="*60)
print(f"{'Channel':<30} {'Type':<10} {'N Videos':<10} {'Mean Drift':<12} {'Std':<10}")
print("-"*72)
for key, info in sorted(all_results.items(), key=lambda x: x[1]['mean_drift'], reverse=True):
    ch_type = "Family" if "Family" in info['label'] else "Control"
    print(f"{info['label']:<30} {ch_type:<10} {info['n_videos']:<10} {info['mean_drift']:<12.4f} {info['std_drift']:<10.4f}")

# Save summary
with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nDone! Results saved to /home/ubuntu/pilot/results/")
