"""
Extract de-channelized content features from video titles.
Two-stage approach:
1. Rule-based features (fast, covers all 41K videos)
2. LLM-based semantic labeling (on batches, for validation + additional features)

The goal is to produce feature vectors that capture CONTENT STRATEGY
rather than channel identity, so that clustering reveals cross-channel patterns.
"""
import pandas as pd
import numpy as np
import re, json, os, time
from openai import OpenAI

# Load data
print("Loading data...")
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
print(f"Total videos: {len(df)}, Channels: {df['channel_short_name'].nunique()}")

# ============================================================
# STAGE 1: Rule-based feature extraction (all 41K videos)
# ============================================================
print("\n[Stage 1] Extracting rule-based content features...")

# Content type indicators (binary)
def extract_features(title):
    t = str(title).lower()
    features = {}
    
    # Content format
    features['is_challenge'] = int(bool(re.search(r'challenge|dare|bet|vs\.?|versus', t)))
    features['is_unboxing'] = int(bool(re.search(r'unbox|opening|haul|surprise\s*(egg|box|toy)', t)))
    features['is_prank'] = int(bool(re.search(r'prank|trick|fool|scare', t)))
    features['is_storytime'] = int(bool(re.search(r'story\s*time|storytime|what happened|the truth', t)))
    features['is_vlog'] = int(bool(re.search(r'vlog|day in|daily|routine|week\s*\d|wk\s*\d', t)))
    features['is_tutorial'] = int(bool(re.search(r'how to|tutorial|diy|recipe|make|craft', t)))
    features['is_reaction'] = int(bool(re.search(r'react|reaction|reacting|watching', t)))
    features['is_qa'] = int(bool(re.search(r'q\s*&\s*a|q&a|answering|ask me|faq', t)))
    features['is_review'] = int(bool(re.search(r'review|rating|rank|tier\s*list|testing', t)))
    features['is_mukbang'] = int(bool(re.search(r'mukbang|eating|food|cook|bak(e|ing)|kitchen', t)))
    features['is_game'] = int(bool(re.search(r'game|play|playing|minecraft|roblox|fortnite|hide.*seek|tag', t)))
    features['is_roleplay'] = int(bool(re.search(r'pretend|roleplay|role\s*play|dress\s*up|superhero|princess|spy', t)))
    features['is_music'] = int(bool(re.search(r'song|music|sing|dance|nursery|rhyme|lullaby', t)))
    features['is_toy'] = int(bool(re.search(r'toy|doll|barbie|lego|slime|playdoh|play-doh|nerf', t)))
    features['is_shorts'] = int(bool(re.search(r'#shorts|short|clip', t)))
    
    # Emotional manipulation signals
    features['has_clickbait_emotion'] = int(bool(re.search(
        r'shocking|unbelievable|you won\'t believe|insane|crazy|epic|worst|best ever|gone wrong|emotional', t)))
    features['has_urgency'] = int(bool(re.search(
        r'last|final|never again|goodbye|the end|emergency|urgent|breaking', t)))
    features['has_mystery'] = int(bool(re.search(
        r'mystery|secret|hidden|reveal|exposed|truth|caught|found', t)))
    features['has_conflict'] = int(bool(re.search(
        r'fight|argue|angry|mad|cry|crying|scream|yell|broke up|divorce|grounded|punish', t)))
    features['has_medical'] = int(bool(re.search(
        r'hospital|doctor|surgery|sick|hurt|injur|broken|emergency room|er visit|ambulance', t)))
    
    # Commercialization signals
    features['has_brand'] = int(bool(re.search(
        r'target|walmart|amazon|disney|nike|iphone|ipad|apple|samsung|xbox|playstation|nintendo', t)))
    features['has_money'] = int(bool(re.search(
        r'\$|dollar|money|expensive|cheap|cost|buy|bought|shopping|spend|afford|rich|million', t)))
    features['has_giveaway'] = int(bool(re.search(
        r'giveaway|give\s*away|free|win|contest|sweepstake', t)))
    
    # Life event / milestone
    features['has_milestone'] = int(bool(re.search(
        r'birthday|christmas|halloween|easter|first\s*(day|time|step|word)|baby|pregnant|born|wedding|married|anniversary|graduat', t)))
    features['has_travel'] = int(bool(re.search(
        r'travel|vacation|trip|hotel|resort|beach|pool|water\s*park|disney\s*(land|world)|theme\s*park|fly|airport|road\s*trip', t)))
    
    # Structural features (non-channel-specific)
    features['title_length'] = len(str(title))
    features['word_count'] = len(str(title).split())
    features['caps_ratio'] = sum(1 for c in str(title) if c.isupper()) / max(len(str(title)), 1)
    features['exclamation_count'] = str(title).count('!')
    features['question_mark'] = int('?' in str(title))
    features['has_emoji'] = int(bool(re.search(r'[^\x00-\x7F]', str(title))))  # non-ASCII proxy
    features['has_ellipsis'] = int('...' in str(title))
    features['has_number'] = int(bool(re.search(r'\d+', t)))
    features['has_all_caps_word'] = int(bool(re.search(r'\b[A-Z]{3,}\b', str(title))))
    
    return features

# Extract for all videos
print("  Extracting features for all videos...")
feature_list = []
for i, row in df.iterrows():
    feature_list.append(extract_features(row['title']))
    if (i+1) % 10000 == 0:
        print(f"    {i+1}/{len(df)}")

features_df = pd.DataFrame(feature_list)
print(f"  Feature matrix: {features_df.shape}")

# ============================================================
# STAGE 2: LLM-based semantic labeling (batch of 2000 samples)
# ============================================================
print("\n[Stage 2] LLM-based content type labeling (stratified sample)...")

# Sample 2000 videos stratified by channel
sample_per_channel = max(2000 // df['channel_short_name'].nunique(), 50)
sampled = df.groupby('channel_short_name').apply(
    lambda x: x.sample(min(len(x), sample_per_channel), random_state=42)
).reset_index(drop=True)
# Ensure we have ~2000
if len(sampled) > 2500:
    sampled = sampled.sample(2000, random_state=42)
print(f"  LLM sample size: {len(sampled)}")

client = OpenAI()

# Process in batches of 50 titles
batch_size = 50
llm_results = []
n_batches = (len(sampled) + batch_size - 1) // batch_size

for batch_idx in range(n_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, len(sampled))
    batch = sampled.iloc[start:end]
    
    titles_text = "\n".join([f"{i+1}. {row['title']}" for i, (_, row) in enumerate(batch.iterrows())])
    
    prompt = f"""Analyze these YouTube video titles and classify each one. For each title, provide:
1. content_type: ONE of [challenge, unboxing, prank, vlog, tutorial, reaction, game, roleplay, music, toy_play, storytime, mukbang, lifestyle, drama, educational, other]
2. target_audience: ONE of [toddler, young_child, tween, teen, family, general]
3. emotional_tone: ONE of [happy, excited, dramatic, sad, neutral, mysterious, urgent]
4. commercialization: ONE of [high, medium, low, none] (based on product/brand signals)
5. exploitation_risk: ONE of [high, medium, low] (child emotional manipulation, privacy invasion, staged distress)

Titles:
{titles_text}

Return ONLY a JSON array with objects for each title (in order). Each object has keys: content_type, target_audience, emotional_tone, commercialization, exploitation_risk.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Handle different response formats
        if isinstance(result, dict):
            if 'results' in result:
                items = result['results']
            elif 'titles' in result:
                items = result['titles']
            else:
                # Try to find the array
                for v in result.values():
                    if isinstance(v, list):
                        items = v
                        break
                else:
                    items = []
        elif isinstance(result, list):
            items = result
        else:
            items = []
        
        # Pad or trim to match batch size
        while len(items) < (end - start):
            items.append({'content_type': 'other', 'target_audience': 'general', 
                         'emotional_tone': 'neutral', 'commercialization': 'none',
                         'exploitation_risk': 'low'})
        items = items[:end-start]
        
        for i, item in enumerate(items):
            item['video_id'] = batch.iloc[i]['id']
            llm_results.append(item)
            
    except Exception as e:
        print(f"  Batch {batch_idx} error: {e}")
        for i in range(end - start):
            llm_results.append({
                'video_id': batch.iloc[i]['id'],
                'content_type': 'other', 'target_audience': 'general',
                'emotional_tone': 'neutral', 'commercialization': 'none',
                'exploitation_risk': 'low'
            })
    
    if (batch_idx + 1) % 10 == 0:
        print(f"    Batch {batch_idx+1}/{n_batches} done")

print(f"  LLM labeling complete: {len(llm_results)} videos labeled")

# Save LLM results
llm_df = pd.DataFrame(llm_results)
llm_df.to_csv('analysis_discovery/llm_content_labels.csv', index=False)

# ============================================================
# COMBINE: Merge rule-based + LLM features
# ============================================================
print("\n[Stage 3] Combining features...")

# Add rule-based features to main df
full_features = pd.concat([df[['id', 'channel_short_name', 'viewCount']], features_df], axis=1)
full_features.to_csv('analysis_discovery/content_features_full.csv', index=False)

# Print feature coverage stats
print("\nFeature coverage (% of videos with feature=1):")
binary_cols = [c for c in features_df.columns if features_df[c].dtype in ['int64', 'float64'] and features_df[c].max() <= 1]
for col in sorted(binary_cols, key=lambda c: features_df[c].mean(), reverse=True):
    pct = features_df[col].mean()
    if pct > 0.01:  # Only show features present in >1% of videos
        print(f"  {col:<25}: {pct:.1%}")

# LLM label distribution
print("\nLLM content_type distribution:")
if 'content_type' in llm_df.columns:
    print(llm_df['content_type'].value_counts().to_string())

print("\nLLM exploitation_risk distribution:")
if 'exploitation_risk' in llm_df.columns:
    print(llm_df['exploitation_risk'].value_counts().to_string())

print("\n=== Feature extraction complete ===")
print(f"Output files:")
print(f"  analysis_discovery/content_features_full.csv ({len(full_features)} rows x {len(full_features.columns)} cols)")
print(f"  analysis_discovery/llm_content_labels.csv ({len(llm_df)} rows)")
