"""
Reclassify channels into three categories:
1. kidfluencer: Child is the primary content protagonist (child_protagonist_rate > 0.3 OR known kid channel)
2. family_adult: Family channel but adults are the main stars (child_protagonist_rate <= 0.3)
3. adult: Non-family adult creator channels

Uses multiple signals:
- child_protagonist_rate from LLM annotations
- Channel name/description keywords
- Known kidfluencer channels from literature
"""
import pandas as pd
import numpy as np
import json

# Load data
risk_v3 = pd.read_csv('analysis_v3/channel_risk_scores_v3.csv')
v4 = pd.read_csv('data/results_v4/full_results_v4.csv')
combined_channels = pd.read_csv('data/combined_channels.csv')
annotations = pd.read_csv('data/annotations_merged.csv')

# Get V4 channel list with original category
v4_channels = v4.groupby(['channel_title', 'channel_category']).agg(
    n_videos=('id', 'count'),
    mean_exploit=('exploit_score_v4', 'mean')
).reset_index()

print("=" * 60)
print("CHANNEL RECLASSIFICATION")
print("=" * 60)

# Step 1: For channels in risk_v3 (75 channels with LLM annotations),
# use child_protagonist_rate directly
print("\n--- Step 1: Channels with LLM annotation data (risk_v3) ---")
print(f"Total channels with child_protagonist_rate: {len(risk_v3)}")

# Step 2: For V4 family channels not in risk_v3, we need to classify manually
# based on channel name, description, and known status

# Known kidfluencer channels (child is clearly the star)
KNOWN_KIDFLUENCER = [
    "Ryan's World",          # Ryan Kaji - the prototypical kidfluencer
    "Like Nastya",           # Anastasia Radzinskaya
    "Vlad and Niki",         # Vlad & Nikita Vashketov
    "Cocomelon - Nursery Rhymes",  # Animated but targets/features kids
    "Family Fun Pack",       # Kids are central
    "TheEngineeringFamily",  # Kids participate heavily
    "Bratayley",             # Kids are stars (Caleb, Annie, Hayley)
    "THE WEISS LIFE",        # 6 kids are the content
    "Tannerites",            # Kids are central to challenges
    "The Bee Family",        # Kids featured
    "Caleb Kids Show",       # Kid is star
    "Star Light Kids",       # Kids
    "Not Enough Nelsons",    # 16 kids
    "Meekah - Educational Videos for Kids",
    "CKN",                   # Kids
    "Daily Bumps",           # Kids growing up on camera
    "Family Fizz",           # Kids are central
    "ForEverAndForAva",      # Kids
    "Vlad and Niki",
]

# Known family-but-adult-led channels (adults are the main stars, kids appear)
KNOWN_FAMILY_ADULT = [
    "The ACE Family",        # Austin & Catherine are the stars, kids appear
    "SACCONEJOLYs",          # Parents are main stars
    "The LeRoys",            # Parents-led
    "KKandbabyJ",            # Parents-led
    "Bonnie Hoellein",       # Mom is the star
    "The Labrant Fãm",       # Cole & Savannah are the stars
    "The Ingham Family",     # Parents-led
    "JesssFam",              # Mom is star
    "The Bucket List Family", # Parents-led travel
]

# Known adult creators who happen to be in "family" category but are really adult influencers
KNOWN_ADULT_MISLABELED = [
    "Rebecca Zamolo",        # Adult creator, game/challenge content
    "Brent Rivera",          # Adult creator
    "Piper Rockelle",        # Teen/young adult creator (was kid, now teen)
    "pierson",               # Adult creator
    "ItsYeBoi",              # Adult creator
    "Jordan Matter",         # Adult photographer
    "Andrew Davila",         # Adult creator
]

# Step 3: Compute child_protagonist_rate for V4 family channels using annotations
print("\n--- Step 2: Computing child involvement from annotations ---")

# Get video-level child_role from annotations
# Map to binary: protagonist/co_star/main_protagonist/Central = child involved
child_involved_roles = ['protagonist', 'co_star', 'main_protagonist', 'Central', 'featured']
annotations['child_involved'] = annotations['child_role'].isin(child_involved_roles).astype(int)

# We need to link annotations to channels
# annotations have video_id, V4 has video info
# Let's use the combined_videos to link
combined_videos = pd.read_csv('data/combined_videos.csv')
ann_with_channel = annotations.merge(
    combined_videos[['video_id', 'channel_id']], 
    on='video_id', how='left'
)

# Get channel-level child involvement rate
channel_child_rate = ann_with_channel.groupby('channel_id').agg(
    child_protagonist_rate=('child_involved', 'mean'),
    n_annotated=('video_id', 'count')
).reset_index()

# Merge with channel info
channel_info = combined_channels[['channel_id', 'title']].rename(columns={'title': 'channel_title'})
channel_child_rate = channel_child_rate.merge(channel_info, on='channel_id', how='left')

print(f"Channels with annotation-based child rate: {len(channel_child_rate)}")
print(channel_child_rate[['channel_title', 'child_protagonist_rate', 'n_annotated']].sort_values(
    'child_protagonist_rate', ascending=False).head(20).to_string(index=False))

# Step 4: Final classification
print("\n--- Step 3: Final Classification ---")

# Start with V4 channels
all_channels = v4_channels.copy()

def classify_channel(row):
    name = row['channel_title']
    orig_cat = row['channel_category']
    
    # If originally adult, keep as adult
    if orig_cat == 'adult':
        return 'adult'
    
    # Check known lists first
    if name in KNOWN_KIDFLUENCER:
        return 'kidfluencer'
    if name in KNOWN_FAMILY_ADULT:
        return 'family_adult'
    if name in KNOWN_ADULT_MISLABELED:
        return 'adult_mislabeled'
    
    # Use child_protagonist_rate if available
    rate_row = channel_child_rate[channel_child_rate['channel_title'] == name]
    if len(rate_row) > 0:
        rate = rate_row.iloc[0]['child_protagonist_rate']
        if rate > 0.35:
            return 'kidfluencer'
        elif rate > 0.15:
            return 'family_adult'
        else:
            return 'adult_mislabeled'
    
    # Also check risk_v3
    rate_row2 = risk_v3[risk_v3['channel_title'] == name]
    if len(rate_row2) > 0:
        rate = rate_row2.iloc[0]['child_protagonist_rate']
        if rate > 0.35:
            return 'kidfluencer'
        elif rate > 0.15:
            return 'family_adult'
        else:
            return 'adult_mislabeled'
    
    # Default: if in family category but no data, classify as family_adult
    return 'family_adult'

all_channels['new_category'] = all_channels.apply(classify_channel, axis=1)

print("\n=== RECLASSIFICATION RESULTS ===")
print(all_channels['new_category'].value_counts())
print()

# Show each category
for cat in ['kidfluencer', 'family_adult', 'adult_mislabeled', 'adult']:
    subset = all_channels[all_channels['new_category'] == cat]
    print(f"\n--- {cat.upper()} ({len(subset)} channels) ---")
    print(subset[['channel_title', 'n_videos', 'mean_exploit']].sort_values('n_videos', ascending=False).to_string(index=False))

# Step 5: Create a 3-way classification for analysis
# Merge adult_mislabeled into adult
all_channels['final_category'] = all_channels['new_category'].replace('adult_mislabeled', 'adult')
print("\n\n=== FINAL 3-WAY CLASSIFICATION ===")
print(all_channels['final_category'].value_counts())

# Save
all_channels.to_csv('analysis_paper1_v2/channel_reclassification.csv', index=False)
print("\nSaved to analysis_paper1_v2/channel_reclassification.csv")

# Also create a mapping for V4 videos
channel_map = all_channels[['channel_title', 'final_category']].drop_duplicates()
v4_reclassified = v4.merge(channel_map, on='channel_title', how='left')
print(f"\n=== V4 Videos by new category ===")
print(v4_reclassified['final_category'].value_counts())
v4_reclassified.to_csv('analysis_paper1_v2/v4_reclassified.csv', index=False)
print("Saved v4_reclassified.csv")

# Summary stats
print("\n\n" + "=" * 60)
print("SUMMARY STATISTICS BY NEW CATEGORY")
print("=" * 60)
for cat in ['kidfluencer', 'family_adult', 'adult']:
    subset = all_channels[all_channels['final_category'] == cat]
    print(f"\n{cat}: {len(subset)} channels, {subset['n_videos'].sum()} videos")
    print(f"  Mean exploit score: {subset['mean_exploit'].mean():.4f}")
