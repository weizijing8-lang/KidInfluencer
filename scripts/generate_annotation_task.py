"""
Generate annotation task for human validation of LLM annotations.
Stratified sample of 100 videos from the annotated dataset.
Output: Excel spreadsheet + annotation guidelines document.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Load data
am = pd.read_csv('data/annotations_merged.csv')
vids = pd.read_csv('data/combined_videos.csv')
ch = pd.read_csv('data/combined_channels.csv')

# Merge to get full info
merged = am.merge(vids, on='video_id', how='inner')
merged = merged.merge(ch[['channel_id', 'title']], on='channel_id', how='left')
merged.rename(columns={'title_x': 'video_title', 'title_y': 'channel_name'}, inplace=True)

# Remove error rows (LLM failed to annotate)
valid = merged[merged['emotional_manipulation'] != 'error'].copy()
print(f"Valid annotations: {len(valid)} / {len(merged)}")

# Stratified sampling: ensure diversity across dimensions
# Sample 100 videos with stratification on emotional_manipulation and child_role
sample_list = []

# Ensure we get some from each emotional_manipulation level
for level in ['none', 'mild', 'moderate', 'severe']:
    subset = valid[valid['emotional_manipulation'] == level]
    n_sample = min(len(subset), {
        'none': 35, 'mild': 30, 'moderate': 25, 'severe': 10
    }[level])
    sample_list.append(subset.sample(n=n_sample, random_state=42))

sample = pd.concat(sample_list).drop_duplicates(subset='video_id')

# If we have more than 100, trim; if less, add more
if len(sample) > 100:
    sample = sample.sample(n=100, random_state=42)
elif len(sample) < 100:
    remaining = valid[~valid['video_id'].isin(sample['video_id'])]
    extra = remaining.sample(n=100-len(sample), random_state=42)
    sample = pd.concat([sample, extra])

print(f"Final sample size: {len(sample)}")
print(f"Channels represented: {sample['channel_name'].nunique()}")
print(f"\nEmotional manipulation distribution in sample:")
print(sample['emotional_manipulation'].value_counts())
print(f"\nChild role distribution in sample:")
print(sample['child_role'].value_counts())

# Create the annotation spreadsheet
# Include: video_id, video_title, channel_name, YouTube URL
# Human annotators will fill in their judgments
annotation_df = pd.DataFrame({
    'video_id': sample['video_id'],
    'video_title': sample['video_title'],
    'channel_name': sample['channel_name'],
    'youtube_url': 'https://www.youtube.com/watch?v=' + sample['video_id'],
    # Annotator columns (to be filled)
    'emotional_manipulation': '',  # none / mild / moderate / severe
    'commercial_signals': '',  # none / brand_mention / sponsored / product_placement
    'child_role': '',  # protagonist / co_star / cameo / absent / unclear
    'clickbait_level': '',  # none / mild / moderate / severe
    'notes': ''
})

# Save annotation task
os.makedirs('annotation_task', exist_ok=True)
annotation_df.to_excel('annotation_task/annotation_sheet.xlsx', index=False)
print(f"\nSaved annotation sheet to annotation_task/annotation_sheet.xlsx")

# Also save the LLM labels separately (for computing agreement later)
llm_labels = sample[['video_id', 'emotional_manipulation', 'commercial_signals', 
                      'child_role', 'privacy_concern', 'clickbait_level']].copy()
llm_labels.to_csv('annotation_task/llm_labels_ground_truth.csv', index=False)
print(f"Saved LLM labels to annotation_task/llm_labels_ground_truth.csv")

# Print some stats
print(f"\n=== Sample Statistics ===")
print(f"Videos from family channels: {sample[sample['channel_name'].notna()].shape[0]}")
print(f"Mean views: {sample['views'].mean():.0f}")
print(f"Median views: {sample['views'].median():.0f}")
