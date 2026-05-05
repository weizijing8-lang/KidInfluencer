"""
Generate a human annotation sheet for ground truth validation.
Samples 200 videos stratified by exploitation score (high/medium/low) and channel type.
Outputs a CSV that annotators can fill in.
"""
import pandas as pd
import numpy as np

# Load the scored dataset
df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v3/videos_with_exploitation_scores.csv')

# Stratified sample: 
# - 70 from high exploitation score (top 25%)
# - 70 from medium (middle 50%)
# - 60 from low (bottom 25%)
# Within each stratum, sample proportionally from child vs teen/adult channels

np.random.seed(42)

q75 = df['exploitation_score'].quantile(0.75)
q25 = df['exploitation_score'].quantile(0.25)

high = df[df['exploitation_score'] >= q75].sample(n=min(70, len(df[df['exploitation_score'] >= q75])), random_state=42)
medium = df[(df['exploitation_score'] > q25) & (df['exploitation_score'] < q75)].sample(n=min(70, len(df[(df['exploitation_score'] > q25) & (df['exploitation_score'] < q75)])), random_state=42)
low = df[df['exploitation_score'] <= q25].sample(n=min(60, len(df[df['exploitation_score'] <= q25])), random_state=42)

annotation_sample = pd.concat([high, medium, low]).sample(frac=1, random_state=42).reset_index(drop=True)

# Create annotation sheet
annotation_sheet = pd.DataFrame({
    'video_id': annotation_sample['id'],
    'title': annotation_sample['title'],
    'channel': annotation_sample['channel_short_name'],
    'youtube_url': annotation_sample['id'].apply(lambda x: f'https://www.youtube.com/watch?v={x}'),
    'view_count': annotation_sample['viewCount'],
    # Model predictions (hidden from annotators in practice, included for validation)
    'model_exploitation_score': annotation_sample['exploitation_score'].round(3),
    'model_performative': annotation_sample['performative'],
    'model_emotional_bait': annotation_sample['emotional_bait'],
    'model_narrative_conflict': annotation_sample['narrative_conflict'],
    'model_challenge_format': annotation_sample['challenge_format'],
    'model_commercial_content': annotation_sample['commercial_content'],
    'model_privacy_violation': annotation_sample['privacy_violation'],
    # Columns for annotator to fill in
    'annotator_performative_labor': '',
    'annotator_emotional_bait': '',
    'annotator_narrative_conflict': '',
    'annotator_challenge_format': '',
    'annotator_commercial_content': '',
    'annotator_privacy_violation': '',
    'annotator_overall_exploitative': '',
    'annotator_notes': '',
})

# Save full sheet (with model predictions for later validation)
annotation_sheet.to_csv('/home/ubuntu/KidInfluencer/data/annotation_sheet_full.csv', index=False)

# Save annotator version (without model predictions to avoid bias)
annotator_cols = ['video_id', 'title', 'channel', 'youtube_url', 'view_count',
                  'annotator_performative_labor', 'annotator_emotional_bait',
                  'annotator_narrative_conflict', 'annotator_challenge_format',
                  'annotator_commercial_content', 'annotator_privacy_violation',
                  'annotator_overall_exploitative', 'annotator_notes']
annotation_sheet[annotator_cols].to_csv('/home/ubuntu/KidInfluencer/data/annotation_sheet_for_annotators.csv', index=False)

print(f"Generated annotation sheet with {len(annotation_sheet)} videos")
print(f"  High exploitation (score >= {q75:.3f}): {len(high)}")
print(f"  Medium exploitation: {len(medium)}")
print(f"  Low exploitation (score <= {q25:.3f}): {len(low)}")
print(f"\nChannels represented: {annotation_sheet['channel'].nunique()}")
print(f"\nFiles saved:")
print(f"  - data/annotation_sheet_full.csv (with model predictions, for validation)")
print(f"  - data/annotation_sheet_for_annotators.csv (without model predictions, for blind annotation)")
print(f"\nInstructions for annotators:")
print(f"  1. Watch the video (or at minimum read the title + view thumbnail)")
print(f"  2. For each dimension, mark 1 (present) or 0 (absent)")
print(f"  3. Mark overall_exploitative as 1 if ANY dimension is present")
print(f"  4. Add notes for ambiguous cases")
