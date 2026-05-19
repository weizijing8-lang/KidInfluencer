"""
Generate Annotation Spreadsheet for Human Validation
=====================================================
Creates an Excel file with:
- Stratified sample of 200 videos (mix of high/low confidence, high/low exploitation)
- YouTube links for easy viewing
- Model predictions per dimension
- Columns for human annotator to fill (Correct/Incorrect per dimension)
"""

import pandas as pd
import numpy as np

# Load the classified data
df = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws.csv')
print(f"Total videos: {len(df)}")

# Define dimensions
dimensions = ['performative_labor', 'emotional_bait', 'narrative_conflict',
              'challenge_format', 'commercial_content', 'privacy_violation']

# ============================================================
# Stratified Sampling Strategy
# ============================================================
# We want a mix of:
# 1. High confidence + predicted exploitative (model is sure it's exploit) - 50
# 2. High confidence + predicted clean (model is sure it's clean) - 50
# 3. Low confidence (model is uncertain) - 50
# 4. Random sample from remaining - 50
# Total: 200 videos

np.random.seed(42)

# Sort by confidence
df_sorted = df.sort_values('confidence', ascending=False)

# Group 1: High confidence exploitative
high_conf_exploit = df_sorted[
    (df_sorted['confidence'] >= df_sorted['confidence'].quantile(0.75)) &
    (df_sorted['is_exploitative_ws'] == 1)
].sample(n=min(50, len(df_sorted[(df_sorted['confidence'] >= df_sorted['confidence'].quantile(0.75)) & (df_sorted['is_exploitative_ws'] == 1)])), random_state=42)

# Group 2: High confidence clean
high_conf_clean = df_sorted[
    (df_sorted['confidence'] >= df_sorted['confidence'].quantile(0.75)) &
    (df_sorted['is_exploitative_ws'] == 0)
].sample(n=min(50, len(df_sorted[(df_sorted['confidence'] >= df_sorted['confidence'].quantile(0.75)) & (df_sorted['is_exploitative_ws'] == 0)])), random_state=42)

# Group 3: Low confidence (uncertain)
low_conf = df_sorted[
    df_sorted['confidence'] <= df_sorted['confidence'].quantile(0.25)
].sample(n=50, random_state=42)

# Group 4: Random from remaining
used_ids = set(high_conf_exploit['id']) | set(high_conf_clean['id']) | set(low_conf['id'])
remaining = df_sorted[~df_sorted['id'].isin(used_ids)]
random_sample = remaining.sample(n=50, random_state=42)

# Combine
sample = pd.concat([high_conf_exploit, high_conf_clean, low_conf, random_sample], ignore_index=True)
sample['sample_group'] = (['high_conf_exploit'] * len(high_conf_exploit) +
                          ['high_conf_clean'] * len(high_conf_clean) +
                          ['low_confidence'] * len(low_conf) +
                          ['random'] * len(random_sample))

print(f"\nAnnotation sample: {len(sample)} videos")
print(f"  High confidence exploit: {len(high_conf_exploit)}")
print(f"  High confidence clean: {len(high_conf_clean)}")
print(f"  Low confidence: {len(low_conf)}")
print(f"  Random: {len(random_sample)}")

# ============================================================
# Create the annotation spreadsheet
# ============================================================

# Build the output dataframe
annotation_df = pd.DataFrame()

# Basic info
annotation_df['video_id'] = sample['id']
annotation_df['youtube_link'] = sample['id'].apply(lambda x: f'https://www.youtube.com/watch?v={x}')
annotation_df['title'] = sample['title']
annotation_df['channel'] = sample['channel_short_name']
annotation_df['views'] = sample['viewCount']
annotation_df['sample_group'] = sample['sample_group']

# Model predictions per dimension (probability)
for dim in dimensions:
    prob = sample[f'{dim}_prob'].values
    pred = sample[f'{dim}_pred'].values
    annotation_df[f'MODEL_{dim}_prob'] = prob.round(3)
    annotation_df[f'MODEL_{dim}_pred'] = np.where(pred == 1, 'YES', np.where(pred == 0, 'NO', 'ABSTAIN'))

# Overall score
annotation_df['MODEL_exploitation_score'] = sample['exploitation_score_ws'].round(3)
annotation_df['MODEL_overall_prediction'] = np.where(sample['is_exploitative_ws'] == 1, 'EXPLOITATIVE', 'CLEAN')
annotation_df['MODEL_confidence'] = sample['confidence'].round(3)
annotation_df['MODEL_n_dims_flagged'] = sample['n_dimensions_flagged']

# Empty columns for human annotation
annotation_df[''] = ''  # separator
annotation_df['HUMAN_performative_labor'] = ''  # Fill: 0 or 1
annotation_df['HUMAN_emotional_bait'] = ''
annotation_df['HUMAN_narrative_conflict'] = ''
annotation_df['HUMAN_challenge_format'] = ''
annotation_df['HUMAN_commercial_content'] = ''
annotation_df['HUMAN_privacy_violation'] = ''
annotation_df['HUMAN_overall_exploitative'] = ''  # Fill: 0 or 1
annotation_df['HUMAN_notes'] = ''  # Optional notes

# Shuffle within groups to avoid bias
annotation_df = annotation_df.sample(frac=1, random_state=123).reset_index(drop=True)

# Save to Excel with formatting
output_path = 'analysis_discovery/snorkel_proper/annotation_sheet_200.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    annotation_df.to_excel(writer, sheet_name='Annotations', index=False)
    
    # Add instructions sheet
    instructions = pd.DataFrame({
        'Instructions': [
            'ANNOTATION GUIDE FOR KIDFLUENCER EXPLOITATION RISK ASSESSMENT',
            '',
            '1. Open each YouTube link and watch the video (at least 30 seconds + thumbnail)',
            '2. For each dimension, fill in 1 (present) or 0 (absent):',
            '',
            'DIMENSIONS:',
            'performative_labor: Child is performing scripted/planned content for the camera',
            '  - Examples: Acting in skits, following a script, performing choreography',
            '  - NOT: Child naturally playing, family vlog with incidental child presence',
            '',
            'emotional_bait: Title/thumbnail uses child\'s emotions as clickbait',
            '  - Examples: "My kid CRIED when...", exaggerated emotional reactions in thumbnail',
            '  - NOT: Genuine emotional moments shared respectfully',
            '',
            'narrative_conflict: Manufactured drama/conflict involving the child',
            '  - Examples: "Prank gone WRONG", fake arguments, staged confrontations',
            '  - NOT: Natural family disagreements, educational conflict resolution',
            '',
            'challenge_format: Child participates in challenge/competition format',
            '  - Examples: "24 hour challenge", "last to leave", endurance tests',
            '  - NOT: Fun games, educational activities, short silly challenges',
            '',
            'commercial_content: Child is used for product promotion/unboxing',
            '  - Examples: Toy reviews, sponsored content, "mystery box" openings',
            '  - NOT: Child naturally playing with toys, non-commercial content',
            '',
            'privacy_violation: Child\'s private/vulnerable moments exposed',
            '  - Examples: Medical situations, bathroom content, embarrassing moments',
            '  - NOT: Normal family activities, child consented public performances',
            '',
            'overall_exploitative: Your overall judgment (1=exploitative, 0=not)',
            '',
            '3. If the video is unavailable/deleted, write "UNAVAILABLE" in notes',
            '4. If you\'re unsure, write "UNSURE" in the relevant cell',
            '',
            'MODEL columns show the algorithm\'s predictions for reference.',
            'Your job is to judge independently, then we compare.',
        ]
    })
    instructions.to_excel(writer, sheet_name='Instructions', index=False, header=False)

print(f"\nAnnotation sheet saved to: {output_path}")
print(f"Total videos to annotate: {len(annotation_df)}")

# Also save a CSV version
csv_path = 'analysis_discovery/snorkel_proper/annotation_sheet_200.csv'
annotation_df.to_csv(csv_path, index=False)
print(f"CSV version saved to: {csv_path}")

# Print summary stats
print(f"\nSample statistics:")
print(f"  Channels represented: {annotation_df['channel'].nunique()}")
print(f"  Model predicts exploitative: {(annotation_df['MODEL_overall_prediction'] == 'EXPLOITATIVE').sum()}")
print(f"  Model predicts clean: {(annotation_df['MODEL_overall_prediction'] == 'CLEAN').sum()}")
print(f"  Mean confidence: {annotation_df['MODEL_confidence'].mean():.3f}")
