"""
Generate Annotation Spreadsheet V2 - Kid-Centric Channels Only
==============================================================
Uses the filtered dataset (adult channels removed).
Stratified sample of 200 videos for human validation.
"""

import pandas as pd
import numpy as np

# Load the filtered dataset
df = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv')
print(f"Total videos (kid-centric only): {len(df)}")
print(f"Channels: {df['channel_short_name'].nunique()}")

# Define dimensions
dimensions = ['performative_labor', 'emotional_bait', 'narrative_conflict',
              'challenge_format', 'commercial_content', 'privacy_violation']

# ============================================================
# Stratified Sampling Strategy
# ============================================================
np.random.seed(42)

# Group 1: High confidence exploitative (model very sure it's exploit)
high_conf_exploit = df[
    (df['confidence'] >= df['confidence'].quantile(0.75)) &
    (df['is_exploitative_ws'] == 1)
].sample(n=50, random_state=42)

# Group 2: High confidence clean (model very sure it's clean)
high_conf_clean = df[
    (df['confidence'] >= df['confidence'].quantile(0.75)) &
    (df['is_exploitative_ws'] == 0)
].sample(n=50, random_state=42)

# Group 3: Low confidence (model uncertain - most informative for validation)
low_conf = df[
    df['confidence'] <= df['confidence'].quantile(0.25)
].sample(n=50, random_state=42)

# Group 4: Random from remaining
used_ids = set(high_conf_exploit['id']) | set(high_conf_clean['id']) | set(low_conf['id'])
remaining = df[~df['id'].isin(used_ids)]
random_sample = remaining.sample(n=50, random_state=42)

# Combine
sample = pd.concat([high_conf_exploit, high_conf_clean, low_conf, random_sample], ignore_index=True)
sample['sample_group'] = (['high_conf_exploit'] * 50 +
                          ['high_conf_clean'] * 50 +
                          ['low_confidence'] * 50 +
                          ['random'] * 50)

print(f"\nAnnotation sample: {len(sample)} videos")
print(f"  High confidence exploit: 50")
print(f"  High confidence clean: 50")
print(f"  Low confidence: 50")
print(f"  Random: 50")
print(f"  Channels represented: {sample['channel_short_name'].nunique()}")

# ============================================================
# Create the annotation spreadsheet
# ============================================================
annotation_df = pd.DataFrame()

# Basic info
annotation_df['video_id'] = sample['id'].values
annotation_df['youtube_link'] = sample['id'].apply(lambda x: f'https://www.youtube.com/watch?v={x}').values
annotation_df['title'] = sample['title'].values
annotation_df['channel'] = sample['channel_short_name'].values
annotation_df['views'] = sample['viewCount'].values
annotation_df['sample_group'] = sample['sample_group'].values

# Model predictions per dimension
for dim in dimensions:
    prob = sample[f'{dim}_prob'].values
    pred = sample[f'{dim}_pred'].values
    annotation_df[f'MODEL_{dim}_prob'] = np.round(prob, 3)
    annotation_df[f'MODEL_{dim}_pred'] = np.where(pred == 1, 'YES', np.where(pred == 0, 'NO', 'ABSTAIN'))

# Overall score
annotation_df['MODEL_exploitation_score'] = np.round(sample['exploitation_score_ws'].values, 3)
annotation_df['MODEL_overall_prediction'] = np.where(sample['is_exploitative_ws'].values == 1, 'EXPLOITATIVE', 'CLEAN')
annotation_df['MODEL_confidence'] = np.round(sample['confidence'].values, 3)
annotation_df['MODEL_n_dims_flagged'] = sample['n_dimensions_flagged'].values

# Empty columns for human annotation
annotation_df['Human Labeled'] = ''  # Fill: 0 or 1 (overall)
annotation_df['Note'] = ''  # Optional notes

# Shuffle to avoid bias
annotation_df = annotation_df.sample(frac=1, random_state=123).reset_index(drop=True)

# Save to Excel
output_path = 'analysis_discovery/snorkel_proper/annotation_sheet_v2_200.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    annotation_df.to_excel(writer, sheet_name='Annotations', index=False)
    
    # Instructions sheet
    instructions = pd.DataFrame({
        'Instructions': [
            'ANNOTATION GUIDE - KID INFLUENCER EXPLOITATION RISK',
            '=' * 60,
            '',
            'IMPORTANT: All channels in this sheet are kid-centric (children are primary subjects).',
            '',
            'STEPS:',
            '1. Click the YouTube link to watch the video (at least 30 seconds)',
            '2. Look at the title and thumbnail',
            '3. Fill "Human Labeled" column: 1 = exploitative, 0 = not exploitative',
            '4. Add notes if needed (especially if you disagree with the model)',
            '',
            'WHAT COUNTS AS EXPLOITATIVE (based on Clark & Jno-Charles 2025):',
            '',
            '- Child performing scripted/staged content (performative labor)',
            '- Child\'s emotions used as clickbait (emotional bait)',
            '- Manufactured drama/conflict involving child (narrative conflict)',
            '- Child in endurance/physical challenges (challenge format)',
            '- Child used for product promotion (commercial content)',
            '- Child\'s private/vulnerable moments exposed (privacy violation)',
            '',
            'WHAT DOES NOT COUNT:',
            '- Child naturally playing or having fun',
            '- Family documenting genuine moments respectfully',
            '- Educational content where child is learning',
            '- Child voluntarily sharing a talent/hobby',
            '',
            'IF VIDEO IS UNAVAILABLE: Write "UNAVAILABLE" in Note column',
            'IF UNSURE: Write "UNSURE" in Note column',
            '',
            'MODEL columns show the algorithm prediction for reference.',
            'Your job: judge independently based on what you see.',
        ]
    })
    instructions.to_excel(writer, sheet_name='Instructions', index=False, header=False)

# Also save CSV
csv_path = 'analysis_discovery/snorkel_proper/annotation_sheet_v2_200.csv'
annotation_df.to_csv(csv_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Saved: {csv_path}")
print(f"\nSample statistics:")
print(f"  Channels: {annotation_df['channel'].nunique()}")
print(f"  Model predicts exploitative: {(annotation_df['MODEL_overall_prediction'] == 'EXPLOITATIVE').sum()}")
print(f"  Model predicts clean: {(annotation_df['MODEL_overall_prediction'] == 'CLEAN').sum()}")
print(f"  Mean confidence: {annotation_df['MODEL_confidence'].mean():.3f}")
