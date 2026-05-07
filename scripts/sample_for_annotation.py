"""
Sample 50 videos from the Snorkel dataset for human annotation.
Strategy: stratified by exploitation score quintiles, diverse channels.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load Snorkel predictions
df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results/videos_with_exploitation_scores.csv')

print(f"Total videos in Snorkel sample: {len(df)}")
print(f"Score range: [{df['exploitation_score'].min():.3f}, {df['exploitation_score'].max():.3f}]")
print(f"Channels: {df['channel_short_name'].nunique()}")

# Create score bins (5 quintiles)
df['score_bin'] = pd.qcut(df['exploitation_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

print("\nScore bin distribution:")
print(df['score_bin'].value_counts().sort_index())

# Sample 10 videos per bin, ensuring channel diversity
samples = []
for bin_label in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    bin_df = df[df['score_bin'] == bin_label]
    
    # Try to get diverse channels
    channels = bin_df['channel_short_name'].unique()
    
    if len(bin_df) >= 10:
        # Sample with channel diversity: max 2 per channel
        sampled = []
        remaining = bin_df.copy()
        
        while len(sampled) < 10 and len(remaining) > 0:
            # Pick one from each channel first
            for ch in np.random.permutation(remaining['channel_short_name'].unique()):
                if len(sampled) >= 10:
                    break
                ch_videos = remaining[remaining['channel_short_name'] == ch]
                pick = ch_videos.sample(1)
                sampled.append(pick)
                remaining = remaining.drop(pick.index)
        
        bin_sample = pd.concat(sampled)
    else:
        bin_sample = bin_df
    
    samples.append(bin_sample)

annotation_df = pd.concat(samples).reset_index(drop=True)

print(f"\nTotal sampled for annotation: {len(annotation_df)}")
print(f"Channels represented: {annotation_df['channel_short_name'].nunique()}")

# Create the annotation sheet
output = annotation_df[['id', 'title', 'channel_short_name', 'exploitation_score', 
                         'performative', 'emotional_bait', 'narrative_conflict',
                         'challenge_format', 'commercial_content']].copy()

output = output.rename(columns={
    'id': 'video_id',
    'channel_short_name': 'channel',
    'exploitation_score': 'snorkel_score',
    'performative': 'snorkel_performative',
    'emotional_bait': 'snorkel_emotional_bait',
    'narrative_conflict': 'snorkel_narrative_conflict',
    'challenge_format': 'snorkel_challenge_format',
    'commercial_content': 'snorkel_commercial_content',
})

# Add YouTube URL
output['youtube_url'] = 'https://www.youtube.com/watch?v=' + output['video_id']

# Add empty columns for human annotation
output['human_performative'] = ''
output['human_emotional_bait'] = ''
output['human_narrative_conflict'] = ''
output['human_challenge_format'] = ''
output['human_commercial_content'] = ''
output['human_privacy_violation'] = ''
output['human_overall_exploitative'] = ''
output['human_notes'] = ''

# Reorder columns for the annotator
annotator_cols = [
    'video_id', 'title', 'channel', 'youtube_url', 'snorkel_score',
    'human_performative', 'human_emotional_bait', 'human_narrative_conflict',
    'human_challenge_format', 'human_commercial_content', 'human_privacy_violation',
    'human_overall_exploitative', 'human_notes',
    'snorkel_performative', 'snorkel_emotional_bait', 'snorkel_narrative_conflict',
    'snorkel_challenge_format', 'snorkel_commercial_content',
]

output = output[annotator_cols]

# Sort by score for easier annotation (mix of easy and hard cases)
output = output.sort_values('snorkel_score').reset_index(drop=True)

# Save
output.to_csv('/home/ubuntu/KidInfluencer/data/annotation_batch_50.csv', index=False)

print(f"\nSaved to: /home/ubuntu/KidInfluencer/data/annotation_batch_50.csv")

# Print summary table
print("\n" + "=" * 100)
print("ANNOTATION BATCH SUMMARY (50 videos)")
print("=" * 100)
print(f"\n{'#':<4} {'Score':<7} {'Channel':<20} {'Title':<60}")
print("─" * 100)
for i, row in output.iterrows():
    title_short = row['title'][:57] + "..." if len(row['title']) > 57 else row['title']
    print(f"{i+1:<4} {row['snorkel_score']:.3f}  {row['channel']:<20} {title_short}")
