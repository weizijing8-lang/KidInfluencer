"""
Generate a table of all disagreements between model predictions and Annotator B's labels,
organized by dimension, for user re-verification.
"""
import pandas as pd
import numpy as np

# Parse B's annotations
rows_b = []
with open('/home/ubuntu/upload/pasted_content_7.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            row = {}
            row['video_id'] = parts[0]
            row['title'] = parts[1]
            row['channel'] = parts[2]
            row['url'] = parts[3]
            row['views'] = parts[4]
            dims = parts[5:11] if len(parts) >= 11 else [''] * 6
            overall = parts[11] if len(parts) >= 12 else parts[-1]
            row['performative_labor'] = int(dims[0]) if dims[0].strip() in ['0','1'] else np.nan
            row['emotional_bait'] = int(dims[1]) if dims[1].strip() in ['0','1'] else np.nan
            row['narrative_conflict'] = int(dims[2]) if dims[2].strip() in ['0','1'] else np.nan
            row['challenge_format'] = int(dims[3]) if dims[3].strip() in ['0','1'] else np.nan
            row['commercial_content'] = int(dims[4]) if dims[4].strip() in ['0','1'] else np.nan
            row['privacy_violation'] = int(dims[5]) if dims[5].strip() in ['0','1'] else np.nan
            row['overall_label'] = int(overall) if overall.strip() in ['0','1'] else np.nan
            # Get notes (last field if it's text)
            if len(parts) > 12:
                row['note'] = parts[12]
            else:
                row['note'] = ''
            rows_b.append(row)

df_b = pd.DataFrame(rows_b)

# Apply updates from pasted_content_9
with open('/home/ubuntu/upload/pasted_content_9.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 8:
            vid = parts[0]
            dims = parts[5:11] if len(parts) >= 11 else [''] * 6
            overall = parts[11] if len(parts) >= 12 else parts[-1]
            mask = df_b['video_id'] == vid
            if mask.any():
                if dims[0].strip() in ['0','1']: df_b.loc[mask, 'performative_labor'] = int(dims[0])
                if dims[1].strip() in ['0','1']: df_b.loc[mask, 'emotional_bait'] = int(dims[1])
                if dims[2].strip() in ['0','1']: df_b.loc[mask, 'narrative_conflict'] = int(dims[2])
                if dims[3].strip() in ['0','1']: df_b.loc[mask, 'challenge_format'] = int(dims[3])
                if dims[4].strip() in ['0','1']: df_b.loc[mask, 'commercial_content'] = int(dims[4])
                if dims[5].strip() in ['0','1']: df_b.loc[mask, 'privacy_violation'] = int(dims[5])
                if overall.strip() in ['0','1']: df_b.loc[mask, 'overall_label'] = int(overall)

# Fix Candle Blowing
mask_candle = df_b['video_id'] == 'sS89FnQWwvA'
if mask_candle.any():
    df_b.loc[mask_candle, 'overall_label'] = 1

# Filter to kid-centric channels
adult_channels = [
    'blippi', 'crazygorilla', 'itsyeboi', 'itsrucka', 'dafuqboom',
    'mrbeast', 'sssniperwolf', 'markiplier', 'pewdiepie', 'dude perfect',
    'unspeakable', 'preston', 'brianna', 'zhong', 'ben azelart',
    'stokes twins', 'alan chikin chow', 'matt and abby', 'the royalty family gaming',
    'faze rug', 'carter sharer', 'rebecca zamolo', 'chad wild clay'
]
df_b_kid = df_b[~df_b['channel'].str.lower().isin([c.lower() for c in adult_channels])].copy()

# Load model predictions
snorkel_data = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv')

# Merge
merged = df_b_kid.merge(snorkel_data, left_on='video_id', right_on='id', how='inner', suffixes=('_human', '_model'))

dimensions = ['performative_labor', 'emotional_bait', 'narrative_conflict', 
              'challenge_format', 'commercial_content', 'privacy_violation']

# Generate disagreement table
all_disagreements = []

for dim in dimensions:
    human_col = dim + '_human' if dim + '_human' in merged.columns else dim
    model_col = f"{dim}_pred"
    prob_col = f"{dim}_prob"
    
    for _, row in merged.iterrows():
        human_val = row[human_col] if not pd.isna(row[human_col]) else 0
        model_val = row[model_col]
        
        # Skip abstains
        if model_val == -1:
            continue
        
        human_val = int(human_val)
        model_val = int(model_val)
        
        if human_val != model_val:
            disagreement_type = "FP (model=1, human=0)" if model_val == 1 else "FN (model=0, human=1)"
            all_disagreements.append({
                'dimension': dim,
                'video_id': row['video_id'],
                'title': row['title_human'] if 'title_human' in row else row['title'],
                'channel': row['channel'] if 'channel' in row else row['channel_short_name'],
                'url': row['url'],
                'views': row['views'],
                'model_pred': model_val,
                'model_prob': row[prob_col] if prob_col in row else np.nan,
                'human_label': human_val,
                'disagreement_type': disagreement_type
            })

df_disagree = pd.DataFrame(all_disagreements)
print(f"Total disagreements: {len(df_disagree)}")
print(f"\nBy dimension:")
print(df_disagree.groupby('dimension')['disagreement_type'].value_counts())

# Save to Excel with separate sheets per dimension
with pd.ExcelWriter('/home/ubuntu/KidInfluencer/disagreements_for_verification.xlsx', engine='openpyxl') as writer:
    # Summary sheet
    summary = df_disagree.groupby(['dimension', 'disagreement_type']).size().reset_index(name='count')
    summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # Per-dimension sheets
    for dim in dimensions:
        dim_data = df_disagree[df_disagree['dimension'] == dim].copy()
        if len(dim_data) > 0:
            # Sort: FP first (most likely annotation errors), then by model confidence
            dim_data = dim_data.sort_values(['disagreement_type', 'model_prob'], ascending=[True, False])
            # Select columns for the sheet
            output_cols = ['video_id', 'title', 'channel', 'url', 'views', 
                          'model_pred', 'model_prob', 'human_label', 'disagreement_type']
            dim_data[output_cols].to_excel(writer, sheet_name=dim[:31], index=False)
            
    # Also create a "most likely annotation errors" sheet
    # These are FPs where model confidence is very high (>0.8)
    likely_errors = df_disagree[
        (df_disagree['disagreement_type'] == 'FP (model=1, human=0)') & 
        (df_disagree['model_prob'] > 0.7)
    ].sort_values('model_prob', ascending=False)
    likely_errors.to_excel(writer, sheet_name='Likely_Annotation_Errors', index=False)

print(f"\n\nSaved to: disagreements_for_verification.xlsx")
print(f"\nLikely annotation errors (FP with model confidence > 0.7): {len(likely_errors)}")
print(f"\nTop 10 most likely annotation errors:")
for _, row in likely_errors.head(10).iterrows():
    print(f"  [{row['dimension']}] {row['title'][:50]} (prob={row['model_prob']:.2f})")
    print(f"    {row['url']}")
