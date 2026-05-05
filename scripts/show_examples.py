import pandas as pd
import numpy as np

df = pd.read_csv('analysis_discovery/snorkel_results_v2/videos_with_exploitation_scores.csv')
df = df[df['viewCount'].notna() & (df['viewCount'] > 0)].copy()

dimensions = ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content', 'privacy_violation']

# Pick channels with clear boost
channels = ['jordanmatter', 'rebeccazamolo', 'brentrivera', 'royaltyfamily', 'aforadley']

print('=' * 90)
print('WITHIN-CHANNEL: Median-representative examples (videos near the median of each group)')
print('=' * 90)

for ch in channels:
    ch_data = df[df['channel_short_name'] == ch].copy()
    if len(ch_data) < 10:
        continue
    
    q75 = ch_data['exploitation_score'].quantile(0.75)
    q25 = ch_data['exploitation_score'].quantile(0.25)
    
    high = ch_data[ch_data['exploitation_score'] > q75].copy()
    low = ch_data[ch_data['exploitation_score'] < q25].copy()
    
    high_median_views = high['viewCount'].median()
    low_median_views = low['viewCount'].median()
    boost = (high_median_views / low_median_views - 1) * 100
    
    print(f'\n{"=" * 90}')
    print(f'Channel: {ch}')
    print(f'  High-exploit group: {len(high)} videos, MEDIAN views = {high_median_views:,.0f}')
    print(f'  Low-exploit group:  {len(low)} videos, MEDIAN views = {low_median_views:,.0f}')
    print(f'  >>> Boost: {boost:+.1f}%')
    print()
    
    # Show videos NEAR the median (within 20% of median views)
    print(f'  TYPICAL high-exploit videos (near median of {high_median_views:,.0f}):')
    high['dist_to_median'] = abs(high['viewCount'] - high_median_views)
    for _, row in high.nsmallest(3, 'dist_to_median').iterrows():
        dims_active = [d.replace('_', ' ') for d in dimensions if row.get(d, 0) == 1]
        print(f'    [{row["viewCount"]:>12,.0f} views] "{row["title"][:65]}"')
        print(f'      Dims: {dims_active}')
    
    print(f'  TYPICAL low-exploit videos (near median of {low_median_views:,.0f}):')
    low['dist_to_median'] = abs(low['viewCount'] - low_median_views)
    for _, row in low.nsmallest(3, 'dist_to_median').iterrows():
        dims_active = [d.replace('_', ' ') for d in dimensions if row.get(d, 0) == 1]
        print(f'    [{row["viewCount"]:>12,.0f} views] "{row["title"][:65]}"')
        print(f'      Dims: {dims_active}')
