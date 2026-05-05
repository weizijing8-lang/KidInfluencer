import pandas as pd
import numpy as np

df = pd.read_csv('analysis_discovery/snorkel_results_v2/videos_with_exploitation_scores.csv')
df = df[df['viewCount'].notna() & (df['viewCount'] > 0)].copy()

results = []

for ch, ch_data in df.groupby('channel_short_name'):
    if len(ch_data) < 10:
        continue
    
    q75 = ch_data['exploitation_score'].quantile(0.75)
    q25 = ch_data['exploitation_score'].quantile(0.25)
    
    high = ch_data[ch_data['exploitation_score'] > q75]
    low = ch_data[ch_data['exploitation_score'] < q25]
    
    if len(high) < 3 or len(low) < 3:
        continue
    
    high_median = high['viewCount'].median()
    low_median = low['viewCount'].median()
    boost_pct = (high_median / low_median - 1) * 100
    
    results.append({
        'channel': ch,
        'n_total': len(ch_data),
        'n_high': len(high),
        'n_low': len(low),
        'high_median_views': high_median,
        'low_median_views': low_median,
        'boost_pct': boost_pct,
    })

res_df = pd.DataFrame(results).sort_values('boost_pct', ascending=False)

print(f"{'Channel':<25} {'N':>4} {'High':>5} {'Low':>5} {'High Median':>14} {'Low Median':>14} {'Boost':>8}")
print("-" * 85)
for _, row in res_df.iterrows():
    print(f"{row['channel']:<25} {row['n_total']:>4} {row['n_high']:>5} {row['n_low']:>5} {row['high_median_views']:>14,.0f} {row['low_median_views']:>14,.0f} {row['boost_pct']:>+7.1f}%")

print("-" * 85)
positive = res_df[res_df['boost_pct'] > 0]
negative = res_df[res_df['boost_pct'] <= 0]
print(f"\nSummary:")
print(f"  Total channels analyzed: {len(res_df)}")
print(f"  Channels where high > low: {len(positive)} ({len(positive)/len(res_df)*100:.0f}%)")
print(f"  Channels where low >= high: {len(negative)} ({len(negative)/len(res_df)*100:.0f}%)")
print(f"  Mean boost across all channels: {res_df['boost_pct'].mean():+.1f}%")
print(f"  Median boost across all channels: {res_df['boost_pct'].median():+.1f}%")

# Weighted by channel size
res_df['weighted'] = res_df['boost_pct'] * res_df['n_total']
weighted_mean = res_df['weighted'].sum() / res_df['n_total'].sum()
print(f"  Weighted mean boost (by channel size): {weighted_mean:+.1f}%")
