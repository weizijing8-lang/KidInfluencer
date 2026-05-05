"""
Hypothesis: Channels targeting child audiences have LOWER exploitation premium,
while channels targeting teen/adult audiences have HIGHER exploitation premium.

We classify channels based on content characteristics:
- Child-audience: animated content, toy play, nursery rhymes, very young children featured
- Teen/Adult-audience: vlogs, challenges, pranks, drama, older children/teens featured
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('analysis_discovery/snorkel_results_v2/videos_with_exploitation_scores.csv')
df = df[df['viewCount'].notna() & (df['viewCount'] > 0)].copy()

# Manual classification based on known channel content
# "child" = primary audience is young children (under 8), content is made FOR kids
# "teen_adult" = primary audience is teens/adults, content is ABOUT kids but watched by older viewers
audience_map = {
    # Child-audience channels (Made for Kids, animated/toy content, preschool-age viewers)
    'likenastya': 'child',
    'kidsdianashow': 'child',
    'vladandniki': 'child',  # toy play, animated, very young
    'ryansworld': 'child',  # toy reviews for kids
    'aforadley': 'child',  # imaginative play for young kids
    'cocomelon': 'child',  # nursery rhymes (excluded if not in data)
    'smellybellytv': 'child',  # young kids content
    'ethangamer': 'child',  # gaming for kids
    'funsquadfamily': 'child',  # family challenges but kid-oriented
    
    # Teen/Adult-audience channels (vlogs, drama, challenges watched by teens/adults)
    'brentrivera': 'teen_adult',
    'piperrockelle': 'teen_adult',
    'rebeccazamolo': 'teen_adult',
    'jordanmatter': 'teen_adult',
    'norrisnuts': 'teen_adult',
    'acefamily': 'teen_adult',
    'royaltyfamily': 'teen_adult',
    'bratayley': 'teen_adult',
    'gavinmagnus': 'teen_adult',
    'wearethedavises': 'teen_adult',
    'tannerites': 'teen_adult',
    'itsyeboi': 'teen_adult',
    'shotofyeagers': 'teen_adult',
    'piersonwodzynski': 'teen_adult',
    'tydustalbott': 'teen_adult',
    'everleighrose': 'teen_adult',
    
    # Family vlog channels (mixed audience, but primarily adult viewers watching family content)
    'dailybumps': 'teen_adult',
    'thesacconejolys': 'teen_adult',
    'ballingerfamily': 'teen_adult',
    'inghamfamily': 'teen_adult',
    'jhousevlogs': 'teen_adult',
    'familyfizz': 'teen_adult',
    'familyfunpack': 'teen_adult',
    'familyfudge': 'teen_adult',
    'ohanaadventure': 'teen_adult',
    'theweisslife': 'teen_adult',
    'theleray': 'teen_adult',
    'jesssfam': 'teen_adult',
    'bonniehoellein': 'teen_adult',
    'thatyoutub3family': 'teen_adult',
    'kkandbabyj': 'teen_adult',
    'johnsonsfam': 'teen_adult',
    'onyxfamily': 'teen_adult',
    'mccluretwins': 'teen_adult',
    'samandnia': 'teen_adult',
    'ehbee': 'teen_adult',
    'jillianandaddie': 'teen_adult',
}

df['audience'] = df['channel_short_name'].map(audience_map)

# Compute within-channel boost for each channel
channel_boosts = []
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
    boost = (high_median / low_median - 1) * 100
    audience = audience_map.get(ch, 'unknown')
    channel_boosts.append({
        'channel': ch,
        'audience': audience,
        'boost_pct': boost,
        'high_median': high_median,
        'low_median': low_median,
        'n': len(ch_data),
    })

boost_df = pd.DataFrame(channel_boosts)

print("=" * 80)
print("AUDIENCE SEGMENTATION ANALYSIS")
print("=" * 80)

# Split by audience type
child_channels = boost_df[boost_df['audience'] == 'child']
teen_adult_channels = boost_df[boost_df['audience'] == 'teen_adult']

print(f"\n--- Child-Audience Channels ({len(child_channels)} channels) ---")
print(f"  Mean boost: {child_channels['boost_pct'].mean():+.1f}%")
print(f"  Median boost: {child_channels['boost_pct'].median():+.1f}%")
print(f"  Channels with positive boost: {(child_channels['boost_pct'] > 0).sum()}/{len(child_channels)}")
print(f"  Individual channels:")
for _, row in child_channels.sort_values('boost_pct', ascending=False).iterrows():
    print(f"    {row['channel']:<20} boost: {row['boost_pct']:+.1f}%")

print(f"\n--- Teen/Adult-Audience Channels ({len(teen_adult_channels)} channels) ---")
print(f"  Mean boost: {teen_adult_channels['boost_pct'].mean():+.1f}%")
print(f"  Median boost: {teen_adult_channels['boost_pct'].median():+.1f}%")
print(f"  Channels with positive boost: {(teen_adult_channels['boost_pct'] > 0).sum()}/{len(teen_adult_channels)}")

# Statistical test: is the boost different between the two groups?
print("\n--- Statistical Tests ---")
t_stat, p_val = stats.ttest_ind(teen_adult_channels['boost_pct'], child_channels['boost_pct'])
print(f"  Independent t-test (teen/adult vs child): t={t_stat:.3f}, p={p_val:.4f}")

u_stat, u_pval = stats.mannwhitneyu(teen_adult_channels['boost_pct'], child_channels['boost_pct'], alternative='greater')
print(f"  Mann-Whitney U (teen/adult > child): U={u_stat:.0f}, p={u_pval:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(
    ((len(teen_adult_channels)-1)*teen_adult_channels['boost_pct'].std()**2 + 
     (len(child_channels)-1)*child_channels['boost_pct'].std()**2) /
    (len(teen_adult_channels) + len(child_channels) - 2)
)
cohens_d = (teen_adult_channels['boost_pct'].mean() - child_channels['boost_pct'].mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f}")

# Also test: within each group, is the boost significantly > 0?
print("\n--- One-sample t-tests (boost > 0) ---")
t_child, p_child = stats.ttest_1samp(child_channels['boost_pct'], 0)
print(f"  Child-audience channels: t={t_child:.3f}, p={p_child:.4f} (two-tailed)")
print(f"    One-tailed p (boost > 0): {p_child/2:.4f}" if t_child > 0 else f"    One-tailed p (boost > 0): {1 - p_child/2:.4f}")

t_teen, p_teen = stats.ttest_1samp(teen_adult_channels['boost_pct'], 0)
print(f"  Teen/Adult-audience channels: t={t_teen:.3f}, p={p_teen:.4f} (two-tailed)")
print(f"    One-tailed p (boost > 0): {p_teen/2:.4f}" if t_teen > 0 else f"    One-tailed p (boost > 0): {1 - p_teen/2:.4f}")

# Summary table
print("\n--- Summary Table ---")
print(f"{'Audience':<15} {'N':>4} {'Mean Boost':>12} {'Median Boost':>14} {'% Positive':>12} {'t-test p':>10}")
print("-" * 70)
print(f"{'Child':<15} {len(child_channels):>4} {child_channels['boost_pct'].mean():>+11.1f}% {child_channels['boost_pct'].median():>+13.1f}% {(child_channels['boost_pct']>0).mean()*100:>10.0f}% {p_child:>10.4f}")
print(f"{'Teen/Adult':<15} {len(teen_adult_channels):>4} {teen_adult_channels['boost_pct'].mean():>+11.1f}% {teen_adult_channels['boost_pct'].median():>+13.1f}% {(teen_adult_channels['boost_pct']>0).mean()*100:>10.0f}% {p_teen:>10.4f}")
print(f"\n  Difference: {teen_adult_channels['boost_pct'].mean() - child_channels['boost_pct'].mean():+.1f} percentage points")
print(f"  Between-group p-value: {p_val:.4f}")
