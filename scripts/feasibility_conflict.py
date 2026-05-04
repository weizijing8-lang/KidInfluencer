"""
Feasibility check: Can we detect staged conflict in titles?
Does it correlate with commercialization?
"""
import pandas as pd
import numpy as np
import re
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency

# Load the big dataset (98K videos with channel info)
v4 = pd.read_csv('data/results_v4/full_results_v4.csv')
print(f"V4 dataset: {v4.shape[0]} videos, {v4['channel_short_name'].nunique()} channels")
print(f"Categories: {v4['channel_category'].value_counts().to_dict()}")

# ============================================================
# STEP 1: Detect staged conflict from titles
# ============================================================
# Keywords that indicate staged conflict / emotional manipulation
conflict_keywords = [
    # Theft/stealing narrative
    'stole', 'stolen', 'stealing', 'thief',
    # Pranks
    'prank', 'pranked', 'pranking',
    # Caught/exposed
    'caught', 'exposed', 'busted', 'exposed',
    # Fighting/conflict
    'fight', 'fighting', 'vs', 'battle', 'war',
    # Destruction
    'destroy', 'destroyed', 'broke', 'broken', 'smash',
    # Deception
    'lie', 'lied', 'lying', 'fake', 'trick', 'tricked',
    # Emergency/danger
    'emergency', '911', 'police', 'arrested', 'jail',
    # Emotional distress
    'cried', 'crying', 'scream', 'screaming', 'angry', 'mad',
    # Challenge/dare (often involves discomfort)
    'challenge', 'dare', 'dared',
    # Punishment
    'grounded', 'punished', 'punishment', 'timeout',
    # Secrets/betrayal
    'secret', 'betrayed', 'snitch',
]

# Also detect emotional amplifiers
emotional_amplifiers = [
    'omg', 'oh no', 'worst', 'never again', 'gone wrong', 
    'not clickbait', 'i can\'t believe', 'shocking', 'insane',
    'crazy', 'epic', 'extreme', 'ultimate'
]

def detect_conflict(title):
    """Returns (has_conflict, conflict_type, n_conflict_words)"""
    if pd.isna(title):
        return False, 'none', 0
    title_lower = str(title).lower()
    found = [kw for kw in conflict_keywords if kw in title_lower]
    if found:
        return True, found[0], len(found)
    return False, 'none', 0

def detect_emotional_amplifier(title):
    if pd.isna(title):
        return False, 0
    title_lower = str(title).lower()
    found = [kw for kw in emotional_amplifiers if kw in title_lower]
    return len(found) > 0, len(found)

# Apply to all videos
v4['has_conflict'] = v4['title'].apply(lambda t: detect_conflict(t)[0])
v4['conflict_type'] = v4['title'].apply(lambda t: detect_conflict(t)[1])
v4['n_conflict_words'] = v4['title'].apply(lambda t: detect_conflict(t)[2])
v4['has_amplifier'] = v4['title'].apply(lambda t: detect_emotional_amplifier(t)[0])
v4['n_amplifiers'] = v4['title'].apply(lambda t: detect_emotional_amplifier(t)[1])
v4['staged_conflict_score'] = v4['n_conflict_words'] + v4['n_amplifiers']

print(f"\n=== Staged Conflict Detection ===")
print(f"Videos with conflict keywords: {v4['has_conflict'].sum()} ({v4['has_conflict'].mean():.1%})")
print(f"Videos with emotional amplifiers: {v4['has_amplifier'].sum()} ({v4['has_amplifier'].mean():.1%})")
print(f"Videos with either: {((v4['has_conflict']) | (v4['has_amplifier'])).sum()} ({((v4['has_conflict']) | (v4['has_amplifier'])).mean():.1%})")

# ============================================================
# STEP 2: Compare family vs adult channels
# ============================================================
print(f"\n=== Family vs Adult ===")
family = v4[v4['channel_category'] == 'family']
adult = v4[v4['channel_category'] != 'family']

print(f"Family: {len(family)} videos, {family['channel_short_name'].nunique()} channels")
print(f"Adult: {len(adult)} videos, {adult['channel_short_name'].nunique()} channels")

fam_conflict_rate = family['has_conflict'].mean()
adult_conflict_rate = adult['has_conflict'].mean()
print(f"\nConflict rate - Family: {fam_conflict_rate:.1%}, Adult: {adult_conflict_rate:.1%}")

fam_amplifier_rate = family['has_amplifier'].mean()
adult_amplifier_rate = adult['has_amplifier'].mean()
print(f"Amplifier rate - Family: {fam_amplifier_rate:.1%}, Adult: {adult_amplifier_rate:.1%}")

# Statistical test
stat, p = mannwhitneyu(family['staged_conflict_score'], adult['staged_conflict_score'], alternative='two-sided')
print(f"Mann-Whitney U (staged_conflict_score): U={stat:.0f}, p={p:.2e}")

# ============================================================
# STEP 3: Within family channels - commercialization vs conflict
# ============================================================
print(f"\n=== Within Family: Commercialization vs Conflict ===")

# Channel-level aggregation
fam_channels = family.groupby('channel_short_name').agg(
    n_videos=('title', 'count'),
    conflict_rate=('has_conflict', 'mean'),
    amplifier_rate=('has_amplifier', 'mean'),
    mean_conflict_score=('staged_conflict_score', 'mean'),
    mean_views=('viewCount', 'mean'),
    total_views=('viewCount', 'sum'),
).reset_index()

# Load channel-level commercialization data
ch_summary = pd.read_csv('data/results_v4/channel_summary_v4.csv')
sponsor = pd.read_csv('data/results_v4/sponsorship_by_channel.csv')

# Merge
fam_channels = fam_channels.merge(
    ch_summary[['channel_short_name', 'n_videos', 'mean_exploit_v4', 'total_views']].rename(columns={'total_views': 'ch_total_views', 'n_videos': 'ch_n_videos'}), 
    on='channel_short_name', how='left'
)
fam_channels = fam_channels.merge(
    sponsor.rename(columns={'channel': 'channel_short_name'})[['channel_short_name', 'n_child_brands', 'sponsor_rate', 'n_sponsored']],
    on='channel_short_name', how='left'
)

# Use total_views as commercialization proxy (bigger = more commercial)
# Also use n_child_brands
print(f"\nFamily channels with data: {len(fam_channels)}")
print(fam_channels[['channel_short_name', 'conflict_rate', 'mean_conflict_score', 'mean_views']].sort_values('conflict_rate', ascending=False).head(10).to_string())

# Correlation: views (proxy for commercialization) vs conflict rate
valid_fam = fam_channels.dropna(subset=['ch_total_views', 'conflict_rate'])
rho, p = spearmanr(valid_fam['ch_total_views'], valid_fam['conflict_rate'])
print(f"\nSpearman: total_views vs conflict_rate: ρ={rho:.3f}, p={p:.3f}")

rho2, p2 = spearmanr(valid_fam['mean_views'], valid_fam['conflict_rate'])
print(f"Spearman: mean_views vs conflict_rate: ρ={rho2:.3f}, p={p2:.3f}")

# n_child_brands vs conflict
valid_brands = fam_channels.dropna(subset=['n_child_brands', 'conflict_rate'])
if len(valid_brands) > 5:
    rho3, p3 = spearmanr(valid_brands['n_child_brands'], valid_brands['conflict_rate'])
    print(f"Spearman: n_child_brands vs conflict_rate: ρ={rho3:.3f}, p={p3:.3f}")

# sponsor_rate vs conflict
valid_sp = fam_channels.dropna(subset=['sponsor_rate', 'conflict_rate'])
if len(valid_sp) > 5:
    rho4, p4 = spearmanr(valid_sp['sponsor_rate'], valid_sp['conflict_rate'])
    print(f"Spearman: sponsor_rate vs conflict_rate: ρ={rho4:.3f}, p={p4:.3f}")

# Split into high/low commercialization (median split on ch_total_views)
median_views = valid_fam['ch_total_views'].median()
high_comm = valid_fam[valid_fam['ch_total_views'] >= median_views]
low_comm = valid_fam[valid_fam['ch_total_views'] < median_views]
print(f"\nHigh commercialization ({len(high_comm)} channels): mean conflict rate = {high_comm['conflict_rate'].mean():.3f}")
print(f"Low commercialization ({len(low_comm)} channels): mean conflict rate = {low_comm['conflict_rate'].mean():.3f}")
stat, p = mannwhitneyu(high_comm['conflict_rate'], low_comm['conflict_rate'], alternative='two-sided')
print(f"Mann-Whitney U: p={p:.3f}")

# ============================================================
# STEP 4: Does conflict predict more views? (within family)
# ============================================================
print(f"\n=== Does Conflict Get More Views? (within family channels) ===")
fam_conflict = family[family['has_conflict'] == True]
fam_no_conflict = family[family['has_conflict'] == False]
print(f"With conflict: median views = {fam_conflict['viewCount'].median():.0f} (n={len(fam_conflict)})")
print(f"Without conflict: median views = {fam_no_conflict['viewCount'].median():.0f} (n={len(fam_no_conflict)})")
stat, p = mannwhitneyu(fam_conflict['viewCount'], fam_no_conflict['viewCount'], alternative='two-sided')
print(f"Mann-Whitney U: p={p:.2e}")
boost = fam_conflict['viewCount'].median() / fam_no_conflict['viewCount'].median() - 1
print(f"View boost from conflict: {boost:+.1%}")

# Same for amplifiers
fam_amp = family[family['has_amplifier'] == True]
fam_no_amp = family[family['has_amplifier'] == False]
print(f"\nWith amplifier: median views = {fam_amp['viewCount'].median():.0f} (n={len(fam_amp)})")
print(f"Without amplifier: median views = {fam_no_amp['viewCount'].median():.0f} (n={len(fam_no_amp)})")
stat, p = mannwhitneyu(fam_amp['viewCount'], fam_no_amp['viewCount'], alternative='two-sided')
print(f"Mann-Whitney U: p={p:.2e}")

# ============================================================
# STEP 5: Top conflict types in family channels
# ============================================================
print(f"\n=== Top Conflict Types in Family Channels ===")
fam_with_conflict = family[family['has_conflict']]
print(fam_with_conflict['conflict_type'].value_counts().head(15))

# Example titles
print(f"\n=== Example Staged Conflict Titles (Family) ===")
high_score = family[family['staged_conflict_score'] >= 3].sort_values('staged_conflict_score', ascending=False)
for _, row in high_score.head(10).iterrows():
    print(f"  [{row['staged_conflict_score']}] {row['title'][:80]} ({row['channel_short_name']})")
