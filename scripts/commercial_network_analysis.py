"""
Commercial Exploitation & Collaboration Network Analysis
=========================================================
Detects:
1. Sponsored content / ads in video descriptions
2. Collaboration patterns between channels (especially kid channels)
3. MCN / management company connections
4. Brand partnerships targeting child audiences
"""

import json
import glob
import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

BASE_DIR = '/home/ubuntu/KidInfluencer'

# ============================================================
# PART 1: Sponsorship / Ad Detection
# ============================================================

# Patterns indicating sponsored content
SPONSOR_PATTERNS = [
    r'#ad\b', r'#sponsored', r'#partner', r'#collab\b',
    r'\bsponsored by\b', r'\bbrought to you by\b', r'\bpaid promotion\b',
    r'\bpaid partnership\b', r'\bin partnership with\b',
    r'\bthank(?:s| you) to .+ for sponsoring\b',
    r'\buse (?:my |our )?code\b', r'\buse (?:my |our )?link\b',
    r'\bdiscount code\b', r'\bpromo code\b', r'\bcoupon code\b',
    r'\baffiliate link\b', r'\bcommission\b',
    r'\bthis video is sponsored\b', r'\btoday\'s sponsor\b',
    r'\bspecial thanks to .+ for sponsoring\b',
]

# Known brands that target children
CHILD_BRANDS = [
    'mattel', 'hasbro', 'lego', 'disney', 'nickelodeon', 'cartoon network',
    'hot wheels', 'barbie', 'nerf', 'play-doh', 'playdoh',
    'roblox', 'fortnite', 'minecraft', 'pokemon', 'nintendo',
    'candy', 'cereal', 'juice', 'gummy', 'fruit snack',
    'squishm', 'squishy', 'slime', 'fidget',
    'ryan\'s world', 'cocomelon', 'paw patrol',
    'target', 'walmart', 'amazon kids',
]

# Brands that are inappropriate for child audiences
INAPPROPRIATE_BRANDS = [
    'betterhelp', 'manscaped', 'ridge wallet', 'raycon',
    'nordvpn', 'expressvpn', 'surfshark',
    'skillshare', 'squarespace', 'audible',
    'hello fresh', 'factor meals',
    'dollar shave', 'keeps', 'hims',
    'raid shadow legends', 'genshin',
]


def detect_sponsorships(descriptions_dir):
    """Detect sponsored content from video descriptions."""
    results = []
    
    desc_files = sorted(glob.glob(os.path.join(descriptions_dir, '*_desc.json')))
    
    for desc_file in desc_files:
        channel_name = os.path.basename(desc_file).replace('_desc.json', '')
        
        with open(desc_file) as f:
            data = json.load(f)
        
        for video_id, info in data.items():
            desc = info.get('description', '')
            tags = info.get('tags', [])
            
            if not desc:
                continue
            
            desc_lower = desc.lower()
            tags_lower = ' '.join(tags).lower() if tags else ''
            combined = desc_lower + ' ' + tags_lower
            
            # Check for sponsorship indicators
            is_sponsored = False
            sponsor_type = []
            
            for pattern in SPONSOR_PATTERNS:
                if re.search(pattern, combined):
                    is_sponsored = True
                    sponsor_type.append(pattern)
            
            # Check for child-targeting brands
            child_brand_mentions = []
            for brand in CHILD_BRANDS:
                if brand in combined:
                    child_brand_mentions.append(brand)
            
            # Check for inappropriate brands
            inappropriate_brand_mentions = []
            for brand in INAPPROPRIATE_BRANDS:
                if brand in combined:
                    inappropriate_brand_mentions.append(brand)
            
            results.append({
                'channel': channel_name,
                'video_id': video_id,
                'is_sponsored': is_sponsored,
                'sponsor_indicators': '|'.join(sponsor_type) if sponsor_type else '',
                'child_brands': '|'.join(child_brand_mentions) if child_brand_mentions else '',
                'inappropriate_brands': '|'.join(inappropriate_brand_mentions) if inappropriate_brand_mentions else '',
                'n_child_brands': len(child_brand_mentions),
                'n_inappropriate_brands': len(inappropriate_brand_mentions),
            })
    
    return pd.DataFrame(results)


# ============================================================
# PART 2: Collaboration Network Detection
# ============================================================

def detect_collaborations(raw_dir, descriptions_dir):
    """Detect collaborations between channels from titles and descriptions."""
    
    # Build a mapping of channel names/handles for detection
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.json')))
    
    channel_info = {}
    all_handles = {}
    all_names = {}
    
    for raw_file in raw_files:
        short_name = os.path.basename(raw_file).replace('.json', '')
        with open(raw_file) as f:
            data = json.load(f)
        
        handle = data.get('handle', '')
        title = data.get('channel_title', '')
        category = data.get('category', 'unknown')
        
        channel_info[short_name] = {
            'handle': handle,
            'title': title,
            'category': category,
        }
        
        if handle:
            # Store without @ for matching
            clean_handle = handle.lstrip('@').lower()
            all_handles[clean_handle] = short_name
        
        if title:
            all_names[title.lower()] = short_name
    
    # Also add common variations
    name_variants = {
        'ace family': 'acefamily',
        'labrant fam': 'labrantfam',
        'ryan': 'ryansworld',
        'piper rockelle': 'piperrockelle',
        'jordan matter': 'jordanmatter',
        'rebecca zamolo': 'rebeccazamolo',
        'brent rivera': 'brentrivera',
        'pierson': 'piersonwodzynski',
        'daily bumps': 'dailybumps',
        'family fun pack': 'familyfunpack',
        'vlad and niki': 'vladandniki',
    }
    
    # Detect collaborations
    collaborations = []  # (channel_a, channel_b, video_id, evidence)
    
    for raw_file in raw_files:
        short_name = os.path.basename(raw_file).replace('.json', '')
        
        with open(raw_file) as f:
            data = json.load(f)
        
        videos = data.get('videos', [])
        
        for video in videos:
            title = video.get('title', '').lower()
            video_id = video['id']
            
            # Check title for mentions of other channels
            for other_name, other_short in name_variants.items():
                if other_short != short_name and other_name in title:
                    collaborations.append({
                        'channel_a': short_name,
                        'channel_b': other_short,
                        'video_id': video_id,
                        'evidence': f'title contains "{other_name}"',
                        'source': 'title',
                    })
            
            # Check for @mentions in title
            at_mentions = re.findall(r'@(\w+)', title)
            for mention in at_mentions:
                mention_lower = mention.lower()
                if mention_lower in all_handles and all_handles[mention_lower] != short_name:
                    collaborations.append({
                        'channel_a': short_name,
                        'channel_b': all_handles[mention_lower],
                        'video_id': video_id,
                        'evidence': f'@{mention} in title',
                        'source': 'title',
                    })
    
    # Also check descriptions for @mentions
    desc_files = sorted(glob.glob(os.path.join(descriptions_dir, '*_desc.json')))
    
    for desc_file in desc_files:
        channel_name = os.path.basename(desc_file).replace('_desc.json', '')
        
        with open(desc_file) as f:
            data = json.load(f)
        
        for video_id, info in data.items():
            desc = info.get('description', '')
            if not desc:
                continue
            
            # Check for @mentions in description
            at_mentions = re.findall(r'@(\w+)', desc.lower())
            for mention in at_mentions:
                if mention in all_handles and all_handles[mention] != channel_name:
                    collaborations.append({
                        'channel_a': channel_name,
                        'channel_b': all_handles[mention],
                        'video_id': video_id,
                        'evidence': f'@{mention} in description',
                        'source': 'description',
                    })
            
            # Check for "feat." or "ft." or "with" + channel name
            for other_name, other_short in name_variants.items():
                if other_short != channel_name:
                    patterns = [
                        f'feat\\.?\\s*{re.escape(other_name)}',
                        f'ft\\.?\\s*{re.escape(other_name)}',
                        f'with\\s+{re.escape(other_name)}',
                        f'featuring\\s+{re.escape(other_name)}',
                    ]
                    for pat in patterns:
                        if re.search(pat, desc.lower()):
                            collaborations.append({
                                'channel_a': channel_name,
                                'channel_b': other_short,
                                'video_id': video_id,
                                'evidence': f'"{other_name}" in description',
                                'source': 'description',
                            })
                            break
    
    collab_df = pd.DataFrame(collaborations)
    return collab_df, channel_info


def build_network(collab_df, channel_info):
    """Build collaboration network and compute metrics."""
    
    if collab_df.empty:
        print("No collaborations detected!")
        return pd.DataFrame()
    
    # Deduplicate: same pair + same video = 1 collaboration
    collab_unique = collab_df.drop_duplicates(subset=['channel_a', 'channel_b', 'video_id'])
    
    # Build edge weights (number of collaboration videos)
    edge_counts = defaultdict(int)
    for _, row in collab_unique.iterrows():
        pair = tuple(sorted([row['channel_a'], row['channel_b']]))
        edge_counts[pair] += 1
    
    # Network metrics
    network_data = []
    for (ch_a, ch_b), count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        cat_a = channel_info.get(ch_a, {}).get('category', 'unknown')
        cat_b = channel_info.get(ch_b, {}).get('category', 'unknown')
        network_data.append({
            'channel_a': ch_a,
            'channel_b': ch_b,
            'category_a': cat_a,
            'category_b': cat_b,
            'collaboration_count': count,
            'both_family': cat_a == 'family' and cat_b == 'family',
        })
    
    network_df = pd.DataFrame(network_data)
    
    # Degree centrality (how many unique collaborators each channel has)
    degree = defaultdict(set)
    for _, row in network_df.iterrows():
        degree[row['channel_a']].add(row['channel_b'])
        degree[row['channel_b']].add(row['channel_a'])
    
    centrality = []
    for ch, partners in sorted(degree.items(), key=lambda x: -len(x[1])):
        cat = channel_info.get(ch, {}).get('category', 'unknown')
        family_partners = sum(1 for p in partners if channel_info.get(p, {}).get('category') == 'family')
        centrality.append({
            'channel': ch,
            'category': cat,
            'degree': len(partners),
            'family_partners': family_partners,
            'adult_partners': len(partners) - family_partners,
        })
    
    centrality_df = pd.DataFrame(centrality)
    
    return network_df, centrality_df


# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == '__main__':
    raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
    desc_dir = os.path.join(BASE_DIR, 'data', 'descriptions')
    output_dir = os.path.join(BASE_DIR, 'data', 'results_v4')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load channel categories
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
    from channel_list import CHANNELS
    channel_cats = {ch[0]: ch[2] for ch in CHANNELS}
    
    # ---- SPONSORSHIP ANALYSIS ----
    print("="*60)
    print("SPONSORSHIP / AD DETECTION")
    print("="*60)
    
    sponsor_df = detect_sponsorships(desc_dir)
    print(f"\nTotal videos analyzed: {len(sponsor_df)}")
    print(f"Sponsored videos detected: {sponsor_df['is_sponsored'].sum()} ({100*sponsor_df['is_sponsored'].mean():.1f}%)")
    
    # Add category
    sponsor_df['category'] = sponsor_df['channel'].map(channel_cats).fillna('unknown')
    
    # Family vs Adult sponsorship rates
    family_sp = sponsor_df[sponsor_df['category'] == 'family']
    adult_sp = sponsor_df[sponsor_df['category'] == 'adult']
    
    print(f"\nFamily channels: {100*family_sp['is_sponsored'].mean():.1f}% sponsored")
    print(f"Adult channels: {100*adult_sp['is_sponsored'].mean():.1f}% sponsored")
    
    # Child brand targeting
    print(f"\nChild brand mentions (family): {family_sp['n_child_brands'].sum()}")
    print(f"Child brand mentions (adult): {adult_sp['n_child_brands'].sum()}")
    
    # Per-channel sponsorship rates
    print("\n--- TOP SPONSORED FAMILY CHANNELS ---")
    ch_sponsor = sponsor_df.groupby(['channel', 'category']).agg(
        n_videos=('video_id', 'count'),
        n_sponsored=('is_sponsored', 'sum'),
        sponsor_rate=('is_sponsored', 'mean'),
        n_child_brands=('n_child_brands', 'sum'),
    ).reset_index()
    
    family_channels = ch_sponsor[ch_sponsor['category'] == 'family'].sort_values('sponsor_rate', ascending=False)
    for _, row in family_channels.head(15).iterrows():
        print(f"  {row['channel']:25s}: {100*row['sponsor_rate']:.1f}% sponsored ({row['n_sponsored']:.0f}/{row['n_videos']} videos), child brands: {row['n_child_brands']:.0f}")
    
    # ---- COLLABORATION NETWORK ----
    print("\n" + "="*60)
    print("COLLABORATION NETWORK ANALYSIS")
    print("="*60)
    
    collab_df, channel_info_dict = detect_collaborations(raw_dir, desc_dir)
    print(f"\nTotal collaboration instances detected: {len(collab_df)}")
    print(f"Unique collaborations (by video): {collab_df.drop_duplicates(subset=['channel_a','channel_b','video_id']).shape[0]}")
    
    if not collab_df.empty:
        network_df, centrality_df = build_network(collab_df, channel_info_dict)
        
        print(f"\nUnique channel pairs: {len(network_df)}")
        print(f"Family-Family pairs: {network_df['both_family'].sum()}")
        
        print("\n--- TOP COLLABORATION PAIRS ---")
        for _, row in network_df.head(20).iterrows():
            marker = " [FAMILY-FAMILY]" if row['both_family'] else ""
            print(f"  {row['channel_a']:20s} <-> {row['channel_b']:20s}: {row['collaboration_count']} videos{marker}")
        
        print("\n--- MOST CONNECTED CHANNELS (Degree Centrality) ---")
        for _, row in centrality_df.head(15).iterrows():
            cat_label = "[F]" if row['category'] == 'family' else "[A]"
            print(f"  {cat_label} {row['channel']:25s}: {row['degree']} partners (family: {row['family_partners']}, adult: {row['adult_partners']})")
        
        # Save results
        network_df.to_csv(os.path.join(output_dir, 'collaboration_network.csv'), index=False)
        centrality_df.to_csv(os.path.join(output_dir, 'network_centrality.csv'), index=False)
    
    # Save sponsorship results
    ch_sponsor.to_csv(os.path.join(output_dir, 'sponsorship_by_channel.csv'), index=False)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: COMMERCIAL EXPLOITATION INDICATORS")
    print("="*60)
    print(f"\n1. Sponsorship rate: Family {100*family_sp['is_sponsored'].mean():.1f}% vs Adult {100*adult_sp['is_sponsored'].mean():.1f}%")
    print(f"2. Child brand targeting: {family_sp['n_child_brands'].sum()} mentions in family channels")
    if not collab_df.empty:
        fam_fam = network_df[network_df['both_family']]['collaboration_count'].sum()
        total_collab = network_df['collaboration_count'].sum()
        print(f"3. Family-Family collaborations: {fam_fam}/{total_collab} ({100*fam_fam/max(total_collab,1):.0f}%)")
        
        # Check for "hub" channels that connect many family channels
        family_hubs = centrality_df[(centrality_df['category'] == 'family') & (centrality_df['family_partners'] >= 3)]
        if len(family_hubs) > 0:
            print(f"\n4. POTENTIAL PUPPET MASTERS (family channels connected to 3+ other family channels):")
            for _, row in family_hubs.iterrows():
                print(f"   {row['channel']:25s}: connected to {row['family_partners']} family channels")
    
    print("\nResults saved to:", output_dir)
