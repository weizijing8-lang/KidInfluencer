"""
Build a labeled dataset of kidfluencer video titles for manipulation detection.
Inspired by fake news detection methodology:
- Define categories specific to kidfluencer content
- Use rule-based + LLM labels as silver standard
- Prepare for ML/DL classification experiments

Categories (kidfluencer-specific):
1. STAGED_CONFLICT: Fake theft, pranks on children, fake fights ("Who STOLE my iPhone")
2. FAKE_EMERGENCY: Fake 911 calls, fake injuries, fake danger ("CALLED THE POLICE")
3. EMOTIONAL_BAIT: Crying, screaming, extreme reactions for clicks ("SHE CRIED SO HARD")
4. CHALLENGE_DARE: Challenges that involve discomfort/risk ("24 HOUR CHALLENGE")
5. DECEPTION_NARRATIVE: Lies, secrets, betrayal storylines ("My mom LIED to me")
6. NEUTRAL: Normal content without manipulation tactics
"""
import pandas as pd
import numpy as np
import re
import json
import os

# Load ONLY family/kid channel data
v4 = pd.read_csv('data/results_v4/full_results_v4.csv')
family = v4[v4['channel_category'] == 'family'].copy()
print(f"Family channel videos: {len(family)} from {family['channel_short_name'].nunique()} channels")

# ============================================================
# STEP 1: Define manipulation categories with keyword patterns
# ============================================================

categories = {
    'STAGED_CONFLICT': {
        'keywords': ['stole', 'stolen', 'stealing', 'thief', 'prank', 'pranked', 'pranking',
                     'caught', 'busted', 'exposed', 'fight', 'fighting', 'destroy', 'destroyed',
                     'broke', 'smash', 'smashed', 'snitch', 'betrayed', 'revenge'],
        'patterns': [
            r'who\s+(stole|took|broke|ate)',
            r'(caught|busted)\s+(him|her|them|my)',
            r'prank\s+(on|gone|war)',
            r'(destroy|broke|smash)\w*\s+(my|his|her)',
        ]
    },
    'FAKE_EMERGENCY': {
        'keywords': ['911', 'police', 'arrested', 'jail', 'hospital', 'ambulance',
                     'emergency', 'kidnapped', 'missing', 'lost', 'disappeared'],
        'patterns': [
            r'call(ed|ing)?\s+(the\s+)?(police|911|cops)',
            r'went\s+to\s+(jail|hospital|prison)',
            r'(got|was)\s+(arrested|kidnapped|lost)',
        ]
    },
    'EMOTIONAL_BAIT': {
        'keywords': ['cried', 'crying', 'tears', 'scream', 'screaming', 'angry', 'mad',
                     'heartbroken', 'devastated', 'freaked out', 'meltdown', 'tantrum',
                     'worst day', 'so sad', 'broke down'],
        'patterns': [
            r'(she|he|i|we)\s+(cried|screamed|freaked)',
            r'(made|got)\s+(him|her|them|me)\s+(cry|scream|mad)',
            r'emotional\s+(reaction|moment|surprise)',
            r'(worst|saddest|scariest)\s+(day|moment|thing)',
        ]
    },
    'CHALLENGE_DARE': {
        'keywords': ['challenge', 'dare', 'dared', '24 hour', '48 hour', 'last to leave',
                     'extreme', 'impossible', 'try not to', 'don\'t', 'never'],
        'patterns': [
            r'\d+\s*hour\s*(challenge|in)',
            r'(last|first)\s+to\s+(leave|stop|eat|laugh)',
            r'try\s+not\s+to',
            r'(extreme|impossible|insane)\s+(challenge|dare)',
            r"don'?t\s+(try|do|eat|touch)",
        ]
    },
    'DECEPTION_NARRATIVE': {
        'keywords': ['lie', 'lied', 'lying', 'secret', 'secrets', 'hidden', 'truth',
                     'fake', 'trick', 'tricked', 'fooled', 'not what it seems',
                     'you won\'t believe', 'shocking truth'],
        'patterns': [
            r'(my|the)\s+(secret|lie|truth)',
            r'(lied|lying)\s+(to|about)',
            r'(trick|fool|deceiv)\w*\s+(my|him|her)',
            r'(hidden|secret)\s+(camera|room|message)',
            r'the\s+truth\s+(about|behind|is)',
        ]
    }
}

# ============================================================
# STEP 2: Apply rule-based classification
# ============================================================

def classify_title(title):
    """Classify a title into manipulation categories."""
    if pd.isna(title):
        return 'NEUTRAL', 0, []
    
    title_lower = str(title).lower()
    matches = {}
    
    for cat, rules in categories.items():
        score = 0
        matched_keywords = []
        
        # Keyword matching
        for kw in rules['keywords']:
            if kw in title_lower:
                score += 1
                matched_keywords.append(kw)
        
        # Pattern matching (stronger signal)
        for pattern in rules['patterns']:
            if re.search(pattern, title_lower):
                score += 2
                matched_keywords.append(f'pattern:{pattern[:20]}')
        
        if score > 0:
            matches[cat] = (score, matched_keywords)
    
    if not matches:
        return 'NEUTRAL', 0, []
    
    # Return the highest-scoring category
    best_cat = max(matches, key=lambda k: matches[k][0])
    return best_cat, matches[best_cat][0], matches[best_cat][1]

# Apply classification
family['manipulation_category'] = family['title'].apply(lambda t: classify_title(t)[0])
family['manipulation_score'] = family['title'].apply(lambda t: classify_title(t)[1])
family['matched_keywords'] = family['title'].apply(lambda t: json.dumps(classify_title(t)[2]))

# Also compute additional title features (for ML)
def extract_title_features(title):
    if pd.isna(title):
        return {}
    title = str(title)
    features = {
        'title_length': len(title),
        'word_count': len(title.split()),
        'caps_ratio': sum(1 for c in title if c.isupper()) / max(len(title), 1),
        'exclamation_count': title.count('!'),
        'question_count': title.count('?'),
        'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', title)),
        'has_emoji': bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', title)),
        'has_ellipsis': '...' in title,
        'has_asterisk': '*' in title,
        'num_count': len(re.findall(r'\d+', title)),
    }
    return features

features_list = family['title'].apply(extract_title_features)
features_df = pd.DataFrame(features_list.tolist())
family = pd.concat([family.reset_index(drop=True), features_df], axis=1)

# ============================================================
# STEP 3: Summary statistics
# ============================================================
print(f"\n=== Manipulation Category Distribution ===")
cat_dist = family['manipulation_category'].value_counts()
print(cat_dist)
print(f"\nPercentages:")
print((cat_dist / len(family) * 100).round(1))

print(f"\n=== By Channel ===")
channel_manip = family.groupby('channel_short_name').agg(
    n_videos=('title', 'count'),
    manipulation_rate=('manipulation_category', lambda x: (x != 'NEUTRAL').mean()),
    staged_conflict_rate=('manipulation_category', lambda x: (x == 'STAGED_CONFLICT').mean()),
    fake_emergency_rate=('manipulation_category', lambda x: (x == 'FAKE_EMERGENCY').mean()),
    emotional_bait_rate=('manipulation_category', lambda x: (x == 'EMOTIONAL_BAIT').mean()),
    challenge_dare_rate=('manipulation_category', lambda x: (x == 'CHALLENGE_DARE').mean()),
    deception_rate=('manipulation_category', lambda x: (x == 'DECEPTION_NARRATIVE').mean()),
    mean_views=('viewCount', 'mean'),
    median_views=('viewCount', 'median'),
).sort_values('manipulation_rate', ascending=False)

print(channel_manip[['n_videos', 'manipulation_rate', 'staged_conflict_rate', 
                      'emotional_bait_rate', 'challenge_dare_rate', 'mean_views']].head(15).to_string())

# ============================================================
# STEP 4: View boost per category
# ============================================================
print(f"\n=== View Boost by Manipulation Category ===")
neutral_median = family[family['manipulation_category'] == 'NEUTRAL']['viewCount'].median()
print(f"NEUTRAL baseline: median views = {neutral_median:,.0f}")
for cat in ['STAGED_CONFLICT', 'FAKE_EMERGENCY', 'EMOTIONAL_BAIT', 'CHALLENGE_DARE', 'DECEPTION_NARRATIVE']:
    cat_vids = family[family['manipulation_category'] == cat]
    if len(cat_vids) > 10:
        cat_median = cat_vids['viewCount'].median()
        boost = cat_median / neutral_median - 1
        print(f"  {cat}: median={cat_median:,.0f}, boost={boost:+.1%}, n={len(cat_vids)}")

# ============================================================
# STEP 5: Example titles per category
# ============================================================
print(f"\n=== Example Titles ===")
for cat in ['STAGED_CONFLICT', 'FAKE_EMERGENCY', 'EMOTIONAL_BAIT', 'CHALLENGE_DARE', 'DECEPTION_NARRATIVE']:
    cat_vids = family[family['manipulation_category'] == cat].sort_values('manipulation_score', ascending=False)
    print(f"\n--- {cat} (n={len(cat_vids)}) ---")
    for _, row in cat_vids.head(5).iterrows():
        print(f"  [{row['manipulation_score']}] {str(row['title'])[:80]} ({row['channel_short_name']})")

# ============================================================
# STEP 6: Save dataset for ML experiments
# ============================================================
os.makedirs('data/manipulation_detection', exist_ok=True)

# Save full labeled dataset
family[['id', 'title', 'channel_short_name', 'viewCount', 'likeCount', 'commentCount',
        'publishedAt', 'manipulation_category', 'manipulation_score',
        'title_length', 'word_count', 'caps_ratio', 'exclamation_count',
        'question_count', 'all_caps_words', 'has_emoji', 'has_ellipsis',
        'has_asterisk', 'num_count']].to_csv('data/manipulation_detection/labeled_titles.csv', index=False)

print(f"\n=== Dataset saved ===")
print(f"Total: {len(family)} videos")
print(f"Manipulative: {(family['manipulation_category'] != 'NEUTRAL').sum()} ({(family['manipulation_category'] != 'NEUTRAL').mean():.1%})")
print(f"Saved to: data/manipulation_detection/labeled_titles.csv")
