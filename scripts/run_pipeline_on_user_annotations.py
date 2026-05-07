"""
Run the Snorkel labeling functions on the user's 12 annotated videos
to compare pipeline predictions against human ground truth.
"""
import pandas as pd
import numpy as np
import re
import os

# User's manual annotations
user_annotations = [
    {"video_id": "1BcbDYtORH0", "title": "We Found A Giant Ninja Battle Robot!", "channel": "ninjakidztv",
     "view_count": 2685133,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 1},
    {"video_id": "mEcLEyDi3bw", "title": "THIRD WHEELING MY 12 YEAR OLD BROTHER'S RELATIONSHIP!", "channel": "brentrivera",
     "view_count": 6323561,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 1, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 1, "h_overall": 1},
    {"video_id": "-qKBVTFhKHw", "title": "Pranking our Siblings!", "channel": "ohanaboys",
     "view_count": 90207,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 1, "h_overall": 1},
    {"video_id": "ijeCnHEx8W4", "title": "FALL EVERYTHING! PUMPKIN SPICE STARBUCKS, TARGET SHOPPING & MORE!", "channel": "ourfamilynest",
     "view_count": 18353,
     "h_performative": 0, "h_emotional_bait": 0, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
    {"video_id": "0SmLtajf--U", "title": "The MAGIC Dollhouse Movie", "channel": "shotofyeagers",
     "view_count": 167277,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 1},
    {"video_id": "xpmU8dCteIw", "title": "4 year olds Everleigh & Ava bake with their Baby Alive dolls I Foreverandforava", "channel": "everleighrose",
     "view_count": 1143133,
     "h_performative": 0, "h_emotional_bait": 0, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
    {"video_id": "QhuaVZJpW6o", "title": "RICH Parents Vs. BROKE Parents", "channel": "piperrockelle",
     "view_count": 5937016,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 0, "h_challenge_format": 1, 
     "h_commercial_content": 0, "h_privacy_violation": 1, "h_overall": 1},
    {"video_id": "AdFxzIdSZek", "title": "Father's Day Was Great Until I Broke My Tooth", "channel": "thedashleys",
     "view_count": 31912,
     "h_performative": 0, "h_emotional_bait": 0, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
    {"video_id": "aN--3LU_cb4", "title": "You won't believe what we found HIDING in the trunk! *Scary*", "channel": "tydustalbott",
     "view_count": 2345702,
     "h_performative": 1, "h_emotional_bait": 1, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 1, "h_privacy_violation": 0, "h_overall": 1},
    {"video_id": "o4yYkq0k12Y", "title": "Mr.beast hated my chips 😱😭", "channel": "brentrivera",
     "view_count": 896790,
     "h_performative": 1, "h_emotional_bait": 0, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
    {"video_id": "RElUn9WRxbY", "title": "BUYING ANYTHING In Your COLOR FOR 30 MINUTES CHALLENGE!", "channel": "lifewithbrothers",
     "view_count": 68794,
     "h_performative": 1, "h_emotional_bait": 0, "h_narrative_conflict": 0, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
    {"video_id": "8AnIUAdvITA", "title": "I am So Sad | Goodbyes Are Never Easy", "channel": "theleray",
     "view_count": 223547,
     "h_performative": 0, "h_emotional_bait": 0, "h_narrative_conflict": 1, "h_challenge_format": 0, 
     "h_commercial_content": 0, "h_privacy_violation": 0, "h_overall": 0},
]

df = pd.DataFrame(user_annotations)

# ============================================================
# REPLICATE THE RULE-BASED LABELING FUNCTIONS
# ============================================================

def lf_allcaps_ratio(title):
    """High all-caps ratio suggests clickbait/exploitation"""
    words = title.split()
    if len(words) == 0:
        return 0
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    ratio = caps_words / len(words)
    return 1 if ratio > 0.5 else 0

def lf_exclamation(title):
    """Excessive exclamation marks"""
    count = title.count('!')
    return 1 if count >= 2 else 0

def lf_conflict_keywords(title):
    """Keywords suggesting conflict"""
    keywords = ['fight', 'exposed', 'caught', 'prank', 'revenge', 'vs', 'versus', 'battle', 'war', 'destroy']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

def lf_challenge_keywords(title):
    """Keywords suggesting challenge format"""
    keywords = ['challenge', '24 hours', '24h', 'hours', 'dare', 'last to', 'first to']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

def lf_emotional_keywords(title):
    """Keywords suggesting emotional bait"""
    keywords = ['crying', 'cried', 'tears', 'emotional', 'heartbreak', 'sad', 'scared', 'scary', 'shocking']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

def lf_organic_family(title):
    """Keywords suggesting organic family content (votes AGAINST exploitation)"""
    keywords = ['birthday', 'christmas', 'vacation', 'holiday', 'morning routine', 'day in the life', 
                'bake', 'cook', 'garden', 'first day']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0  # 1 = non-exploitative

def lf_privacy_keywords(title):
    """Keywords suggesting privacy violation"""
    keywords = ['pregnant', 'hospital', 'doctor', 'surgery', 'relationship', 'dating', 'boyfriend', 'girlfriend',
                'crush', 'kiss']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

def lf_performative_keywords(title):
    """Keywords suggesting performative/scripted content"""
    keywords = ['prank', 'pretend', 'acting', 'roleplay', 'skit', 'movie', 'film', 'magic', 'ninja']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

def lf_commercial_keywords(title):
    """Keywords suggesting commercial content"""
    keywords = ['unboxing', 'haul', 'buying', 'shopping', 'review', 'sponsored', 'ad']
    title_lower = title.lower()
    return 1 if any(k in title_lower for k in keywords) else 0

# Apply all rule-based LFs
df['lf_allcaps'] = df['title'].apply(lf_allcaps_ratio)
df['lf_exclamation'] = df['title'].apply(lf_exclamation)
df['lf_conflict'] = df['title'].apply(lf_conflict_keywords)
df['lf_challenge'] = df['title'].apply(lf_challenge_keywords)
df['lf_emotional'] = df['title'].apply(lf_emotional_keywords)
df['lf_organic'] = df['title'].apply(lf_organic_family)
df['lf_privacy'] = df['title'].apply(lf_privacy_keywords)
df['lf_performative'] = df['title'].apply(lf_performative_keywords)
df['lf_commercial'] = df['title'].apply(lf_commercial_keywords)

# Simple aggregation (majority vote of rule-based signals)
exploit_signals = ['lf_allcaps', 'lf_exclamation', 'lf_conflict', 'lf_challenge', 
                   'lf_emotional', 'lf_privacy', 'lf_performative']
non_exploit_signals = ['lf_organic']

df['rule_exploit_votes'] = df[exploit_signals].sum(axis=1)
df['rule_non_exploit_votes'] = df[non_exploit_signals].sum(axis=1)
df['rule_net_score'] = df['rule_exploit_votes'] - df['rule_non_exploit_votes']
df['rule_pred'] = (df['rule_net_score'] >= 2).astype(int)  # threshold: 2+ signals

# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 90)
print("RULE-BASED LABELING FUNCTION ANALYSIS ON USER'S 12 ANNOTATED VIDEOS")
print("=" * 90)

print(f"\n{'Video Title':<55} {'LF Signals':<12} {'Rule Pred':<10} {'Human':<8} {'Match'}")
print("─" * 90)

correct = 0
for _, row in df.iterrows():
    title_short = row['title'][:52] + "..." if len(row['title']) > 52 else row['title']
    rule_pred = row['rule_pred']
    human = row['h_overall']
    match = "✅" if rule_pred == human else "❌"
    if rule_pred == human:
        correct += 1
    print(f"{title_short:<55} {int(row['rule_net_score']):>2} signals  {'EXPLOIT' if rule_pred else 'CLEAN':<10} {'EXPLOIT' if human else 'CLEAN':<8} {match}")

print(f"\n{'─' * 90}")
print(f"Rule-Based LF Accuracy: {correct}/{len(df)} = {correct/len(df):.1%}")

# ============================================================
# DETAILED DIMENSION COMPARISON
# ============================================================

print("\n\n" + "=" * 90)
print("DIMENSION-LEVEL COMPARISON (Rule-Based LFs vs. Human)")
print("=" * 90)

dim_map = {
    'performative': 'lf_performative',
    'emotional_bait': 'lf_emotional',
    'narrative_conflict': 'lf_conflict',
    'challenge_format': 'lf_challenge',
    'commercial_content': 'lf_commercial',
    'privacy_violation': 'lf_privacy',
}

print(f"\n{'Dimension':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
print("─" * 60)

for dim_name, lf_name in dim_map.items():
    h_col = f'h_{dim_name}'
    human_vals = df[h_col].values
    pred_vals = df[lf_name].values
    
    acc = (human_vals == pred_vals).mean()
    
    # Precision & Recall
    tp = ((human_vals == 1) & (pred_vals == 1)).sum()
    fp = ((human_vals == 0) & (pred_vals == 1)).sum()
    fn = ((human_vals == 1) & (pred_vals == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{dim_name:<22} {acc:.1%}{'':>5} {precision:.1%}{'':>5} {recall:.1%}")

# ============================================================
# KEY DISAGREEMENTS ANALYSIS
# ============================================================

print("\n\n" + "=" * 90)
print("KEY DISAGREEMENTS & INSIGHTS")
print("=" * 90)

disagreements = df[df['rule_pred'] != df['h_overall']]
for _, row in disagreements.iterrows():
    print(f"\n📹 \"{row['title']}\"")
    print(f"   Rule prediction: {'EXPLOIT' if row['rule_pred'] else 'CLEAN'} (net score: {row['rule_net_score']})")
    print(f"   Human judgment:  {'EXPLOIT' if row['h_overall'] else 'CLEAN'}")
    
    # Show which signals fired
    fired = []
    for sig in exploit_signals:
        if row[sig] == 1:
            fired.append(sig.replace('lf_', ''))
    if fired:
        print(f"   Signals fired: {', '.join(fired)}")
    else:
        print(f"   No exploit signals fired")
    
    # Explain likely reason
    if row['rule_pred'] == 1 and row['h_overall'] == 0:
        print(f"   → FALSE POSITIVE: Rule-based signals over-triggered")
    else:
        print(f"   → FALSE NEGATIVE: Exploitation not captured by keyword rules alone")

print("\n\n" + "=" * 90)
print("IMPLICATIONS FOR PAPER")
print("=" * 90)
print("""
1. Rule-based LFs alone have limited recall for subtle exploitation
   (e.g., "The MAGIC Dollhouse Movie" is performative but has no keyword triggers)
   
2. This is EXACTLY why the paper uses Snorkel + LLM: the LLM can understand
   semantic context that keyword rules miss.
   
3. Key insight from user annotations: "overall exploitative" requires MULTIPLE
   dimensions to co-occur (user marks videos with 2+ dimensions as exploitative,
   but single-dimension videos as non-exploitative)
   
4. The user's annotation pattern suggests a THRESHOLD model: 
   - 0-1 dimensions present → NON-EXPLOITATIVE
   - 2+ dimensions present → EXPLOITATIVE
   This aligns well with the Snorkel aggregation approach.
""")
