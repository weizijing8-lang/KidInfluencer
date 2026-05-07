import pandas as pd
import numpy as np

# User's manual annotations
user_annotations = [
    {"video_id": "1BcbDYtORH0", "title": "We Found A Giant Ninja Battle Robot!", "channel": "ninjakidztv",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 1},
    {"video_id": "mEcLEyDi3bw", "title": "THIRD WHEELING MY 12 YEAR OLD BROTHER'S RELATIONSHIP!", "channel": "brentrivera",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 1, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 1, "overall": 1},
    {"video_id": "-qKBVTFhKHw", "title": "Pranking our Siblings!", "channel": "ohanaboys",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 1, "overall": 1},
    {"video_id": "ijeCnHEx8W4", "title": "FALL EVERYTHING! PUMPKIN SPICE STARBUCKS, TARGET SHOPPING & MORE!", "channel": "ourfamilynest",
     "performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
    {"video_id": "0SmLtajf--U", "title": "The MAGIC Dollhouse Movie", "channel": "shotofyeagers",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 1},
    {"video_id": "xpmU8dCteIw", "title": "4 year olds Everleigh & Ava bake with their Baby Alive dolls", "channel": "everleighrose",
     "performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
    {"video_id": "QhuaVZJpW6o", "title": "RICH Parents Vs. BROKE Parents", "channel": "piperrockelle",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 1, "commercial_content": 0, "privacy_violation": 1, "overall": 1},
    {"video_id": "AdFxzIdSZek", "title": "Father's Day Was Great Until I Broke My Tooth", "channel": "thedashleys",
     "performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
    {"video_id": "aN--3LU_cb4", "title": "You won't believe what we found HIDING in the trunk! *Scary*", "channel": "tydustalbott",
     "performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 1, "privacy_violation": 0, "overall": 1},
    {"video_id": "o4yYkq0k12Y", "title": "Mr.beast hated my chips", "channel": "brentrivera",
     "performative": 1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
    {"video_id": "RElUn9WRxbY", "title": "BUYING ANYTHING In Your COLOR FOR 30 MINUTES CHALLENGE!", "channel": "lifewithbrothers",
     "performative": 1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
    {"video_id": "8AnIUAdvITA", "title": "I am So Sad | Goodbyes Are Never Easy", "channel": "theleray",
     "performative": 0, "emotional_bait": 0, "narrative_conflict": 1, "challenge_format": 0, "commercial_content": 0, "privacy_violation": 0, "overall": 0},
]

user_df = pd.DataFrame(user_annotations)

# Load Snorkel predictions
snorkel_df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results/videos_with_exploitation_scores.csv')

# Also load the LLM dimension labels
llm_df = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/llm_content_labels.csv')

print("=" * 80)
print("COMPARISON: User Annotations vs. Snorkel Pipeline Predictions")
print("=" * 80)

# Merge user annotations with Snorkel scores
merged = user_df.merge(snorkel_df[['id', 'performative', 'emotional_bait', 'narrative_conflict', 
                                    'challenge_format', 'commercial_content', 'exploitation_score', 'exploitation_pred']],
                       left_on='video_id', right_on='id', how='left', suffixes=('_human', '_snorkel'))

# Check which videos are in the Snorkel dataset
print(f"\nVideos in user annotations: {len(user_df)}")
print(f"Videos found in Snorkel dataset: {merged['id'].notna().sum()}")
print(f"Videos NOT in Snorkel dataset: {merged['id'].isna().sum()}")

# Also check LLM labels
llm_merged = user_df.merge(llm_df, left_on='video_id', right_on='video_id', how='left', suffixes=('_human', '_llm'))
print(f"\nLLM labels columns: {[c for c in llm_df.columns if 'video_id' in c or 'performative' in c or 'emotional' in c or 'narrative' in c or 'challenge' in c or 'commercial' in c or 'privacy' in c]}")
print(f"LLM dataset shape: {llm_df.shape}")
print(f"LLM columns: {llm_df.columns.tolist()}")

print("\n" + "=" * 80)
print("DETAILED VIDEO-BY-VIDEO COMPARISON")
print("=" * 80)

for _, row in merged.iterrows():
    print(f"\n{'─' * 70}")
    print(f"📹 {row['title']}")
    print(f"   Channel: {row['channel']} | Video ID: {row['video_id']}")
    
    if pd.isna(row.get('id')):
        print(f"   ⚠️  NOT IN SNORKEL SAMPLE (may not have been in stratified sample)")
        print(f"   Human Overall: {'EXPLOITATIVE' if row['overall'] else 'NON-EXPLOITATIVE'}")
    else:
        human_overall = row['overall']
        snorkel_pred = row['exploitation_pred']
        snorkel_score = row['exploitation_score']
        
        match = "✅ MATCH" if human_overall == snorkel_pred else "❌ MISMATCH"
        print(f"   {match}")
        print(f"   Human Overall:  {'EXPLOITATIVE' if human_overall else 'NON-EXPLOITATIVE'}")
        print(f"   Snorkel Score:  {snorkel_score:.3f} → Pred: {'EXPLOITATIVE' if snorkel_pred else 'NON-EXPLOITATIVE'}")
        
        # Dimension comparison
        dims = ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content']
        print(f"\n   Dimension Comparison:")
        print(f"   {'Dimension':<22} {'Human':<8} {'Snorkel':<8} {'Match'}")
        print(f"   {'─' * 50}")
        for dim in dims:
            h = row.get(f'{dim}_human', row.get(dim))
            s = row.get(f'{dim}_snorkel')
            if pd.notna(s):
                m = "✅" if h == s else "❌"
                print(f"   {dim:<22} {int(h):<8} {int(s):<8} {m}")
            else:
                print(f"   {dim:<22} {int(h) if pd.notna(h) else '?':<8} {'N/A':<8}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

matched = merged[merged['id'].notna()]
if len(matched) > 0:
    # Overall accuracy
    correct = (matched['overall'] == matched['exploitation_pred']).sum()
    total = len(matched)
    print(f"\nOverall Accuracy (binary): {correct}/{total} = {correct/total:.1%}")
    
    # Per-dimension accuracy
    dims = ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content']
    print(f"\nPer-Dimension Accuracy:")
    for dim in dims:
        h_col = f'{dim}_human' if f'{dim}_human' in matched.columns else dim
        s_col = f'{dim}_snorkel'
        if s_col in matched.columns:
            valid = matched[[h_col, s_col]].dropna()
            if len(valid) > 0:
                acc = (valid[h_col] == valid[s_col]).mean()
                print(f"   {dim:<22}: {acc:.1%} ({(valid[h_col] == valid[s_col]).sum()}/{len(valid)})")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)
print("""
Note: The Snorkel stratified sample contains 2,306 videos (from 4,685 total).
Not all 12 user-annotated videos may appear in the sample.
Videos not in the sample cannot be compared directly.
""")
