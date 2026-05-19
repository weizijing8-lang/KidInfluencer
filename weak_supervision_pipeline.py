"""
Proper Snorkel-style Weak Supervision Pipeline for KidInfluencer Exploitation Detection
=========================================================================================
This implements a genuine weak supervision framework with:
- Multiple heterogeneous Labeling Functions (LFs) per dimension
- Snorkel's LabelModel for learning LF accuracies and correlations
- Per-dimension label models (not just a single overall score)
- Proper abstention handling (ABSTAIN = -1)
"""

import pandas as pd
import numpy as np
import json
import glob
import re
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Constants
ABSTAIN = -1
EXPLOIT = 1
CLEAN = 0

print("=" * 70)
print("WEAK SUPERVISION PIPELINE FOR KIDFLUENCER EXPLOITATION DETECTION")
print("=" * 70)

# ============================================================
# STEP 1: Load and merge all data sources
# ============================================================
print("\n[1/6] Loading data...")

# Main classified dataset (v4)
v4 = pd.read_csv('analysis_discovery/snorkel_results_v4/classified_videos_v4.csv')
print(f"  V4 dataset: {len(v4)} videos, {v4.channel_short_name.nunique()} channels")

# Vision results (continuous VLM scores)
vision = pd.read_csv('data/vision_results/vision_classifications.csv')
vision = vision.rename(columns={'video_id': 'id'})
vision_cols = ['id', 'performative_labor', 'emotional_bait', 'narrative_conflict',
               'challenge_format', 'commercial_content', 'privacy_violation', 'overall_exploitative']
vision = vision[vision_cols].rename(columns={
    'performative_labor': 'vlm_performative',
    'emotional_bait': 'vlm_emotional',
    'narrative_conflict': 'vlm_narrative',
    'challenge_format': 'vlm_challenge',
    'commercial_content': 'vlm_commercial',
    'privacy_violation': 'vlm_privacy',
    'overall_exploitative': 'vlm_overall'
})
print(f"  VLM scores: {len(vision)} videos")

# Descriptions
all_descriptions = {}
for f in glob.glob('data/descriptions/*.json'):
    with open(f) as fh:
        all_descriptions.update(json.load(fh))
print(f"  Descriptions available: {len(all_descriptions)}")

# Durations
all_durations = {}
for f in glob.glob('data/durations/*.json'):
    with open(f) as fh:
        all_durations.update(json.load(fh))
print(f"  Durations available: {len(all_durations)}")

# Merge everything into one master dataframe
df = v4[['id', 'title', 'channel_short_name', 'viewCount',
          'performative_labor_llm', 'emotional_bait_llm', 'narrative_conflict_llm',
          'challenge_format_llm', 'commercial_content_llm', 'privacy_violation_llm',
          'publishedAt', 'channelId']].copy()

# Add VLM scores
df = df.merge(vision, on='id', how='left')

# Add descriptions
df['description'] = df['id'].map(lambda x: all_descriptions.get(x, {}).get('description', '') if x in all_descriptions else '')
df['tags'] = df['id'].map(lambda x: ' '.join(all_descriptions.get(x, {}).get('tags', [])) if x in all_descriptions else '')

# Add durations
df['duration_seconds'] = df['id'].map(lambda x: all_durations.get(x, {}).get('duration_seconds', np.nan) if x in all_durations else np.nan)

# Clean title - lowercase for matching
df['title_lower'] = df['title'].fillna('').str.lower()
df['desc_lower'] = df['description'].fillna('').str.lower()
df['tags_lower'] = df['tags'].fillna('').str.lower()

print(f"\n  Master dataset: {len(df)} videos")
print(f"  With VLM scores: {df['vlm_performative'].notna().sum()}")
print(f"  With descriptions: {(df['description'] != '').sum()}")
print(f"  With durations: {df['duration_seconds'].notna().sum()}")

# ============================================================
# STEP 2: Define Labeling Functions for each dimension
# ============================================================
print("\n[2/6] Defining Labeling Functions...")

# ---- DIMENSION 1: PERFORMATIVE LABOR ----
def lf_llm_performative(x):
    """LLM text classification for performative labor"""
    return EXPLOIT if x.performative_labor_llm == 1 else CLEAN

def lf_vlm_performative(x):
    """VLM vision classification for performative labor (high confidence)"""
    if pd.isna(x.vlm_performative):
        return ABSTAIN
    if x.vlm_performative >= 0.7:
        return EXPLOIT
    elif x.vlm_performative <= 0.2:
        return CLEAN
    return ABSTAIN

def lf_vlm_performative_medium(x):
    """VLM vision classification for performative labor (medium confidence)"""
    if pd.isna(x.vlm_performative):
        return ABSTAIN
    if x.vlm_performative >= 0.5:
        return EXPLOIT
    elif x.vlm_performative <= 0.3:
        return CLEAN
    return ABSTAIN

def lf_scripted_keywords(x):
    """Title keywords suggesting scripted/planned content"""
    keywords = ['challenge', 'prank', '24 hours', '24hrs', 'last to', 'first to',
                'try not to', 'don\'t', 'vs', 'battle', 'competition', 'race',
                'game', 'quiz', 'test', 'experiment']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_organic_keywords(x):
    """Title keywords suggesting organic/unscripted content"""
    keywords = ['vlog', 'day in', 'morning routine', 'grocery', 'shopping',
                'haul', 'update', 'q&a', 'get ready', 'what i eat',
                'clean with me', 'organize', 'tour']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return CLEAN
    return ABSTAIN

def lf_roleplay_title(x):
    """Title patterns suggesting roleplay/acting"""
    patterns = [r'pretend', r'play.*house', r'role\s*play', r'act.*like',
                r'spy', r'undercover', r'sneak', r'hide.*seek', r'escape']
    title = x.title_lower
    if any(re.search(p, title) for p in patterns):
        return EXPLOIT
    return ABSTAIN

def lf_desc_scripted(x):
    """Description mentions scripted elements"""
    desc = x.desc_lower
    if not desc:
        return ABSTAIN
    script_indicators = ['subscribe', 'like and subscribe', 'new video every',
                         'follow us on', 'business inquiries', 'collab']
    # These are generic - look for actual content indicators
    content_indicators = ['challenge', 'prank', 'skit', 'sketch', 'acted',
                          'scripted', 'directed by', 'produced by', 'cast:']
    if any(kw in desc for kw in content_indicators):
        return EXPLOIT
    return ABSTAIN

def lf_duration_long_challenge(x):
    """Very long videos (>25 min) more likely to be challenge/performative"""
    if pd.isna(x.duration_seconds):
        return ABSTAIN
    if x.duration_seconds > 1500:  # >25 min
        return EXPLOIT
    elif x.duration_seconds < 120:  # <2 min (shorts)
        return CLEAN
    return ABSTAIN

# ---- DIMENSION 2: EMOTIONAL BAIT ----
def lf_llm_emotional(x):
    """LLM text classification for emotional bait"""
    return EXPLOIT if x.emotional_bait_llm == 1 else ABSTAIN  # Only positive signal

def lf_vlm_emotional(x):
    """VLM vision classification for emotional bait"""
    if pd.isna(x.vlm_emotional):
        return ABSTAIN
    if x.vlm_emotional >= 0.6:
        return EXPLOIT
    elif x.vlm_emotional <= 0.15:
        return CLEAN
    return ABSTAIN

def lf_emotional_title_caps(x):
    """ALL CAPS in title suggests emotional clickbait"""
    title = x.title
    if not isinstance(title, str):
        return ABSTAIN
    # Count ratio of uppercase words
    words = title.split()
    if len(words) < 3:
        return ABSTAIN
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    if caps_words / len(words) > 0.5:
        return EXPLOIT
    return ABSTAIN

def lf_emotional_punctuation(x):
    """Excessive punctuation (!!!, ???) suggests emotional bait"""
    title = str(x.title)
    excl_count = title.count('!')
    quest_count = title.count('?')
    if excl_count >= 3 or quest_count >= 3 or (excl_count + quest_count) >= 4:
        return EXPLOIT
    return ABSTAIN

def lf_emotional_keywords(x):
    """Emotional keywords in title"""
    keywords = ['crying', 'cried', 'tears', 'emotional', 'heartbreaking',
                'shocking', 'scared', 'terrified', 'angry', 'furious',
                'devastated', 'broke down', 'freaked out', 'screaming',
                'panic', 'meltdown', 'tantrum']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_emotional_emojis(x):
    """Emotional emojis in title suggesting bait"""
    title = str(x.title)
    emotional_emojis = ['😱', '😭', '😢', '😡', '🤯', '😳', '💔', '😰', '🥺', '😤']
    count = sum(title.count(e) for e in emotional_emojis)
    if count >= 2:
        return EXPLOIT
    return ABSTAIN

# ---- DIMENSION 3: NARRATIVE CONFLICT ----
def lf_llm_narrative(x):
    """LLM text classification for narrative conflict"""
    return EXPLOIT if x.narrative_conflict_llm == 1 else ABSTAIN

def lf_vlm_narrative(x):
    """VLM vision classification for narrative conflict"""
    if pd.isna(x.vlm_narrative):
        return ABSTAIN
    if x.vlm_narrative >= 0.5:
        return EXPLOIT
    elif x.vlm_narrative <= 0.1:
        return CLEAN
    return ABSTAIN

def lf_conflict_keywords(x):
    """Title keywords suggesting manufactured conflict"""
    keywords = ['fight', 'argument', 'broke up', 'divorce', 'kicked out',
                'caught', 'cheating', 'betrayed', 'revenge', 'exposed',
                'confronted', 'called out', 'drama', 'gone wrong',
                'not friends', 'hate', 'vs']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_prank_gone_wrong(x):
    """Prank/gone wrong pattern - manufactured drama"""
    title = x.title_lower
    if 'gone wrong' in title or 'prank' in title or 'backfired' in title:
        return EXPLOIT
    return ABSTAIN

def lf_clickbait_patterns(x):
    """Clickbait narrative patterns"""
    patterns = [r'you won\'t believe', r'what happens next', r'the truth about',
                r'finally revealed', r'i can\'t believe', r'never expected',
                r'plot twist', r'didn\'t expect']
    title = x.title_lower
    if any(re.search(p, title) for p in patterns):
        return EXPLOIT
    return ABSTAIN

# ---- DIMENSION 4: CHALLENGE FORMAT ----
def lf_llm_challenge(x):
    """LLM text classification for challenge format"""
    return EXPLOIT if x.challenge_format_llm == 1 else ABSTAIN

def lf_vlm_challenge(x):
    """VLM vision classification for challenge format"""
    if pd.isna(x.vlm_challenge):
        return ABSTAIN
    if x.vlm_challenge >= 0.5:
        return EXPLOIT
    elif x.vlm_challenge <= 0.1:
        return CLEAN
    return ABSTAIN

def lf_challenge_keywords(x):
    """Explicit challenge keywords in title"""
    keywords = ['challenge', '24 hour', '24hr', 'last to leave', 'first to',
                'try not to', 'impossible', 'extreme', 'ultimate',
                'world record', 'marathon', 'endurance']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_competition_format(x):
    """Competition/game format keywords"""
    keywords = ['vs', 'versus', 'battle', 'race', 'competition',
                'tournament', 'who can', 'winner gets', 'loser has to']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_duration_challenge(x):
    """Challenges tend to be longer videos"""
    if pd.isna(x.duration_seconds):
        return ABSTAIN
    # Very long videos with challenge keywords
    title = x.title_lower
    if x.duration_seconds > 1800 and any(kw in title for kw in ['hour', 'challenge', 'last']):
        return EXPLOIT
    return ABSTAIN

# ---- DIMENSION 5: COMMERCIAL CONTENT ----
def lf_llm_commercial(x):
    """LLM text classification for commercial content"""
    return EXPLOIT if x.commercial_content_llm == 1 else ABSTAIN

def lf_vlm_commercial(x):
    """VLM vision classification for commercial content"""
    if pd.isna(x.vlm_commercial):
        return ABSTAIN
    if x.vlm_commercial >= 0.5:
        return EXPLOIT
    elif x.vlm_commercial <= 0.1:
        return CLEAN
    return ABSTAIN

def lf_commercial_keywords_title(x):
    """Commercial keywords in title"""
    keywords = ['unboxing', 'haul', 'review', 'sponsored', 'ad', 'gifted',
                'opening', 'new toys', 'toy review', 'surprise toys',
                'mystery box', 'shopping spree']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_commercial_desc(x):
    """Commercial indicators in description"""
    desc = x.desc_lower
    if not desc:
        return ABSTAIN
    indicators = ['#ad', '#sponsored', '#gifted', 'use code', 'discount code',
                  'affiliate link', 'paid partnership', 'sponsored by',
                  'thanks to', 'provided by', 'use my link']
    if any(kw in desc for kw in indicators):
        return EXPLOIT
    return ABSTAIN

def lf_commercial_tags(x):
    """Commercial indicators in tags"""
    tags = x.tags_lower
    if not tags:
        return ABSTAIN
    indicators = ['unboxing', 'toy review', 'sponsored', 'haul', 'ad']
    if any(kw in tags for kw in indicators):
        return EXPLOIT
    return ABSTAIN

# ---- DIMENSION 6: PRIVACY VIOLATION ----
def lf_llm_privacy(x):
    """LLM text classification for privacy violation"""
    return EXPLOIT if x.privacy_violation_llm == 1 else ABSTAIN

def lf_vlm_privacy(x):
    """VLM vision classification for privacy violation"""
    if pd.isna(x.vlm_privacy):
        return ABSTAIN
    if x.vlm_privacy >= 0.4:
        return EXPLOIT
    elif x.vlm_privacy <= 0.05:
        return CLEAN
    return ABSTAIN

def lf_privacy_keywords(x):
    """Privacy-related keywords in title"""
    keywords = ['hospital', 'emergency', 'surgery', 'doctor', 'sick',
                'potty', 'bath', 'diaper', 'naked', 'bedroom',
                'private', 'secret', 'diary', 'embarrassing',
                'caught on camera', 'hidden camera']
    title = x.title_lower
    if any(kw in title for kw in keywords):
        return EXPLOIT
    return ABSTAIN

def lf_privacy_medical_desc(x):
    """Medical/private content in description"""
    desc = x.desc_lower
    if not desc:
        return ABSTAIN
    indicators = ['hospital', 'emergency room', 'surgery', 'diagnosis',
                  'medical', 'therapy', 'counseling', 'mental health']
    if any(kw in desc for kw in indicators):
        return EXPLOIT
    return ABSTAIN

# ============================================================
# STEP 3: Apply LFs and train Label Models per dimension
# ============================================================
print("\n[3/6] Applying Labeling Functions and training Label Models...")

# Define LF groups per dimension
dimensions = {
    'performative_labor': {
        'lfs': [
            LabelingFunction(name="lf_llm_performative", f=lf_llm_performative),
            LabelingFunction(name="lf_vlm_performative_high", f=lf_vlm_performative),
            LabelingFunction(name="lf_vlm_performative_med", f=lf_vlm_performative_medium),
            LabelingFunction(name="lf_scripted_keywords", f=lf_scripted_keywords),
            LabelingFunction(name="lf_organic_keywords", f=lf_organic_keywords),
            LabelingFunction(name="lf_roleplay_title", f=lf_roleplay_title),
            LabelingFunction(name="lf_desc_scripted", f=lf_desc_scripted),
            LabelingFunction(name="lf_duration_long", f=lf_duration_long_challenge),
        ]
    },
    'emotional_bait': {
        'lfs': [
            LabelingFunction(name="lf_llm_emotional", f=lf_llm_emotional),
            LabelingFunction(name="lf_vlm_emotional", f=lf_vlm_emotional),
            LabelingFunction(name="lf_emotional_caps", f=lf_emotional_title_caps),
            LabelingFunction(name="lf_emotional_punct", f=lf_emotional_punctuation),
            LabelingFunction(name="lf_emotional_keywords", f=lf_emotional_keywords),
            LabelingFunction(name="lf_emotional_emojis", f=lf_emotional_emojis),
        ]
    },
    'narrative_conflict': {
        'lfs': [
            LabelingFunction(name="lf_llm_narrative", f=lf_llm_narrative),
            LabelingFunction(name="lf_vlm_narrative", f=lf_vlm_narrative),
            LabelingFunction(name="lf_conflict_keywords", f=lf_conflict_keywords),
            LabelingFunction(name="lf_prank_gone_wrong", f=lf_prank_gone_wrong),
            LabelingFunction(name="lf_clickbait_patterns", f=lf_clickbait_patterns),
        ]
    },
    'challenge_format': {
        'lfs': [
            LabelingFunction(name="lf_llm_challenge", f=lf_llm_challenge),
            LabelingFunction(name="lf_vlm_challenge", f=lf_vlm_challenge),
            LabelingFunction(name="lf_challenge_keywords", f=lf_challenge_keywords),
            LabelingFunction(name="lf_competition_format", f=lf_competition_format),
            LabelingFunction(name="lf_duration_challenge", f=lf_duration_challenge),
        ]
    },
    'commercial_content': {
        'lfs': [
            LabelingFunction(name="lf_llm_commercial", f=lf_llm_commercial),
            LabelingFunction(name="lf_vlm_commercial", f=lf_vlm_commercial),
            LabelingFunction(name="lf_commercial_title", f=lf_commercial_keywords_title),
            LabelingFunction(name="lf_commercial_desc", f=lf_commercial_desc),
            LabelingFunction(name="lf_commercial_tags", f=lf_commercial_tags),
        ]
    },
    'privacy_violation': {
        'lfs': [
            LabelingFunction(name="lf_llm_privacy", f=lf_llm_privacy),
            LabelingFunction(name="lf_vlm_privacy", f=lf_vlm_privacy),
            LabelingFunction(name="lf_privacy_keywords", f=lf_privacy_keywords),
            LabelingFunction(name="lf_privacy_medical_desc", f=lf_privacy_medical_desc),
        ]
    }
}

# Results storage
all_results = {}
lf_analysis_all = {}

for dim_name, dim_config in dimensions.items():
    print(f"\n  --- {dim_name.upper()} ---")
    lfs = dim_config['lfs']
    
    # Apply LFs
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    
    # LF Analysis
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    lf_analysis_all[dim_name] = analysis
    print(f"  LFs: {len(lfs)}, Coverage range: {analysis['Coverage'].min():.3f} - {analysis['Coverage'].max():.3f}")
    print(f"  Conflicts: {analysis['Conflicts'].mean():.3f} mean")
    
    # Train Label Model
    label_model = LabelModel(cardinality=2, verbose=False)
    try:
        label_model.fit(L_train=L_train, n_epochs=500, lr=0.01, log_freq=100, seed=42)
        # Get probabilistic labels
        probs = label_model.predict_proba(L=L_train)
        preds = label_model.predict(L=L_train, tie_break_policy="abstain")
        
        # Store results
        df[f'{dim_name}_prob'] = probs[:, 1]  # P(exploit)
        df[f'{dim_name}_pred'] = preds
        
        # Also get majority vote for comparison
        mv = MajorityLabelVoter(cardinality=2)
        mv_preds = mv.predict(L=L_train, tie_break_policy="abstain")
        df[f'{dim_name}_mv'] = mv_preds
        
        # Get learned weights
        weights = label_model.get_weights()
        weight_dict = {lfs[i].name: float(weights[i]) for i in range(len(lfs))}
        
        all_results[dim_name] = {
            'n_lfs': len(lfs),
            'prevalence_label_model': float((preds == 1).mean()),
            'prevalence_majority_vote': float((mv_preds == 1).mean()),
            'abstain_rate': float((preds == -1).mean()),
            'learned_weights': weight_dict,
        }
        
        print(f"  Label Model prevalence: {(preds == 1).mean():.3f}")
        print(f"  Majority Vote prevalence: {(mv_preds == 1).mean():.3f}")
        print(f"  Abstain rate: {(preds == -1).mean():.3f}")
        print(f"  Top LF weights: {sorted(weight_dict.items(), key=lambda x: -x[1])[:3]}")
        
    except Exception as e:
        print(f"  ERROR training Label Model: {e}")
        # Fallback to majority vote
        mv = MajorityLabelVoter(cardinality=2)
        mv_preds = mv.predict(L=L_train, tie_break_policy="abstain")
        df[f'{dim_name}_prob'] = np.nan
        df[f'{dim_name}_pred'] = mv_preds
        df[f'{dim_name}_mv'] = mv_preds
        all_results[dim_name] = {
            'n_lfs': len(lfs),
            'prevalence_majority_vote': float((mv_preds == 1).mean()),
            'error': str(e)
        }

# ============================================================
# STEP 4: Compute overall exploitation score
# ============================================================
print("\n[4/6] Computing overall exploitation score...")

# Use the probabilistic labels from each dimension
prob_cols = [f'{dim}_prob' for dim in dimensions.keys()]
df['exploitation_score_ws'] = df[prob_cols].mean(axis=1)

# Also compute a binary prediction (any dimension flagged)
pred_cols = [f'{dim}_pred' for dim in dimensions.keys()]
df['n_dimensions_flagged'] = df[pred_cols].apply(lambda row: (row == 1).sum(), axis=1)
df['is_exploitative_ws'] = (df['exploitation_score_ws'] >= 0.5).astype(int)

# Confidence: how certain is the model?
df['confidence'] = df[prob_cols].apply(
    lambda row: row.apply(lambda p: abs(p - 0.5) * 2 if not pd.isna(p) else 0).mean(), axis=1
)

print(f"  Mean exploitation score: {df['exploitation_score_ws'].mean():.3f}")
print(f"  Exploitative (score >= 0.5): {df['is_exploitative_ws'].sum()} ({df['is_exploitative_ws'].mean()*100:.1f}%)")
print(f"  Mean confidence: {df['confidence'].mean():.3f}")

# ============================================================
# STEP 5: Correlation with engagement
# ============================================================
print("\n[5/6] Analyzing engagement correlation...")

valid = df[df['viewCount'].notna() & df['exploitation_score_ws'].notna()]
rho, pval = spearmanr(valid['exploitation_score_ws'], valid['viewCount'])
print(f"  Spearman rho (score vs views): {rho:.4f}, p = {pval:.2e}")

all_results['overall'] = {
    'n_videos': len(df),
    'n_channels': df['channel_short_name'].nunique(),
    'total_lfs': sum(r.get('n_lfs', 0) for r in all_results.values()),
    'mean_exploitation_score': float(df['exploitation_score_ws'].mean()),
    'exploitative_count': int(df['is_exploitative_ws'].sum()),
    'spearman_rho': float(rho),
    'spearman_pval': float(pval),
}

# Per-dimension engagement correlation
print("\n  Per-dimension correlations:")
for dim in dimensions.keys():
    prob_col = f'{dim}_prob'
    valid_dim = df[df[prob_col].notna() & df['viewCount'].notna()]
    if len(valid_dim) > 100:
        r, p = spearmanr(valid_dim[prob_col], valid_dim['viewCount'])
        print(f"    {dim}: rho={r:.4f}, p={p:.2e}")
        all_results[dim]['spearman_rho'] = float(r)
        all_results[dim]['spearman_pval'] = float(p)

# ============================================================
# STEP 6: Save results
# ============================================================
print("\n[6/6] Saving results...")

import os
output_dir = 'analysis_discovery/snorkel_proper'
os.makedirs(output_dir, exist_ok=True)

# Save full classified dataset
output_cols = ['id', 'title', 'channel_short_name', 'viewCount', 'publishedAt', 'channelId',
               'description', 'duration_seconds'] + \
              [f'{dim}_prob' for dim in dimensions.keys()] + \
              [f'{dim}_pred' for dim in dimensions.keys()] + \
              ['exploitation_score_ws', 'is_exploitative_ws', 'n_dimensions_flagged', 'confidence']
df[output_cols].to_csv(f'{output_dir}/classified_videos_ws.csv', index=False)

# Save summary
with open(f'{output_dir}/pipeline_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Save LF analysis per dimension
for dim_name, analysis in lf_analysis_all.items():
    analysis.to_csv(f'{output_dir}/lf_analysis_{dim_name}.csv')

# Save learned weights
weights_summary = []
for dim_name, result in all_results.items():
    if dim_name == 'overall':
        continue
    if 'learned_weights' in result:
        for lf_name, weight in result['learned_weights'].items():
            weights_summary.append({
                'dimension': dim_name,
                'lf_name': lf_name,
                'learned_weight': weight
            })
if weights_summary:
    pd.DataFrame(weights_summary).to_csv(f'{output_dir}/learned_weights.csv', index=False)

print(f"\n  Results saved to {output_dir}/")
print(f"  - classified_videos_ws.csv ({len(df)} videos)")
print(f"  - pipeline_summary.json")
print(f"  - lf_analysis_*.csv (per dimension)")
print(f"  - learned_weights.csv")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print(f"Total LFs: {all_results['overall']['total_lfs']}")
print(f"Spearman rho: {all_results['overall']['spearman_rho']:.4f}")
print("=" * 70)
