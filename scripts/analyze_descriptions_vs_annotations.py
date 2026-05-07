"""
Analyze video descriptions to test whether description-based signals
can improve performative labor and emotional bait detection.
Compare against user's human annotations.
"""
import json
import pandas as pd
import numpy as np
import re

# Load descriptions
with open('/home/ubuntu/KidInfluencer/data/descriptions/annotated_23_descriptions.json') as f:
    descriptions = json.load(f)

# Load user annotations
user = pd.read_csv('/home/ubuntu/upload/pasted_content.txt', sep='\t')
dim_cols = ['annotator_performative_labor', 'annotator_emotional_bait', 'annotator_narrative_conflict',
            'annotator_challenge_format', 'annotator_commercial_content', 'annotator_privacy_violation',
            'annotator_overall_exploitative']
for col in dim_cols:
    user[col] = pd.to_numeric(user[col], errors='coerce').fillna(0).astype(int)

# Add descriptions to user dataframe
user['description'] = user['video_id'].map(lambda x: descriptions.get(x, {}).get('description', ''))
user['tags'] = user['video_id'].map(lambda x: descriptions.get(x, {}).get('tags', []))

print(f"Videos with descriptions: {(user['description'] != '').sum()}/{len(user)}")

# ============================================================
# DESCRIPTION-BASED LABELING FUNCTIONS
# ============================================================

def lf_desc_skit_keywords(desc, tags):
    """Detect skit/roleplay/scripted content from description or tags"""
    keywords = ['skit', 'skits', 'roleplay', 'role play', 'scripted', 'acting', 
                'short film', 'movie', 'pretend', 'story time', 'storytime']
    text = (desc + ' ' + ' '.join(tags)).lower()
    return 1 if any(k in text for k in keywords) else 0

def lf_desc_challenge(desc, tags):
    """Detect challenge format from description or tags"""
    keywords = ['challenge', '24 hour', '24h', 'last to leave', 'first to', 
                'dare', 'competition', 'contest', 'game']
    text = (desc + ' ' + ' '.join(tags)).lower()
    return 1 if any(k in text for k in keywords) else 0

def lf_desc_disclaimer(desc):
    """Detect stunt/entertainment disclaimers (strong signal for performative)"""
    keywords = ['do not attempt', 'entertainment purposes', 'do not try this',
                'performed by professionals', 'trained professional', 'safety equipment',
                'at your own risk', 'properly trained']
    text = desc.lower()
    return 1 if any(k in text for k in keywords) else 0

def lf_desc_commercial_links(desc):
    """Detect commercial links (merch, affiliate, shop)"""
    patterns = ['amazon.', 'amzn.to', 'bit.ly', 'shop', 'merch', 'store', 
                'use code', 'discount', 'sponsored', 'target.com', 'walmart']
    text = desc.lower()
    return 1 if any(p in text for p in patterns) else 0

def lf_desc_manufactured_scenario(desc, tags):
    """Detect manufactured/scripted scenarios from description"""
    # Look for phrases that indicate the content was created for the camera
    keywords = ['we built', 'we made', 'we created', 'surprise', 'prank',
                'we set up', 'we designed', 'we turned', 'transformed',
                'mystery', 'escape room', 'obstacle course', 'treasure hunt']
    text = (desc + ' ' + ' '.join(tags)).lower()
    return 1 if any(k in text for k in keywords) else 0

def lf_desc_organic_activity(desc, tags):
    """Detect organic/natural activities (AGAINST performative)"""
    keywords = ['vlog', 'day in the life', 'routine', 'travel', 'vacation',
                'birthday party', 'family trip', 'visiting', 'moving',
                'doctor', 'hospital', 'school', 'grocery', 'shopping trip']
    text = (desc + ' ' + ' '.join(tags)).lower()
    # Only count as organic if NO skit/challenge signals
    if any(k in text for k in keywords):
        # Check it's not also a manufactured scenario
        manuf_kw = ['prank', 'challenge', 'skit', 'mystery', 'escape']
        if not any(k in text for k in manuf_kw):
            return 1
    return 0

def lf_desc_emotional_bait_signals(desc, title, tags):
    """Detect emotional bait from title + description combined"""
    text = (title + ' ' + desc + ' ' + ' '.join(tags)).lower()
    keywords = ['you won\'t believe', 'shocking', 'emotional', 'crying', 'cried',
                'scared', 'scary', 'terrifying', 'heartbreaking', 'devastating',
                'regret', 'instantly regret', 'gone wrong', 'exposed', 'caught',
                'not clickbait', 'this is real', 'i can\'t believe']
    return 1 if any(k in text for k in keywords) else 0

def lf_tags_performative(tags):
    """Check tags for performative signals"""
    perf_tags = ['skit', 'skits', 'roleplay', 'challenge', 'prank', 'pranks',
                 'acting', 'short film', 'comedy skit', 'funny skit']
    tags_lower = [t.lower() for t in tags]
    return 1 if any(t in tags_lower for t in perf_tags) else 0

# ============================================================
# APPLY ALL DESCRIPTION-BASED LFs
# ============================================================

results = []
for _, row in user.iterrows():
    desc = row['description']
    tags = row['tags'] if isinstance(row['tags'], list) else []
    title = row['title']
    
    r = {
        'video_id': row['video_id'],
        'title': title,
        'h_performative': row['annotator_performative_labor'],
        'h_emotional_bait': row['annotator_emotional_bait'],
        'h_overall': row['annotator_overall_exploitative'],
        'lf_skit': lf_desc_skit_keywords(desc, tags),
        'lf_challenge': lf_desc_challenge(desc, tags),
        'lf_disclaimer': lf_desc_disclaimer(desc),
        'lf_commercial': lf_desc_commercial_links(desc),
        'lf_manufactured': lf_desc_manufactured_scenario(desc, tags),
        'lf_organic': lf_desc_organic_activity(desc, tags),
        'lf_emotional': lf_desc_emotional_bait_signals(desc, title, tags),
        'lf_tags_perf': lf_tags_performative(tags),
    }
    results.append(r)

rdf = pd.DataFrame(results)

# Aggregate: performative = skit OR disclaimer OR manufactured OR tags_perf OR challenge (and NOT organic)
rdf['pred_performative'] = ((rdf['lf_skit'] | rdf['lf_disclaimer'] | rdf['lf_manufactured'] | 
                             rdf['lf_tags_perf'] | rdf['lf_challenge']) & ~rdf['lf_organic'].astype(bool)).astype(int)

# Emotional bait prediction
rdf['pred_emotional'] = rdf['lf_emotional']

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DESCRIPTION-BASED LF RESULTS: PERFORMATIVE LABOR")
print("=" * 90)

print(f"\n{'#':<3} {'Title':<42} {'H_perf':<7} {'Pred':<5} {'Signals':<40} {'Match'}")
print("─" * 100)

perf_correct = 0
for _, row in rdf.iterrows():
    title_short = row['title'][:39] + "..." if len(row['title']) > 39 else row['title']
    h = row['h_performative']
    p = row['pred_performative']
    match = "✅" if h == p else "❌"
    if h == p:
        perf_correct += 1
    
    # Show which signals fired
    signals = []
    if row['lf_skit']: signals.append('skit')
    if row['lf_disclaimer']: signals.append('disclaimer')
    if row['lf_manufactured']: signals.append('manufactured')
    if row['lf_tags_perf']: signals.append('tags')
    if row['lf_challenge']: signals.append('challenge')
    if row['lf_organic']: signals.append('ORGANIC(-)')
    
    sig_str = ', '.join(signals) if signals else '—'
    print(f"   {title_short:<42} {h:<7} {p:<5} {sig_str:<40} {match}")

print(f"\nPerformative Accuracy: {perf_correct}/{len(rdf)} = {perf_correct/len(rdf):.1%}")

# Compare with title-only approach
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score

y_true_perf = rdf['h_performative'].values
y_pred_perf = rdf['pred_performative'].values

print(f"\n  F1:        {f1_score(y_true_perf, y_pred_perf, zero_division=0):.3f}")
print(f"  Precision: {precision_score(y_true_perf, y_pred_perf, zero_division=0):.3f}")
print(f"  Recall:    {recall_score(y_true_perf, y_pred_perf, zero_division=0):.3f}")
print(f"  Cohen's κ: {cohen_kappa_score(y_true_perf, y_pred_perf):.3f}")

print("\n" + "=" * 90)
print("DESCRIPTION-BASED LF RESULTS: EMOTIONAL BAIT")
print("=" * 90)

print(f"\n{'#':<3} {'Title':<42} {'H_emo':<7} {'Pred':<5} {'Match'}")
print("─" * 65)

emo_correct = 0
for _, row in rdf.iterrows():
    title_short = row['title'][:39] + "..." if len(row['title']) > 39 else row['title']
    h = row['h_emotional_bait']
    p = row['pred_emotional']
    match = "✅" if h == p else "❌"
    if h == p:
        emo_correct += 1
    print(f"   {title_short:<42} {h:<7} {p:<5} {match}")

y_true_emo = rdf['h_emotional_bait'].values
y_pred_emo = rdf['pred_emotional'].values

print(f"\nEmotional Bait Accuracy: {emo_correct}/{len(rdf)} = {emo_correct/len(rdf):.1%}")
print(f"  F1:        {f1_score(y_true_emo, y_pred_emo, zero_division=0):.3f}")
print(f"  Precision: {precision_score(y_true_emo, y_pred_emo, zero_division=0):.3f}")
print(f"  Recall:    {recall_score(y_true_emo, y_pred_emo, zero_division=0):.3f}")
print(f"  Cohen's κ: {cohen_kappa_score(y_true_emo, y_pred_emo):.3f}")

# ============================================================
# KEY INSIGHTS FROM DESCRIPTIONS
# ============================================================
print("\n" + "=" * 90)
print("KEY INSIGHTS: WHAT DESCRIPTIONS REVEAL")
print("=" * 90)

# Show descriptions for videos where human said performative but model missed
fn_perf = rdf[(rdf['h_performative'] == 1) & (rdf['pred_performative'] == 0)]
print(f"\n🔍 FALSE NEGATIVES for Performative (Human=1, Desc-LF=0): {len(fn_perf)}")
for _, row in fn_perf.iterrows():
    vid = row['video_id']
    desc_text = descriptions.get(vid, {}).get('description', '')[:200]
    tags = descriptions.get(vid, {}).get('tags', [])[:5]
    print(f"\n  📹 {row['title'][:50]}")
    print(f"     Tags: {tags}")
    print(f"     Desc: {desc_text[:150]}...")

# Show descriptions for videos where desc-LF correctly identified performative
tp_perf = rdf[(rdf['h_performative'] == 1) & (rdf['pred_performative'] == 1)]
print(f"\n✅ TRUE POSITIVES for Performative (Human=1, Desc-LF=1): {len(tp_perf)}")
for _, row in tp_perf.iterrows():
    vid = row['video_id']
    tags = descriptions.get(vid, {}).get('tags', [])[:5]
    signals = []
    if row['lf_skit']: signals.append('skit')
    if row['lf_disclaimer']: signals.append('disclaimer')
    if row['lf_manufactured']: signals.append('manufactured')
    if row['lf_tags_perf']: signals.append('tags')
    if row['lf_challenge']: signals.append('challenge')
    print(f"  📹 {row['title'][:50]} → signals: {signals}")
    print(f"     Tags: {tags}")
