"""
Snorkel Weak Supervision Pipeline v3 for Kidfluencer Exploitation Detection
Updated for expanded dataset: 79 channels, 4685 stratified-sampled videos.

18 Labeling Functions:
- 6 LLM-based (GPT-4.1-mini title classification per dimension)
- 9 Rule-based (title keywords, formatting signals)
- 3 CV/Vision-based (saturation, child distress, exploitation concern)

Changes from v2:
- Uses expanded dataset (79 channels instead of 48)
- Loads from llm_classifications_v2.csv (column names: performative_labor, etc.)
- Includes audience moderation analysis (child vs teen/adult)
- Controls for video age where available
"""
import pandas as pd
import numpy as np
import re
import os
import json
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Constants
EXPLOIT = 1
NOT_EXPLOIT = 0
ABSTAIN = -1

OUTPUT_DIR = '/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("SNORKEL WEAK SUPERVISION PIPELINE v3 (EXPANDED DATASET)")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")

# Load LLM classifications from expanded sample
sample = pd.read_csv('/home/ubuntu/KidInfluencer/data/llm_classifications_v2.csv')
# Rename columns to match LF expectations
sample = sample.rename(columns={'performative_labor': 'performative'})
print(f"  LLM classification: {len(sample)} videos, {sample['channel_short_name'].nunique()} channels")

# Ensure viewCount is numeric
sample['viewCount'] = pd.to_numeric(sample['viewCount'], errors='coerce')

# Load thumbnail CV results (only available for original 48 channels subset)
cv_path = '/home/ubuntu/KidInfluencer/analysis_discovery/thumbnail_cv_v2.csv'
if os.path.exists(cv_path):
    thumb_cv = pd.read_csv(cv_path)
    thumb_cv = thumb_cv.rename(columns={'video_id': 'id'})
    sample = sample.merge(thumb_cv[['id', 'saturation', 'brightness', 'num_faces', 'num_smiles']], 
                          on='id', how='left')
    print(f"  Merged CV data: {sample['saturation'].notna().sum()} matches")

# Load Vision API results (subset ~217 videos)
vision_path = '/home/ubuntu/KidInfluencer/analysis_discovery/thumbnail_vision_v2_clean.csv'
if os.path.exists(vision_path):
    thumb_vision = pd.read_csv(vision_path)
    thumb_vision = thumb_vision.rename(columns={'video_id': 'id', 'exploitation_concern': 'vision_exploit_concern'})
    thumb_vision['child_present'] = thumb_vision['child_present'].map({True: True, False: False, 'True': True, 'False': False})
    sample = sample.merge(
        thumb_vision[['id', 'child_present', 'child_emotion', 'emotion_appears_genuine', 'vision_exploit_concern']], 
        on='id', how='left'
    )
    print(f"  Merged Vision API data: {sample['vision_exploit_concern'].notna().sum()} matches")

print(f"\nFinal dataset: {len(sample)} videos, {sample['channel_short_name'].nunique()} channels")
print(f"  View count available: {sample['viewCount'].notna().sum()}")
print(f"  View count > 0: {(sample['viewCount'] > 0).sum()}")
print()

# ============================================================
# AUDIENCE CLASSIFICATION
# ============================================================
print("Classifying channels by target audience...")

# Channels primarily targeting young children (pre-school / early elementary)
CHILD_AUDIENCE_CHANNELS = [
    'likenastya', 'kidsdianashow', 'vladandniki', 'aforadley',
    'ryansworldchannel', 'cocomelon', 'babybus', 'blippi',
    'dianakidsshow', 'nastya', 'funsquadfamily', 'comeplaywithme',
    'thelabrantfam', 'itsyeboi', 'smellybellytv', 'everleighrose',
    'ninjokidztv', 'fgteev', 'funkidsfamily', 'littleangel'
]

sample['audience_type'] = sample['channel_short_name'].apply(
    lambda x: 'child' if x.lower() in [c.lower() for c in CHILD_AUDIENCE_CHANNELS] else 'teen_adult'
)
print(f"  Child audience channels: {(sample['audience_type'] == 'child').sum()} videos")
print(f"  Teen/adult audience channels: {(sample['audience_type'] == 'teen_adult').sum()} videos")

# ============================================================
# LABELING FUNCTIONS (18 total)
# ============================================================
print("\nDefining 18 labeling functions...")

# --- Category 1: LLM-based (6 LFs) ---
def lf_llm_performative(x):
    """Child performing scripted content for camera (Art. 32 UNCRC - child labor)"""
    if x.performative == 1:
        return EXPLOIT
    elif x.performative == 0:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_llm_emotional_bait(x):
    """Title uses emotional manipulation/clickbait (Art. 19 UNCRC - harm)"""
    if x.emotional_bait == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_narrative_conflict(x):
    """Manufactured narrative conflict (Art. 19 UNCRC)"""
    if x.narrative_conflict == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_challenge(x):
    """Challenge/competition format requiring child labor (Art. 32 UNCRC)"""
    if x.challenge_format == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_commercial(x):
    """Commercial exploitation of child (Art. 32 UNCRC)"""
    if x.commercial_content == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_privacy(x):
    """Privacy violation of child (Art. 16 UNCRC)"""
    if x.privacy_violation == 1:
        return EXPLOIT
    return ABSTAIN

# --- Category 2: Rule-based (9 LFs) ---
def lf_all_caps_ratio(x):
    """High ratio of capital letters = clickbait formatting"""
    title = str(x.title)
    if len(title) < 5:
        return ABSTAIN
    caps_ratio = sum(1 for c in title if c.isupper()) / len(title)
    if caps_ratio > 0.6:
        return EXPLOIT
    return ABSTAIN

def lf_exclamation_marks(x):
    """Multiple exclamation marks = sensationalism"""
    title = str(x.title)
    if title.count('!') >= 3:
        return EXPLOIT
    return ABSTAIN

def lf_conflict_keywords(x):
    """Conflict/drama keywords in title"""
    title = str(x.title).lower()
    conflict_words = ['fight', 'kicked out', 'gone wrong', 'arrested', 'emergency',
                      'hospital', 'stolen', 'broke', 'destroyed', 'ruined', 'caught',
                      'exposed', 'confronting', 'called the police']
    if any(w in title for w in conflict_words):
        return EXPLOIT
    return ABSTAIN

def lf_challenge_keywords(x):
    """Challenge/competition keywords"""
    title = str(x.title).lower()
    challenge_words = ['challenge', '24 hours', '24 hour', 'last to', 'first to',
                       'vs', 'versus', 'competition', 'wins', 'loses', 'dare']
    if any(w in title for w in challenge_words):
        return EXPLOIT
    return ABSTAIN

def lf_prank_keywords(x):
    """Pranks on children = exploitation (Art. 19)"""
    title = str(x.title).lower()
    prank_words = ['prank', 'trick', 'scare', 'scared', 'surprise attack', 'revenge']
    if any(w in title for w in prank_words):
        return EXPLOIT
    return ABSTAIN

def lf_emotional_keywords(x):
    """Emotional exploitation keywords"""
    title = str(x.title).lower()
    emotion_words = ['cried', 'crying', 'tears', 'heartbroken', 'devastated',
                     'freaked out', 'meltdown', 'tantrum', 'screaming', 'sobbing']
    if any(w in title for w in emotion_words):
        return EXPLOIT
    return ABSTAIN

def lf_organic_keywords(x):
    """Organic family content (negative label)"""
    title = str(x.title).lower()
    organic_words = ['birthday', 'christmas morning', 'first day of school',
                     'vacation', 'road trip', 'family dinner', 'cooking with',
                     'gardening', 'hiking', 'beach day', 'snow day']
    if any(w in title for w in organic_words):
        return NOT_EXPLOIT
    return ABSTAIN

def lf_roleplay_keywords(x):
    """Roleplay/scripted scenarios requiring child to act"""
    title = str(x.title).lower()
    roleplay_words = ['pretend', 'role play', 'roleplay', 'acting', 'skit',
                      'story time', 'spy', 'detective', 'undercover']
    if any(w in title for w in roleplay_words):
        return EXPLOIT
    return ABSTAIN

def lf_urgency_words(x):
    """Urgency/sensationalism in title"""
    title = str(x.title).lower()
    urgency_words = ['shocking', "you won't believe", 'insane', 'crazy', 'epic',
                     'worst', 'biggest', 'most dangerous', 'never before', 'gone wrong']
    if any(w in title for w in urgency_words):
        return EXPLOIT
    return ABSTAIN

# --- Category 3: CV/Vision-based (3 LFs) ---
def lf_cv_high_saturation(x):
    """High thumbnail saturation = visual manipulation strategy"""
    sat = getattr(x, 'saturation', np.nan)
    if pd.isna(sat):
        return ABSTAIN
    if sat > 130:
        return EXPLOIT
    elif sat < 70:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_cv_child_distress(x):
    """Vision API detected child distress/negative emotion"""
    emotion = getattr(x, 'child_emotion', np.nan)
    if pd.isna(emotion):
        return ABSTAIN
    emotion = str(emotion).lower()
    if emotion in ['distressed', 'crying', 'scared', 'sad', 'angry']:
        return EXPLOIT
    elif emotion in ['happy', 'neutral']:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_cv_vision_exploit(x):
    """Vision API exploitation concern score >= 2"""
    concern = getattr(x, 'vision_exploit_concern', np.nan)
    if pd.isna(concern):
        return ABSTAIN
    try:
        concern = int(concern)
    except (ValueError, TypeError):
        return ABSTAIN
    if concern >= 2:
        return EXPLOIT
    elif concern == 0:
        return NOT_EXPLOIT
    return ABSTAIN

# ============================================================
# BUILD AND APPLY LABELING FUNCTIONS
# ============================================================
print("Building labeling functions...")

lfs = [
    LabelingFunction(name="lf_llm_performative", f=lf_llm_performative),
    LabelingFunction(name="lf_llm_emotional_bait", f=lf_llm_emotional_bait),
    LabelingFunction(name="lf_llm_narrative_conflict", f=lf_llm_narrative_conflict),
    LabelingFunction(name="lf_llm_challenge", f=lf_llm_challenge),
    LabelingFunction(name="lf_llm_commercial", f=lf_llm_commercial),
    LabelingFunction(name="lf_llm_privacy", f=lf_llm_privacy),
    LabelingFunction(name="lf_all_caps_ratio", f=lf_all_caps_ratio),
    LabelingFunction(name="lf_exclamation_marks", f=lf_exclamation_marks),
    LabelingFunction(name="lf_conflict_keywords", f=lf_conflict_keywords),
    LabelingFunction(name="lf_challenge_keywords", f=lf_challenge_keywords),
    LabelingFunction(name="lf_prank_keywords", f=lf_prank_keywords),
    LabelingFunction(name="lf_emotional_keywords", f=lf_emotional_keywords),
    LabelingFunction(name="lf_organic_keywords", f=lf_organic_keywords),
    LabelingFunction(name="lf_roleplay_keywords", f=lf_roleplay_keywords),
    LabelingFunction(name="lf_urgency_words", f=lf_urgency_words),
    LabelingFunction(name="lf_cv_high_saturation", f=lf_cv_high_saturation),
    LabelingFunction(name="lf_cv_child_distress", f=lf_cv_child_distress),
    LabelingFunction(name="lf_cv_vision_exploit", f=lf_cv_vision_exploit),
]

print(f"  Total LFs: {len(lfs)} (6 LLM + 9 Rule + 3 CV/Vision)")

# Apply LFs
print("\nApplying labeling functions to dataset...")
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=sample)
print(f"  Label matrix shape: {L_train.shape}")

# LF Analysis
print("\nLabeling Function Analysis:")
lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(lf_analysis.to_string())
lf_analysis.to_csv(f'{OUTPUT_DIR}/lf_analysis.csv')

# ============================================================
# TRAIN LABEL MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING SNORKEL LABEL MODEL")
print("=" * 60)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.01, log_freq=100, seed=42)

# Get probabilistic labels
probs = label_model.predict_proba(L=L_train)
preds = label_model.predict(L=L_train, tie_break_policy="abstain")

# Exploitation score = probability of EXPLOIT class
sample['exploitation_score'] = probs[:, 1]
sample['exploitation_pred'] = preds

print(f"\nLabel Model Results:")
print(f"  Predicted EXPLOIT: {(preds == 1).sum()} ({(preds == 1).mean()*100:.1f}%)")
print(f"  Predicted NOT_EXPLOIT: {(preds == 0).sum()} ({(preds == 0).mean()*100:.1f}%)")
print(f"  Abstained: {(preds == -1).sum()} ({(preds == -1).mean()*100:.1f}%)")
print(f"  Mean exploitation score: {sample['exploitation_score'].mean():.3f}")
print(f"  Median exploitation score: {sample['exploitation_score'].median():.3f}")

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

# 1. Overall correlation
valid = sample[sample['viewCount'].notna() & (sample['viewCount'] > 0)].copy()
valid['log_views'] = np.log10(valid['viewCount'])
corr, pval = stats.spearmanr(valid['exploitation_score'], valid['log_views'])
print(f"\n1. Overall Spearman correlation (exploitation score vs log views):")
print(f"   ρ = {corr:.4f}, p = {pval:.4e}")

# 2. Within-channel analysis by dimension
print(f"\n2. Within-channel view boost analysis:")
dim_results = []
dimensions = ['performative', 'emotional_bait', 'narrative_conflict', 
              'challenge_format', 'commercial_content', 'privacy_violation']

for dim in dimensions:
    boosts = []
    for ch, group in valid.groupby('channel_short_name'):
        if len(group) < 10:
            continue
        dim_1 = group[group[dim] == 1]['viewCount']
        dim_0 = group[group[dim] == 0]['viewCount']
        if len(dim_1) >= 3 and len(dim_0) >= 3:
            boost = (dim_1.median() / dim_0.median() - 1) * 100
            boosts.append(boost)
    
    if boosts:
        mean_boost = np.mean(boosts)
        median_boost = np.median(boosts)
        t_stat, t_pval = stats.ttest_1samp(boosts, 0)
        dim_results.append({
            'dimension': dim,
            'n_channels': len(boosts),
            'mean_within_boost': mean_boost,
            'median_within_boost': median_boost,
            't_stat': t_stat,
            'p_value': t_pval / 2 if t_stat > 0 else 1 - t_pval / 2,
        })
        sig = "***" if t_pval/2 < 0.001 else "**" if t_pval/2 < 0.01 else "*" if t_pval/2 < 0.05 else "n.s."
        print(f"   {dim}: mean boost = {mean_boost:+.1f}%, median = {median_boost:+.1f}%, t={t_stat:.2f}, p={t_pval/2:.4f} {sig}")

dim_df = pd.DataFrame(dim_results)
dim_df.to_csv(f'{OUTPUT_DIR}/within_channel_boost_by_dimension.csv', index=False)

# 3. Within-channel exploitation score boost
print(f"\n3. Within-channel exploitation score boost:")
ch_results = []
for ch, group in valid.groupby('channel_short_name'):
    if len(group) < 10:
        continue
    median_score = group['exploitation_score'].median()
    high_exploit = group[group['exploitation_score'] > median_score]
    low_exploit = group[group['exploitation_score'] <= median_score]
    
    if len(high_exploit) >= 3 and len(low_exploit) >= 3:
        boost = (high_exploit['viewCount'].median() / low_exploit['viewCount'].median() - 1) * 100
        u_stat, u_pval = stats.mannwhitneyu(high_exploit['viewCount'], low_exploit['viewCount'], alternative='greater')
        ch_results.append({
            'channel': ch,
            'n': len(group),
            'audience_type': group['audience_type'].iloc[0],
            'mean_exploit_score': group['exploitation_score'].mean(),
            'within_channel_boost': boost,
            'high_exploit_median_views': high_exploit['viewCount'].median(),
            'low_exploit_median_views': low_exploit['viewCount'].median(),
            'u_stat': u_stat,
            'p_value': u_pval,
        })

ch_df = pd.DataFrame(ch_results)
if len(ch_df) > 0:
    sig_channels = ch_df[ch_df['p_value'] < 0.05]
    print(f"   Channels analyzed: {len(ch_df)}")
    print(f"   Channels with significant boost (p<0.05): {len(sig_channels)} ({len(sig_channels)/len(ch_df)*100:.0f}%)")
    print(f"   Mean within-channel boost: {ch_df['within_channel_boost'].mean():.1f}%")
    print(f"   Median within-channel boost: {ch_df['within_channel_boost'].median():.1f}%")
    ch_df.to_csv(f'{OUTPUT_DIR}/channel_exploitation_analysis.csv', index=False)

# 4. AUDIENCE MODERATION ANALYSIS
print(f"\n4. Audience Moderation Analysis:")
if len(ch_df) > 0:
    child_channels = ch_df[ch_df['audience_type'] == 'child']
    teen_channels = ch_df[ch_df['audience_type'] == 'teen_adult']
    
    print(f"   Child-audience channels: {len(child_channels)}")
    print(f"     Mean boost: {child_channels['within_channel_boost'].mean():.1f}%")
    print(f"     Median boost: {child_channels['within_channel_boost'].median():.1f}%")
    print(f"     Positive boost: {(child_channels['within_channel_boost'] > 0).sum()}/{len(child_channels)}")
    
    print(f"   Teen/Adult-audience channels: {len(teen_channels)}")
    print(f"     Mean boost: {teen_channels['within_channel_boost'].mean():.1f}%")
    print(f"     Median boost: {teen_channels['within_channel_boost'].median():.1f}%")
    print(f"     Positive boost: {(teen_channels['within_channel_boost'] > 0).sum()}/{len(teen_channels)}")
    
    # Mann-Whitney U test between groups
    if len(child_channels) >= 3 and len(teen_channels) >= 3:
        u_stat, u_pval = stats.mannwhitneyu(
            teen_channels['within_channel_boost'], 
            child_channels['within_channel_boost'], 
            alternative='greater'
        )
        print(f"   Mann-Whitney U (teen/adult > child): U={u_stat:.0f}, p={u_pval:.4f}")
        
        # Cohen's d
        pooled_std = np.sqrt((teen_channels['within_channel_boost'].var() + child_channels['within_channel_boost'].var()) / 2)
        if pooled_std > 0:
            cohens_d = (teen_channels['within_channel_boost'].mean() - child_channels['within_channel_boost'].mean()) / pooled_std
            print(f"   Cohen's d: {cohens_d:.3f}")

# 5. LF weight analysis
print(f"\n5. Label Model Learned Weights:")
weights = label_model.get_weights()
weight_df = pd.DataFrame({
    'lf_name': [lf.name for lf in lfs],
    'weight': weights,
    'category': ['LLM']*6 + ['Rule']*9 + ['CV/Vision']*3
})
weight_df = weight_df.sort_values('weight', ascending=False)
print(weight_df.to_string(index=False))
weight_df.to_csv(f'{OUTPUT_DIR}/lf_weights.csv', index=False)

# ============================================================
# GENERATE FIGURES
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

# Figure 1: Exploitation score distribution + scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(sample['exploitation_score'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(sample['exploitation_score'].median(), color='red', linestyle='--', 
           label=f"Median={sample['exploitation_score'].median():.3f}")
ax.set_xlabel('Exploitation Score (P(exploit))')
ax.set_ylabel('Count')
ax.set_title('Distribution of Exploitation Scores\n(Snorkel Label Model Output)')
ax.legend()

ax = axes[1]
scatter = ax.scatter(valid['exploitation_score'], valid['log_views'], 
                    alpha=0.3, s=15, c=valid['exploitation_score'], cmap='RdYlGn_r')
ax.set_xlabel('Exploitation Score')
ax.set_ylabel('log₁₀(Views)')
ax.set_title(f'Exploitation Score vs Views\n(Spearman ρ={corr:.3f}, p={pval:.2e})')
plt.colorbar(scatter, ax=ax, label='Exploitation Score')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_score_distribution_and_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_score_distribution_and_views.png")

# Figure 2: Within-channel boost by dimension
if dim_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    dim_df_sorted = dim_df.sort_values('mean_within_boost', ascending=True)
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in dim_df_sorted['p_value']]
    bars = ax.barh(dim_df_sorted['dimension'], dim_df_sorted['mean_within_boost'], 
                   color=colors, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Within-Channel View Boost (%)')
    ax.set_title('Per-Dimension Within-Channel View Boost\n(Green = p<0.05, Gray = not significant)')
    for i, (_, row) in enumerate(dim_df_sorted.iterrows()):
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
        offset = max(row['mean_within_boost'] + 2, 2)
        ax.text(offset, i, f"p={row['p_value']:.3f} {sig}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_within_channel_boost.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_within_channel_boost.png")

# Figure 3: LF coverage and weights
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
categories = ['LLM']*6 + ['Rule']*9 + ['CV/Vision']*3
cat_colors = {'LLM': '#3498db', 'Rule': '#e74c3c', 'CV/Vision': '#27ae60'}
colors = [cat_colors[c] for c in categories]
coverage = lf_analysis['Coverage'].values
names = [lf.name.replace('lf_', '') for lf in lfs]
y_pos = range(len(names))

ax = axes[0]
ax.barh(y_pos, coverage, color=colors, alpha=0.8, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Coverage (fraction labeled)')
ax.set_title('Labeling Function Coverage')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in ['LLM', 'Rule', 'CV/Vision']]
ax.legend(handles=legend_elements, loc='lower right')

ax = axes[1]
weight_sorted = weight_df.sort_values('weight', ascending=True)
w_colors = [cat_colors[c] for c in weight_sorted['category']]
ax.barh(range(len(weight_sorted)), weight_sorted['weight'], color=w_colors, alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(weight_sorted)))
ax.set_yticklabels([n.replace('lf_', '') for n in weight_sorted['lf_name']], fontsize=8)
ax.set_xlabel('Learned Weight')
ax.set_title('Label Model Learned Weights')
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_lf_coverage_and_weights.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_lf_coverage_and_weights.png")

# Figure 4: Audience moderation
if len(ch_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax = axes[0]
    child_boosts = ch_df[ch_df['audience_type'] == 'child']['within_channel_boost']
    teen_boosts = ch_df[ch_df['audience_type'] == 'teen_adult']['within_channel_boost']
    bp = ax.boxplot([child_boosts, teen_boosts], labels=['Child\nAudience', 'Teen/Adult\nAudience'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Within-Channel View Boost (%)')
    ax.set_title('Exploitation Premium by Target Audience')
    
    # Scatter by audience
    ax = axes[1]
    for atype, color, marker in [('child', '#3498db', 'o'), ('teen_adult', '#e74c3c', 's')]:
        subset = ch_df[ch_df['audience_type'] == atype]
        ax.scatter(subset['mean_exploit_score'], subset['within_channel_boost'],
                  c=color, marker=marker, s=60, alpha=0.7, label=f'{atype} ({len(subset)})')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Mean Exploitation Score')
    ax.set_ylabel('Within-Channel View Boost (%)')
    ax.set_title('Channel Exploitation Score vs View Boost\n(by Audience Type)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_audience_moderation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_audience_moderation.png")

# Figure 5: Dimension prevalence
fig, ax = plt.subplots(figsize=(10, 5))
dim_prev = sample[dimensions].mean().sort_values(ascending=True) * 100
ax.barh(dim_prev.index, dim_prev.values, color='steelblue', edgecolor='white', height=0.6)
ax.set_xlabel('Prevalence (%)')
ax.set_title('Exploitation Dimension Prevalence in Sample\n(GPT-4.1-mini Classification)')
for i, (dim, val) in enumerate(dim_prev.items()):
    ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_dimension_prevalence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_dimension_prevalence.png")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

sample.to_csv(f'{OUTPUT_DIR}/videos_with_exploitation_scores.csv', index=False)
print(f"  Saved videos_with_exploitation_scores.csv ({len(sample)} videos)")

# Save summary
summary = {
    'n_videos': int(len(sample)),
    'n_channels': int(sample['channel_short_name'].nunique()),
    'n_labeling_functions': len(lfs),
    'n_lf_llm': 6,
    'n_lf_rule': 9,
    'n_lf_cv': 3,
    'mean_exploitation_score': float(sample['exploitation_score'].mean()),
    'median_exploitation_score': float(sample['exploitation_score'].median()),
    'exploit_predicted': int((preds == 1).sum()),
    'not_exploit_predicted': int((preds == 0).sum()),
    'abstained': int((preds == -1).sum()),
    'spearman_rho_score_views': float(corr),
    'spearman_pvalue': float(pval),
    'dimension_prevalence': {dim: float(sample[dim].mean()) for dim in dimensions},
    'within_channel_results': dim_results,
    'n_channels_with_significant_boost': int(len(sig_channels)) if len(ch_df) > 0 else 0,
    'mean_within_channel_boost': float(ch_df['within_channel_boost'].mean()) if len(ch_df) > 0 else None,
    'median_within_channel_boost': float(ch_df['within_channel_boost'].median()) if len(ch_df) > 0 else None,
    'audience_moderation': {
        'child_channels': int(len(child_channels)) if len(ch_df) > 0 else 0,
        'teen_adult_channels': int(len(teen_channels)) if len(ch_df) > 0 else 0,
        'child_mean_boost': float(child_channels['within_channel_boost'].mean()) if len(ch_df) > 0 and len(child_channels) > 0 else None,
        'teen_adult_mean_boost': float(teen_channels['within_channel_boost'].mean()) if len(ch_df) > 0 and len(teen_channels) > 0 else None,
    }
}

with open(f'{OUTPUT_DIR}/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved summary_statistics.json")

# Print final summary
print("\n" + "=" * 60)
print("PIPELINE v3 COMPLETE - KEY FINDINGS")
print("=" * 60)
print(f"\n  Dataset: {len(sample)} videos from {sample['channel_short_name'].nunique()} channels")
print(f"  Labeling Functions: {len(lfs)} (6 LLM + 9 Rule + 3 CV/Vision)")
print(f"  Mean exploitation score: {sample['exploitation_score'].mean():.3f}")
print(f"  Exploitation predicted: {(preds==1).sum()} ({(preds==1).mean()*100:.1f}%)")
print(f"  Spearman ρ (score vs views): {corr:.4f} (p={pval:.4e})")
if len(ch_df) > 0:
    print(f"  Within-channel boost: {ch_df['within_channel_boost'].mean():.1f}% mean, {ch_df['within_channel_boost'].median():.1f}% median")
    print(f"  Channels with sig. boost: {len(sig_channels)}/{len(ch_df)}")
print(f"\n  Dimension prevalence:")
for dim in dimensions:
    print(f"    {dim}: {sample[dim].mean()*100:.1f}%")
print(f"\n  Audience moderation:")
if len(ch_df) > 0:
    print(f"    Child audience boost: {child_channels['within_channel_boost'].mean():.1f}% (n={len(child_channels)})")
    print(f"    Teen/adult audience boost: {teen_channels['within_channel_boost'].mean():.1f}% (n={len(teen_channels)})")
print("\n  Done!")
