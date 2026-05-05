"""
Snorkel Weak Supervision Pipeline for Kidfluencer Exploitation Detection

This script implements a data programming approach to detect child exploitation
in kidfluencer content using multiple noisy labeling functions from:
1. LLM-based content analysis (GPT-4.1-mini title classification)
2. Rule-based metadata signals (title formatting, keywords)
3. CV-based thumbnail analysis (facial emotions, visual clickbait)

The label model aggregates these signals into a unified exploitation score
without requiring any manually labeled ground truth.

Literature basis:
- Clark & Jno-Charles (2025) "Five Fundamental Threats" framework (UNCRC)
- Freitas (2024) "playbour" concept
- Ratner et al. (2020) Snorkel data programming methodology
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

OUTPUT_DIR = '/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
# Load the gpt-4.1-mini classified sample (2306 videos, 48 channels)
sample = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/classification_5dim_sample.csv')

# Load thumbnail CV results if available
thumb_cv_path = '/home/ubuntu/KidInfluencer/analysis_discovery/thumbnail_cv_results.csv'
thumb_vision_path = '/home/ubuntu/KidInfluencer/analysis_discovery/thumbnail_vision_results.csv'

if os.path.exists(thumb_cv_path):
    thumb_cv = pd.read_csv(thumb_cv_path)
    # Columns: id, n_faces, has_face, has_large_face, total_face_area_ratio, n_smiles, has_smile, smile_ratio, mean_saturation, mean_brightness, edge_density, color_std
    sample = sample.merge(thumb_cv[['id', 'n_faces', 'has_smile', 'mean_saturation']], 
                         on='id', how='left')
    sample.rename(columns={'n_faces': 'num_faces', 'mean_saturation': 'saturation_mean'}, inplace=True)
    print(f"  Merged thumbnail CV data: {sample['num_faces'].notna().sum()} matches")

if os.path.exists(thumb_vision_path):
    thumb_vision = pd.read_csv(thumb_vision_path)
    # Columns: child_visible, child_count, child_emotion, adult_visible, adult_emotion, scene_type, is_animated, has_text_overlay, emotional_tone, exploitation_concern, brief_description, id, title, channel
    sample = sample.merge(thumb_vision[['id', 'child_visible', 'child_emotion', 'exploitation_concern']], 
                         on='id', how='left')
    sample.rename(columns={'child_visible': 'child_present'}, inplace=True)
    print(f"  Merged thumbnail vision data: {sample['child_present'].notna().sum()} matches")

print(f"Final dataset: {len(sample)} videos, {sample['channel_short_name'].nunique()} channels")
print()

# ============================================================
# LABELING FUNCTIONS
# ============================================================

# --- Category 1: LLM-based Labeling Functions ---
# These use GPT-4.1-mini's classification as noisy labels

def lf_llm_performative(x):
    """Child is performing scripted/planned content for the camera (Art. 32 UNCRC)"""
    if x.performative == 1:
        return EXPLOIT
    elif x.performative == 0:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_llm_emotional_bait(x):
    """Title uses emotional manipulation/clickbait to drive engagement (Art. 19 UNCRC)"""
    if x.emotional_bait == 1:
        return EXPLOIT
    return ABSTAIN  # Absence of emotional bait doesn't mean not exploitative

def lf_llm_narrative_conflict(x):
    """Title contains manufactured narrative conflict (Art. 19 UNCRC)"""
    if x.narrative_conflict == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_challenge(x):
    """Content is a challenge/competition format requiring child labor (Art. 32 UNCRC)"""
    if x.challenge_format == 1:
        return EXPLOIT
    return ABSTAIN

def lf_llm_commercial(x):
    """Content contains commercial product placement exploiting child (Art. 13 UNCRC)"""
    if x.commercial_content == 1:
        return EXPLOIT
    return ABSTAIN

# --- Category 2: Rule-based Metadata Labeling Functions ---
# These use title formatting and keyword patterns

def lf_all_caps_ratio(x):
    """Titles with high ALL CAPS ratio indicate clickbait exploitation strategy"""
    title = str(x.title)
    if len(title) == 0:
        return ABSTAIN
    upper_ratio = sum(1 for c in title if c.isupper()) / max(len(title.replace(' ', '')), 1)
    if upper_ratio > 0.7:
        return EXPLOIT
    elif upper_ratio < 0.2:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_exclamation_marks(x):
    """Excessive exclamation marks indicate emotional manipulation"""
    title = str(x.title)
    excl_count = title.count('!')
    if excl_count >= 3:
        return EXPLOIT
    elif excl_count == 0:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_conflict_keywords(x):
    """Keywords indicating manufactured conflict/drama"""
    title = str(x.title).lower()
    conflict_words = ['stole', 'stolen', 'broke', 'destroyed', 'ruined', 'caught', 
                      'exposed', 'confronting', 'fight', 'kicked out', 'gone wrong',
                      'arrested', 'called the police', 'emergency', 'hospital']
    if any(w in title for w in conflict_words):
        return EXPLOIT
    return ABSTAIN

def lf_challenge_keywords(x):
    """Keywords indicating challenge/competition format"""
    title = str(x.title).lower()
    challenge_words = ['challenge', '24 hours', '24 hour', 'last to', 'first to',
                       'vs', 'versus', 'competition', 'wins', 'loses', 'dare']
    if any(w in title for w in challenge_words):
        return EXPLOIT
    return ABSTAIN

def lf_prank_keywords(x):
    """Pranks on children indicate exploitation (Art. 19 - freedom from harm)"""
    title = str(x.title).lower()
    prank_words = ['prank', 'trick', 'scare', 'scared', 'surprise attack', 'revenge']
    if any(w in title for w in prank_words):
        return EXPLOIT
    return ABSTAIN

def lf_emotional_keywords(x):
    """Emotional exploitation keywords - using child's emotions for content"""
    title = str(x.title).lower()
    emotion_words = ['cried', 'crying', 'tears', 'heartbroken', 'devastated', 
                     'freaked out', 'meltdown', 'tantrum', 'screaming', 'sobbing']
    if any(w in title for w in emotion_words):
        return EXPLOIT
    return ABSTAIN

def lf_organic_keywords(x):
    """Keywords suggesting organic/natural family content (not exploitation)"""
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
                      'story time', 'if i was', 'superhero', 'spy', 'detective']
    if any(w in title for w in roleplay_words):
        return EXPLOIT
    return ABSTAIN

def lf_question_clickbait(x):
    """Clickbait question format designed to maximize engagement"""
    title = str(x.title)
    # "WHO...", "WHAT HAPPENS WHEN...", "IS THIS..."
    if re.match(r'^(WHO|WHAT|WHY|HOW|IS|ARE|DID|WILL|CAN)\b', title):
        return EXPLOIT
    return ABSTAIN

def lf_urgency_words(x):
    """Urgency/sensationalism in title"""
    title = str(x.title).lower()
    urgency_words = ['shocking', 'you won\'t believe', 'insane', 'crazy', 'epic',
                     'worst', 'biggest', 'most dangerous', 'never before']
    if any(w in title for w in urgency_words):
        return EXPLOIT
    return ABSTAIN

# --- Category 3: CV-based Labeling Functions ---
# These use thumbnail visual analysis

def lf_cv_high_saturation(x):
    """High color saturation in thumbnail indicates visual manipulation strategy"""
    if pd.isna(x.get('saturation_mean', np.nan)):
        return ABSTAIN
    if x.saturation_mean > 150:  # High saturation = visually manipulative
        return EXPLOIT
    elif x.saturation_mean < 80:  # Low saturation = natural/organic
        return NOT_EXPLOIT
    return ABSTAIN

def lf_cv_child_distress(x):
    """CV-detected child showing distress/negative emotion in thumbnail"""
    if pd.isna(x.get('child_emotion', np.nan)):
        return ABSTAIN
    emotion = str(x.child_emotion).lower()
    if emotion in ['distressed', 'crying', 'scared', 'sad', 'angry', 'surprised']:
        return EXPLOIT
    elif emotion in ['happy', 'neutral', 'calm']:
        return NOT_EXPLOIT
    return ABSTAIN

def lf_cv_exploitation_concern(x):
    """Vision model flagged exploitation concern in thumbnail"""
    if pd.isna(x.get('exploitation_concern', np.nan)):
        return ABSTAIN
    if x.exploitation_concern == 1 or str(x.exploitation_concern).lower() == 'true':
        return EXPLOIT
    return ABSTAIN

# ============================================================
# BUILD AND APPLY LABELING FUNCTIONS
# ============================================================

print("Building labeling functions...")

# Create LF objects
lfs = [
    LabelingFunction(name="lf_llm_performative", f=lf_llm_performative),
    LabelingFunction(name="lf_llm_emotional_bait", f=lf_llm_emotional_bait),
    LabelingFunction(name="lf_llm_narrative_conflict", f=lf_llm_narrative_conflict),
    LabelingFunction(name="lf_llm_challenge", f=lf_llm_challenge),
    LabelingFunction(name="lf_llm_commercial", f=lf_llm_commercial),
    LabelingFunction(name="lf_all_caps_ratio", f=lf_all_caps_ratio),
    LabelingFunction(name="lf_exclamation_marks", f=lf_exclamation_marks),
    LabelingFunction(name="lf_conflict_keywords", f=lf_conflict_keywords),
    LabelingFunction(name="lf_challenge_keywords", f=lf_challenge_keywords),
    LabelingFunction(name="lf_prank_keywords", f=lf_prank_keywords),
    LabelingFunction(name="lf_emotional_keywords", f=lf_emotional_keywords),
    LabelingFunction(name="lf_organic_keywords", f=lf_organic_keywords),
    LabelingFunction(name="lf_roleplay_keywords", f=lf_roleplay_keywords),
    LabelingFunction(name="lf_question_clickbait", f=lf_question_clickbait),
    LabelingFunction(name="lf_urgency_words", f=lf_urgency_words),
]

# Add CV-based LFs only if data is available
has_cv = 'saturation_mean' in sample.columns
has_vision = 'child_emotion' in sample.columns

if has_cv:
    lfs.append(LabelingFunction(name="lf_cv_high_saturation", f=lf_cv_high_saturation))
if has_vision:
    lfs.append(LabelingFunction(name="lf_cv_child_distress", f=lf_cv_child_distress))
    lfs.append(LabelingFunction(name="lf_cv_exploitation_concern", f=lf_cv_exploitation_concern))

print(f"  Total LFs: {len(lfs)} ({5} LLM + {10} rule-based + {len(lfs)-15} CV-based)")

# Apply LFs to the dataset
print("\nApplying labeling functions...")
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=sample)

print(f"  Label matrix shape: {L_train.shape}")
print(f"  Coverage (at least 1 LF votes): {(L_train != ABSTAIN).any(axis=1).mean():.1%}")

# LF Analysis
print("\n" + "="*60)
print("LABELING FUNCTION ANALYSIS")
print("="*60)
lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(lf_analysis.to_string())

# Save LF analysis
lf_analysis.to_csv(f'{OUTPUT_DIR}/lf_analysis.csv')

# ============================================================
# TRAIN LABEL MODEL
# ============================================================

print("\n" + "="*60)
print("TRAINING LABEL MODEL")
print("="*60)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.01, log_freq=100, seed=42)

# Get probabilistic labels
probs = label_model.predict_proba(L=L_train)
preds = label_model.predict(L=L_train)

# Exploitation score = probability of EXPLOIT class
sample['exploitation_score'] = probs[:, 1]
sample['exploitation_pred'] = preds

print(f"\n  Predictions: EXPLOIT={sum(preds==1)}, NOT_EXPLOIT={sum(preds==0)}, ABSTAIN={sum(preds==-1)}")
print(f"  Mean exploitation score: {sample['exploitation_score'].mean():.3f}")
print(f"  Median exploitation score: {sample['exploitation_score'].median():.3f}")

# ============================================================
# ANALYSIS: EXPLOITATION SCORE vs VIEWS
# ============================================================

print("\n" + "="*60)
print("ANALYSIS: EXPLOITATION SCORE vs ALGORITHMIC REWARD")
print("="*60)

# Filter to videos with views
df = sample[sample['viewCount'] > 0].copy()
df['log_views'] = np.log10(df['viewCount'] + 1)

# 1. Correlation between exploitation score and views
corr, pval = stats.spearmanr(df['exploitation_score'], df['log_views'])
print(f"\n1. Spearman correlation (exploitation_score vs log_views):")
print(f"   rho = {corr:.4f}, p = {pval:.6f}")

# 2. Compare high vs low exploitation videos
high_exploit = df[df['exploitation_score'] > 0.7]
low_exploit = df[df['exploitation_score'] < 0.3]
print(f"\n2. High exploitation (score>0.7): n={len(high_exploit)}, median views={high_exploit['viewCount'].median():,.0f}")
print(f"   Low exploitation (score<0.3): n={len(low_exploit)}, median views={low_exploit['viewCount'].median():,.0f}")
if len(high_exploit) > 5 and len(low_exploit) > 5:
    u_stat, u_pval = stats.mannwhitneyu(high_exploit['viewCount'], low_exploit['viewCount'], alternative='greater')
    boost = high_exploit['viewCount'].median() / max(low_exploit['viewCount'].median(), 1) - 1
    print(f"   View boost: +{boost*100:.1f}%")
    print(f"   Mann-Whitney U test: U={u_stat:.0f}, p={u_pval:.6f}")

# 3. Within-channel analysis
print(f"\n3. Within-channel exploitation premium:")
channel_results = []
for ch, grp in df.groupby('channel_short_name'):
    if len(grp) < 10:
        continue
    ch_high = grp[grp['exploitation_score'] > grp['exploitation_score'].median()]
    ch_low = grp[grp['exploitation_score'] <= grp['exploitation_score'].median()]
    if len(ch_high) >= 3 and len(ch_low) >= 3:
        ratio = ch_high['viewCount'].median() / max(ch_low['viewCount'].median(), 1)
        channel_results.append({
            'channel': ch,
            'n': len(grp),
            'mean_exploit_score': grp['exploitation_score'].mean(),
            'high_exploit_median_views': ch_high['viewCount'].median(),
            'low_exploit_median_views': ch_low['viewCount'].median(),
            'within_channel_boost': (ratio - 1) * 100
        })

ch_df = pd.DataFrame(channel_results)
if len(ch_df) > 0:
    print(f"   Channels analyzed: {len(ch_df)}")
    print(f"   Mean within-channel boost: {ch_df['within_channel_boost'].mean():.1f}%")
    print(f"   Median within-channel boost: {ch_df['within_channel_boost'].median():.1f}%")
    # Sign test
    positive = (ch_df['within_channel_boost'] > 0).sum()
    negative = (ch_df['within_channel_boost'] < 0).sum()
    sign_pval = stats.binomtest(positive, positive + negative, 0.5).pvalue if (positive + negative) > 0 else 1.0
    print(f"   Sign test: {positive}/{positive+negative} channels positive, p={sign_pval:.4f}")
    
    # Print top channels
    print(f"\n   Top 10 channels by within-channel boost:")
    for _, row in ch_df.sort_values('within_channel_boost', ascending=False).head(10).iterrows():
        print(f"   {row['channel']:20s} boost={row['within_channel_boost']:+.1f}% (n={row['n']}, score={row['mean_exploit_score']:.2f})")

# 4. Per-dimension analysis (controlling for other dimensions)
print(f"\n4. Per-dimension view boost (within-channel controlled):")
dimensions = ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content']
dim_results = []
for dim in dimensions:
    within_boosts = []
    for ch, grp in df.groupby('channel_short_name'):
        if len(grp) < 10:
            continue
        has_dim = grp[grp[dim] == 1]
        no_dim = grp[grp[dim] == 0]
        if len(has_dim) >= 3 and len(no_dim) >= 3:
            boost = has_dim['viewCount'].median() / max(no_dim['viewCount'].median(), 1) - 1
            within_boosts.append(boost * 100)
    
    if within_boosts:
        mean_boost = np.mean(within_boosts)
        t_stat, t_pval = stats.ttest_1samp(within_boosts, 0)
        dim_results.append({
            'dimension': dim,
            'n_channels': len(within_boosts),
            'mean_within_boost': mean_boost,
            'median_within_boost': np.median(within_boosts),
            't_stat': t_stat,
            'p_value': t_pval
        })
        sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else ""
        print(f"   {dim:25s}: within-channel boost = {mean_boost:+.1f}% (p={t_pval:.4f}) {sig}")

dim_df = pd.DataFrame(dim_results)
dim_df.to_csv(f'{OUTPUT_DIR}/dimension_within_channel_boost.csv', index=False)

# ============================================================
# GENERATE FIGURES
# ============================================================

print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

# Figure 1: Exploitation score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(sample['exploitation_score'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(0.5, color='red', linestyle='--', label='Decision boundary')
axes[0].set_xlabel('Exploitation Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Exploitation Scores\n(Label Model Output)')
axes[0].legend()

# Figure 2: Score vs Views scatter
ax = axes[1]
scatter = ax.scatter(df['exploitation_score'], df['log_views'], 
                    alpha=0.3, s=10, c=df['exploitation_score'], cmap='RdYlGn_r')
ax.set_xlabel('Exploitation Score')
ax.set_ylabel('log₁₀(Views)')
ax.set_title(f'Exploitation Score vs Views\n(ρ={corr:.3f}, p={pval:.4f})')
plt.colorbar(scatter, ax=ax, label='Exploitation Score')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_score_distribution_and_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_score_distribution_and_views.png")

# Figure 3: Within-channel boost by dimension
if dim_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    dim_df_sorted = dim_df.sort_values('mean_within_boost', ascending=True)
    colors = ['green' if p < 0.05 else 'gray' for p in dim_df_sorted['p_value']]
    bars = ax.barh(dim_df_sorted['dimension'], dim_df_sorted['mean_within_boost'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Within-Channel View Boost (%)')
    ax.set_title('Per-Dimension Within-Channel View Boost\n(Green = p<0.05, Gray = not significant)')
    for i, (_, row) in enumerate(dim_df_sorted.iterrows()):
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
        ax.text(row['mean_within_boost'] + 1, i, f"p={row['p_value']:.3f} {sig}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_within_channel_boost.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_within_channel_boost.png")

# Figure 4: Channel-level exploitation score vs median views
if len(ch_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(ch_df['mean_exploit_score'], 
              np.log10(ch_df['high_exploit_median_views']), 
              s=ch_df['n']*2, alpha=0.7, c='steelblue', edgecolor='white')
    for _, row in ch_df.iterrows():
        ax.annotate(row['channel'], 
                   (row['mean_exploit_score'], np.log10(row['high_exploit_median_views'])),
                   fontsize=7, alpha=0.7)
    ax.set_xlabel('Mean Exploitation Score')
    ax.set_ylabel('log₁₀(Median Views)')
    ax.set_title('Channel-Level: Exploitation Score vs Popularity\n(bubble size = number of videos)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_channel_exploitation_vs_views.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_channel_exploitation_vs_views.png")

# Figure 5: LF coverage and accuracy heatmap
fig, ax = plt.subplots(figsize=(12, 8))
lf_summary = lf_analysis[['Polarity', 'Coverage', 'Overlaps', 'Conflicts']].copy()
# Categorize LFs
categories = []
for name in lf_analysis.index:
    if 'llm' in name:
        categories.append('LLM-based')
    elif 'cv' in name:
        categories.append('CV-based')
    else:
        categories.append('Rule-based')
lf_summary['Category'] = categories

# Plot coverage by category
for cat, color in [('LLM-based', 'steelblue'), ('Rule-based', 'coral'), ('CV-based', 'green')]:
    mask = lf_summary['Category'] == cat
    if mask.any():
        subset = lf_summary[mask]
        ax.barh([f"{idx} [{cat[:3]}]" for idx in subset.index], 
               subset['Coverage'], color=color, alpha=0.7, label=cat)

ax.set_xlabel('Coverage (fraction of dataset labeled)')
ax.set_title('Labeling Function Coverage by Category')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_lf_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_lf_coverage.png")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save the scored dataset
sample.to_csv(f'{OUTPUT_DIR}/videos_with_exploitation_scores.csv', index=False)
print(f"  Saved videos_with_exploitation_scores.csv ({len(sample)} videos)")

# Save channel-level results
if len(ch_df) > 0:
    ch_df.to_csv(f'{OUTPUT_DIR}/channel_exploitation_analysis.csv', index=False)
    print(f"  Saved channel_exploitation_analysis.csv ({len(ch_df)} channels)")

# Save summary statistics
summary = {
    'n_videos': len(sample),
    'n_channels': sample['channel_short_name'].nunique(),
    'n_labeling_functions': len(lfs),
    'n_lf_llm': 5,
    'n_lf_rule': 10,
    'n_lf_cv': len(lfs) - 15,
    'mean_exploitation_score': float(sample['exploitation_score'].mean()),
    'median_exploitation_score': float(sample['exploitation_score'].median()),
    'exploit_predicted': int(sum(preds == 1)),
    'not_exploit_predicted': int(sum(preds == 0)),
    'correlation_score_views': float(corr),
    'correlation_pvalue': float(pval),
    'within_channel_mean_boost': float(ch_df['within_channel_boost'].mean()) if len(ch_df) > 0 else None,
    'within_channel_median_boost': float(ch_df['within_channel_boost'].median()) if len(ch_df) > 0 else None,
}
with open(f'{OUTPUT_DIR}/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved summary_statistics.json")

print("\n✅ Snorkel pipeline complete!")
