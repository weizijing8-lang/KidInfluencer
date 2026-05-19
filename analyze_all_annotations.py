"""
Comprehensive analysis of all three annotators' labels.
- Annotator A (pasted_content_6.txt): ~49 videos, structured format with model predictions
- Annotator B (pasted_content_7.txt): ~74 videos, different format (per-dimension + overall)
- Annotator C (pasted_content_8.txt): ~19 videos, same format as annotator A (v1 sheet)

Compute:
1. Per-annotator model accuracy
2. Inter-rater reliability (on overlapping videos)
3. Combined validation metrics
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ============================================================
# ANNOTATOR A (you - 49 videos, kid-centric only, v2 sheet)
# ============================================================
df_a = pd.read_csv('/home/ubuntu/upload/pasted_content_6.txt', sep='\t', header=None)
cols_a = ['video_id', 'youtube_link', 'title', 'channel', 'views', 'sample_group',
          'MODEL_performative_labor_prob', 'MODEL_performative_labor_pred',
          'MODEL_emotional_bait_prob', 'MODEL_emotional_bait_pred',
          'MODEL_narrative_conflict_prob', 'MODEL_narrative_conflict_pred',
          'MODEL_challenge_format_prob', 'MODEL_challenge_format_pred',
          'MODEL_commercial_content_prob', 'MODEL_commercial_content_pred',
          'MODEL_privacy_violation_prob', 'MODEL_privacy_violation_pred',
          'MODEL_exploitation_score', 'MODEL_overall_prediction', 'MODEL_confidence',
          'MODEL_n_dims_flagged', 'Human_Labeled', 'Note']
df_a.columns = cols_a[:len(df_a.columns)]
df_a['model_pred'] = (df_a['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_a['human_label'] = df_a['Human_Labeled'].astype(int)
df_a['annotator'] = 'A'
print(f"Annotator A: {len(df_a)} videos")

# ============================================================
# ANNOTATOR B (74 videos, different format)
# ============================================================
df_b_raw = pd.read_csv('/home/ubuntu/upload/pasted_content_7.txt', sep='\t', header=None)
# Format: video_id, title, channel, youtube_link, views, dim1-6 flags, overall_label, note
# Columns: 0=video_id, 1=title, 2=channel, 3=youtube_link, 4=views, 
#           5=performative, 6=emotional, 7=narrative, 8=challenge, 9=commercial, 10=privacy, 11=overall, 12=note
cols_b = ['video_id', 'title', 'channel', 'youtube_link', 'views',
          'perf_labor', 'emotional_bait', 'narrative_conflict', 
          'challenge_format', 'commercial_content', 'privacy_violation',
          'human_label', 'note']
if len(df_b_raw.columns) >= 13:
    df_b_raw.columns = cols_b[:len(df_b_raw.columns)]
elif len(df_b_raw.columns) == 12:
    df_b_raw.columns = cols_b[:12]
else:
    # Try to figure out the format
    print(f"  Warning: Annotator B has {len(df_b_raw.columns)} columns")
    df_b_raw.columns = [f'col_{i}' for i in range(len(df_b_raw.columns))]

df_b = pd.DataFrame()
df_b['video_id'] = df_b_raw['video_id']
df_b['title'] = df_b_raw['title']
df_b['channel'] = df_b_raw['channel']
df_b['human_label'] = pd.to_numeric(df_b_raw['human_label'], errors='coerce').fillna(0).astype(int)
df_b['annotator'] = 'B'
print(f"Annotator B: {len(df_b)} videos")

# ============================================================
# ANNOTATOR C (19 videos, same format as v1 annotation sheet)
# ============================================================
df_c = pd.read_csv('/home/ubuntu/upload/pasted_content_8.txt', sep='\t')
df_c['model_pred'] = (df_c['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_c['human_label'] = df_c['Human Labeled'].astype(int)
df_c['annotator'] = 'C'
print(f"Annotator C: {len(df_c)} videos")
print()

# ============================================================
# MODEL ACCURACY PER ANNOTATOR
# ============================================================
print("=" * 60)
print("  MODEL ACCURACY PER ANNOTATOR")
print("=" * 60)

# Annotator A
tp_a = ((df_a['model_pred'] == 1) & (df_a['human_label'] == 1)).sum()
fp_a = ((df_a['model_pred'] == 1) & (df_a['human_label'] == 0)).sum()
tn_a = ((df_a['model_pred'] == 0) & (df_a['human_label'] == 0)).sum()
fn_a = ((df_a['model_pred'] == 0) & (df_a['human_label'] == 1)).sum()
acc_a = (tp_a + tn_a) / len(df_a)
prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0
print(f"\n  Annotator A (n={len(df_a)}, kid-centric only):")
print(f"    Accuracy={acc_a:.1%}, Precision={prec_a:.3f}, Recall={rec_a:.3f}, F1={f1_a:.3f}")
print(f"    TP={tp_a}, FP={fp_a}, TN={tn_a}, FN={fn_a}")

# Annotator C
tp_c = ((df_c['model_pred'] == 1) & (df_c['human_label'] == 1)).sum()
fp_c = ((df_c['model_pred'] == 1) & (df_c['human_label'] == 0)).sum()
tn_c = ((df_c['model_pred'] == 0) & (df_c['human_label'] == 0)).sum()
fn_c = ((df_c['model_pred'] == 0) & (df_c['human_label'] == 1)).sum()
acc_c = (tp_c + tn_c) / len(df_c)
prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0
print(f"\n  Annotator C (n={len(df_c)}, includes adult channels):")
print(f"    Accuracy={acc_c:.1%}, Precision={prec_c:.3f}, Recall={rec_c:.3f}, F1={f1_c:.3f}")
print(f"    TP={tp_c}, FP={fp_c}, TN={tn_c}, FN={fn_c}")

# Annotator C - filtered (kid-centric only)
adult_channels = ['brentrivera','crazygorilla','blippi','itsyeboi','jordanmatter',
                  'piersonwodzynski','rebeccazamolo','ronaldomg','itsrucka',
                  'jordynjones','gavinmagnus','itsjudyslife','jesssfam',
                  'thebramfam','thedashleys','samandnia','meetthemillers',
                  'family5vlogs','yawivlogs','bonniehoellein','babybus',
                  'kidssongs','salishmatter']
df_c_kid = df_c[~df_c['channel'].isin(adult_channels)]
if len(df_c_kid) > 0:
    tp_ck = ((df_c_kid['model_pred'] == 1) & (df_c_kid['human_label'] == 1)).sum()
    fp_ck = ((df_c_kid['model_pred'] == 1) & (df_c_kid['human_label'] == 0)).sum()
    tn_ck = ((df_c_kid['model_pred'] == 0) & (df_c_kid['human_label'] == 0)).sum()
    fn_ck = ((df_c_kid['model_pred'] == 0) & (df_c_kid['human_label'] == 1)).sum()
    acc_ck = (tp_ck + tn_ck) / len(df_c_kid)
    prec_ck = tp_ck / (tp_ck + fp_ck) if (tp_ck + fp_ck) > 0 else 0
    rec_ck = tp_ck / (tp_ck + fn_ck) if (tp_ck + fn_ck) > 0 else 0
    f1_ck = 2 * prec_ck * rec_ck / (prec_ck + rec_ck) if (prec_ck + rec_ck) > 0 else 0
    print(f"\n  Annotator C filtered (n={len(df_c_kid)}, kid-centric only):")
    print(f"    Accuracy={acc_ck:.1%}, Precision={prec_ck:.3f}, Recall={rec_ck:.3f}, F1={f1_ck:.3f}")
    print(f"    TP={tp_ck}, FP={fp_ck}, TN={tn_ck}, FN={fn_ck}")

# ============================================================
# INTER-RATER RELIABILITY
# ============================================================
print()
print("=" * 60)
print("  INTER-RATER RELIABILITY")
print("=" * 60)

# Find overlapping videos between annotators
# A uses video_id, B uses video_id, C uses video_id
all_a_ids = set(df_a['video_id'].values)
all_b_ids = set(df_b['video_id'].values)
all_c_ids = set(df_c['video_id'].values)

overlap_ab = all_a_ids & all_b_ids
overlap_ac = all_a_ids & all_c_ids
overlap_bc = all_b_ids & all_c_ids
overlap_abc = all_a_ids & all_b_ids & all_c_ids

print(f"\n  Overlap A-B: {len(overlap_ab)} videos")
print(f"  Overlap A-C: {len(overlap_ac)} videos")
print(f"  Overlap B-C: {len(overlap_bc)} videos")
print(f"  Overlap A-B-C: {len(overlap_abc)} videos")

# Compute agreement for each pair
def compute_agreement(labels1, labels2):
    """Compute percent agreement and Cohen's Kappa"""
    n = len(labels1)
    if n == 0:
        return 0, 0, 0
    agree = (labels1 == labels2).sum()
    po = agree / n
    # Cohen's Kappa
    p1_pos = labels1.mean()
    p2_pos = labels2.mean()
    pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    return po, kappa, n

# A vs B overlap
if len(overlap_ab) > 0:
    merged_ab = df_a[df_a['video_id'].isin(overlap_ab)][['video_id', 'human_label']].merge(
        df_b[df_b['video_id'].isin(overlap_ab)][['video_id', 'human_label']],
        on='video_id', suffixes=('_A', '_B'))
    po_ab, kappa_ab, n_ab = compute_agreement(
        merged_ab['human_label_A'].values, merged_ab['human_label_B'].values)
    print(f"\n  A vs B (n={n_ab}): Agreement={po_ab:.1%}, Kappa={kappa_ab:.3f}")
    # Show disagreements
    disagree = merged_ab[merged_ab['human_label_A'] != merged_ab['human_label_B']]
    if len(disagree) > 0:
        print(f"    Disagreements:")
        for _, row in disagree.iterrows():
            title = df_a[df_a['video_id'] == row['video_id']]['title'].values[0][:50]
            print(f"      {row['video_id']}: A={row['human_label_A']}, B={row['human_label_B']} | {title}")

# A vs C overlap
if len(overlap_ac) > 0:
    merged_ac = df_a[df_a['video_id'].isin(overlap_ac)][['video_id', 'human_label']].merge(
        df_c[df_c['video_id'].isin(overlap_ac)][['video_id', 'human_label']],
        on='video_id', suffixes=('_A', '_C'))
    po_ac, kappa_ac, n_ac = compute_agreement(
        merged_ac['human_label_A'].values, merged_ac['human_label_C'].values)
    print(f"\n  A vs C (n={n_ac}): Agreement={po_ac:.1%}, Kappa={kappa_ac:.3f}")
    disagree = merged_ac[merged_ac['human_label_A'] != merged_ac['human_label_C']]
    if len(disagree) > 0:
        print(f"    Disagreements:")
        for _, row in disagree.iterrows():
            title = df_a[df_a['video_id'] == row['video_id']]['title'].values[0][:50]
            print(f"      {row['video_id']}: A={row['human_label_A']}, C={row['human_label_C']} | {title}")

# B vs C overlap
if len(overlap_bc) > 0:
    merged_bc = df_b[df_b['video_id'].isin(overlap_bc)][['video_id', 'human_label']].merge(
        df_c[df_c['video_id'].isin(overlap_bc)][['video_id', 'human_label']],
        on='video_id', suffixes=('_B', '_C'))
    po_bc, kappa_bc, n_bc = compute_agreement(
        merged_bc['human_label_B'].values, merged_bc['human_label_C'].values)
    print(f"\n  B vs C (n={n_bc}): Agreement={po_bc:.1%}, Kappa={kappa_bc:.3f}")
    disagree = merged_bc[merged_bc['human_label_B'] != merged_bc['human_label_C']]
    if len(disagree) > 0:
        print(f"    Disagreements:")
        for _, row in disagree.iterrows():
            title = df_b[df_b['video_id'] == row['video_id']]['title'].values[0][:50]
            print(f"      {row['video_id']}: B={row['human_label_B']}, C={row['human_label_C']} | {title}")

# ============================================================
# COMBINED METRICS (all unique videos, majority vote where overlap)
# ============================================================
print()
print("=" * 60)
print("  COMBINED VALIDATION SUMMARY")
print("=" * 60)

# Total unique videos annotated
all_annotated_ids = all_a_ids | all_b_ids | all_c_ids
print(f"\n  Total unique videos annotated: {len(all_annotated_ids)}")
print(f"  By Annotator A: {len(all_a_ids)}")
print(f"  By Annotator B: {len(all_b_ids)}")
print(f"  By Annotator C: {len(all_c_ids)}")

# Annotator B - compute model accuracy
# Need to get model predictions for annotator B's videos
# Load the full classified dataset to get model predictions
df_full = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_proper/classified_videos_ws.csv')
df_b_with_model = df_b.merge(
    df_full[['id', 'is_exploitative_ws']].rename(columns={'id': 'video_id', 'is_exploitative_ws': 'model_pred'}),
    on='video_id', how='left')
df_b_valid = df_b_with_model[df_b_with_model['model_pred'].notna()]
if len(df_b_valid) > 0:
    tp_b = ((df_b_valid['model_pred'] == 1) & (df_b_valid['human_label'] == 1)).sum()
    fp_b = ((df_b_valid['model_pred'] == 1) & (df_b_valid['human_label'] == 0)).sum()
    tn_b = ((df_b_valid['model_pred'] == 0) & (df_b_valid['human_label'] == 0)).sum()
    fn_b = ((df_b_valid['model_pred'] == 0) & (df_b_valid['human_label'] == 1)).sum()
    acc_b = (tp_b + tn_b) / len(df_b_valid)
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0
    print(f"\n  Annotator B vs Model (n={len(df_b_valid)}, all channels):")
    print(f"    Accuracy={acc_b:.1%}, Precision={prec_b:.3f}, Recall={rec_b:.3f}, F1={f1_b:.3f}")
    print(f"    TP={tp_b}, FP={fp_b}, TN={tn_b}, FN={fn_b}")
    
    # Filtered to kid-centric
    df_b_kid = df_b_valid[~df_b_valid['channel'].isin(adult_channels)]
    if len(df_b_kid) > 0:
        tp_bk = ((df_b_kid['model_pred'] == 1) & (df_b_kid['human_label'] == 1)).sum()
        fp_bk = ((df_b_kid['model_pred'] == 1) & (df_b_kid['human_label'] == 0)).sum()
        tn_bk = ((df_b_kid['model_pred'] == 0) & (df_b_kid['human_label'] == 0)).sum()
        fn_bk = ((df_b_kid['model_pred'] == 0) & (df_b_kid['human_label'] == 1)).sum()
        acc_bk = (tp_bk + tn_bk) / len(df_b_kid)
        prec_bk = tp_bk / (tp_bk + fp_bk) if (tp_bk + fp_bk) > 0 else 0
        rec_bk = tp_bk / (tp_bk + fn_bk) if (tp_bk + fn_bk) > 0 else 0
        f1_bk = 2 * prec_bk * rec_bk / (prec_bk + rec_bk) if (prec_bk + rec_bk) > 0 else 0
        print(f"\n  Annotator B vs Model (n={len(df_b_kid)}, kid-centric only):")
        print(f"    Accuracy={acc_bk:.1%}, Precision={prec_bk:.3f}, Recall={rec_bk:.3f}, F1={f1_bk:.3f}")
        print(f"    TP={tp_bk}, FP={fp_bk}, TN={tn_bk}, FN={fn_bk}")

# ============================================================
# OVERALL COMBINED (kid-centric only)
# ============================================================
print()
print("=" * 60)
print("  OVERALL COMBINED (KID-CENTRIC CHANNELS ONLY)")
print("=" * 60)

# Combine all annotations for kid-centric channels
combined = []

# Annotator A (already kid-centric)
for _, row in df_a.iterrows():
    combined.append({'video_id': row['video_id'], 'human_label': row['human_label'], 
                     'model_pred': row['model_pred'], 'annotator': 'A'})

# Annotator B (filter to kid-centric, add model pred)
for _, row in df_b_with_model[~df_b_with_model['channel'].isin(adult_channels)].iterrows():
    if pd.notna(row.get('model_pred')):
        combined.append({'video_id': row['video_id'], 'human_label': int(row['human_label']),
                         'model_pred': int(row['model_pred']), 'annotator': 'B'})

# Annotator C (filter to kid-centric)
for _, row in df_c[~df_c['channel'].isin(adult_channels)].iterrows():
    combined.append({'video_id': row['video_id'], 'human_label': int(row['human_label']),
                     'model_pred': int(row['model_pred']), 'annotator': 'C'})

combined_df = pd.DataFrame(combined)
print(f"\n  Total annotations (kid-centric): {len(combined_df)}")
print(f"  Unique videos: {combined_df['video_id'].nunique()}")
print(f"  By annotator: A={len(combined_df[combined_df.annotator=='A'])}, "
      f"B={len(combined_df[combined_df.annotator=='B'])}, "
      f"C={len(combined_df[combined_df.annotator=='C'])}")

# Overall metrics
tp_all = ((combined_df['model_pred'] == 1) & (combined_df['human_label'] == 1)).sum()
fp_all = ((combined_df['model_pred'] == 1) & (combined_df['human_label'] == 0)).sum()
tn_all = ((combined_df['model_pred'] == 0) & (combined_df['human_label'] == 0)).sum()
fn_all = ((combined_df['model_pred'] == 0) & (combined_df['human_label'] == 1)).sum()
acc_all = (tp_all + tn_all) / len(combined_df)
prec_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
rec_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
f1_all = 2 * prec_all * rec_all / (prec_all + rec_all) if (prec_all + rec_all) > 0 else 0

# Cohen's Kappa
po = acc_all
pe = ((tp_all+fp_all)*(tp_all+fn_all) + (tn_all+fn_all)*(tn_all+fp_all)) / (len(combined_df)**2)
kappa_all = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

print(f"\n  COMBINED METRICS (kid-centric, all annotators):")
print(f"    n = {len(combined_df)} annotations, {combined_df['video_id'].nunique()} unique videos")
print(f"    Accuracy = {acc_all:.1%}")
print(f"    Precision = {prec_all:.3f}")
print(f"    Recall = {rec_all:.3f}")
print(f"    F1 = {f1_all:.3f}")
print(f"    Cohen's Kappa = {kappa_all:.3f}")
print(f"    TP={tp_all}, FP={fp_all}, TN={tn_all}, FN={fn_all}")
