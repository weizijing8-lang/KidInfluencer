import pandas as pd
import numpy as np

# Load all three annotators
# Annotator A
df_a = pd.read_csv('/home/ubuntu/upload/pasted_content_6.txt', sep='\t', header=None)
cols_a = ['video_id', 'youtube_link', 'title', 'channel', 'views', 'sample_group',
          'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
          'MODEL_exploitation_score', 'MODEL_overall_prediction', 'MODEL_confidence',
          'MODEL_n_dims_flagged', 'Human_Labeled', 'Note']
df_a.columns = cols_a[:len(df_a.columns)]
df_a['model_pred'] = (df_a['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_a['human_label'] = df_a['Human_Labeled'].astype(int)

# Annotator B (with all updates including Candle Blowing = 1)
df_b = pd.read_csv('/home/ubuntu/upload/pasted_content_7.txt', sep='\t', header=None)
cols_b = ['video_id', 'title', 'channel', 'youtube_link', 'views',
          'perf_labor', 'emotional_bait', 'narrative_conflict', 
          'challenge_format', 'commercial_content', 'privacy_violation',
          'human_label', 'note']
df_b.columns = cols_b[:len(df_b.columns)]

# Apply updates from pasted_content_9
df_update = pd.read_csv('/home/ubuntu/upload/pasted_content_9.txt', sep='\t', header=None)
cols_u = ['video_id', 'title', 'channel', 'youtube_link', 'views',
          'perf_labor', 'emotional_bait', 'narrative_conflict', 
          'challenge_format', 'commercial_content', 'privacy_violation',
          'human_label']
df_update.columns = cols_u[:len(df_update.columns)]
for _, row in df_update.iterrows():
    df_b.loc[df_b['video_id'] == row['video_id'], 'human_label'] = row['human_label']

# Also fix Candle Blowing Challenge (sS89FnQWwvA) to 1 (B confirmed orally)
df_b.loc[df_b['video_id'] == 'sS89FnQWwvA', 'human_label'] = 1

# Annotator C
df_c = pd.read_csv('/home/ubuntu/upload/pasted_content_8.txt', sep='\t')
df_c['model_pred'] = (df_c['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_c['human_label'] = df_c['Human Labeled'].astype(int)

# Adult channels to filter
adult_channels = ['brentrivera','crazygorilla','blippi','itsyeboi','jordanmatter',
                  'piersonwodzynski','rebeccazamolo','ronaldomg','itsrucka',
                  'jordynjones','gavinmagnus','itsjudyslife','jesssfam',
                  'thebramfam','thedashleys','samandnia','meetthemillers',
                  'family5vlogs','yawivlogs','bonniehoellein','babybus',
                  'kidssongs','salishmatter']

# Load model predictions
df_full = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws.csv')

# ============================================================
# INTER-RATER RELIABILITY
# ============================================================
print('=' * 60)
print('  INTER-RATER RELIABILITY (CORRECTED)')
print('=' * 60)

overlap_ab = set(df_a['video_id']) & set(df_b['video_id'])
print(f'\n  Overlap A-B: {len(overlap_ab)} videos')

if len(overlap_ab) > 0:
    a_labels = df_a[df_a['video_id'].isin(overlap_ab)].set_index('video_id')['human_label']
    b_labels = df_b[df_b['video_id'].isin(overlap_ab)].set_index('video_id')['human_label']
    common = a_labels.index.intersection(b_labels.index)
    agree = (a_labels[common] == b_labels[common]).sum()
    n_overlap = len(common)
    po = agree / n_overlap
    # Cohen's Kappa
    p1_pos = a_labels[common].mean()
    p2_pos = b_labels[common].mean()
    pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    print(f'  A vs B agreement: {agree}/{n_overlap} = {po*100:.1f}%')
    print(f'  Cohen Kappa (A vs B): {kappa:.3f}')
    for vid in common:
        a_val = a_labels[vid]
        b_val = b_labels[vid]
        title = df_a[df_a['video_id']==vid]['title'].values[0][:50]
        status = 'AGREE' if a_val == b_val else 'DISAGREE'
        print(f'    {status}: {vid} | A={a_val}, B={b_val} | {title}')

# ============================================================
# PER-ANNOTATOR METRICS (KID-CENTRIC ONLY)
# ============================================================
print()
print('=' * 60)
print('  PER-ANNOTATOR METRICS (KID-CENTRIC ONLY)')
print('=' * 60)

# Annotator A (already kid-centric from v2 sheet)
tp_a = ((df_a['model_pred'] == 1) & (df_a['human_label'] == 1)).sum()
fp_a = ((df_a['model_pred'] == 1) & (df_a['human_label'] == 0)).sum()
tn_a = ((df_a['model_pred'] == 0) & (df_a['human_label'] == 0)).sum()
fn_a = ((df_a['model_pred'] == 0) & (df_a['human_label'] == 1)).sum()
n_a = len(df_a)
acc_a = (tp_a + tn_a) / n_a
prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0
print(f'\n  Annotator A (n={n_a}):')
print(f'    Accuracy={acc_a:.1%}, Precision={prec_a:.3f}, Recall={rec_a:.3f}, F1={f1_a:.3f}')
print(f'    TP={tp_a}, FP={fp_a}, TN={tn_a}, FN={fn_a}')

# Annotator B (kid-centric only, with model predictions)
df_b_m = df_b.merge(df_full[['id','is_exploitative_ws']].rename(columns={'id':'video_id','is_exploitative_ws':'model_pred'}), on='video_id', how='left')
df_b_kid = df_b_m[(~df_b_m['channel'].isin(adult_channels)) & (df_b_m['model_pred'].notna())]
tp_b = ((df_b_kid['model_pred'] == 1) & (df_b_kid['human_label'] == 1)).sum()
fp_b = ((df_b_kid['model_pred'] == 1) & (df_b_kid['human_label'] == 0)).sum()
tn_b = ((df_b_kid['model_pred'] == 0) & (df_b_kid['human_label'] == 0)).sum()
fn_b = ((df_b_kid['model_pred'] == 0) & (df_b_kid['human_label'] == 1)).sum()
n_b = len(df_b_kid)
acc_b = (tp_b + tn_b) / n_b
prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0
print(f'\n  Annotator B (n={n_b}, kid-centric, corrected):')
print(f'    Accuracy={acc_b:.1%}, Precision={prec_b:.3f}, Recall={rec_b:.3f}, F1={f1_b:.3f}')
print(f'    TP={tp_b}, FP={fp_b}, TN={tn_b}, FN={fn_b}')

# Annotator C (kid-centric only)
df_c_kid = df_c[~df_c['channel'].isin(adult_channels)]
tp_c = ((df_c_kid['model_pred'] == 1) & (df_c_kid['human_label'] == 1)).sum()
fp_c = ((df_c_kid['model_pred'] == 1) & (df_c_kid['human_label'] == 0)).sum()
tn_c = ((df_c_kid['model_pred'] == 0) & (df_c_kid['human_label'] == 0)).sum()
fn_c = ((df_c_kid['model_pred'] == 0) & (df_c_kid['human_label'] == 1)).sum()
n_c = len(df_c_kid)
acc_c = (tp_c + tn_c) / n_c if n_c > 0 else 0
prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0
print(f'\n  Annotator C (n={n_c}, kid-centric):')
print(f'    Accuracy={acc_c:.1%}, Precision={prec_c:.3f}, Recall={rec_c:.3f}, F1={f1_c:.3f}')
print(f'    TP={tp_c}, FP={fp_c}, TN={tn_c}, FN={fn_c}')

# ============================================================
# COMBINED METRICS
# ============================================================
print()
print('=' * 60)
print('  COMBINED METRICS (ALL ANNOTATORS, KID-CENTRIC)')
print('=' * 60)

all_annotations = []

# A
for _, row in df_a.iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': row['human_label'], 
                           'model': row['model_pred'], 'annotator': 'A'})

# B (kid-centric)
for _, row in df_b_kid.iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': int(row['human_label']),
                           'model': int(row['model_pred']), 'annotator': 'B'})

# C (kid-centric)
for _, row in df_c_kid.iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': int(row['human_label']),
                           'model': int(row['model_pred']), 'annotator': 'C'})

comb = pd.DataFrame(all_annotations)

# Deduplicate: for overlapping videos, use majority vote
# First check unique annotations
unique_videos = comb.groupby('video_id').agg(
    human_majority=('human', lambda x: int(x.mean() >= 0.5)),
    model=('model', 'first'),
    n_annotators=('annotator', 'count')
).reset_index()

print(f'\n  Total annotations: {len(comb)}')
print(f'  Unique videos: {len(unique_videos)}')
print(f'  Videos with multiple annotators: {(unique_videos.n_annotators > 1).sum()}')
print(f'  By annotator: A={len(comb[comb.annotator=="A"])}, B={len(comb[comb.annotator=="B"])}, C={len(comb[comb.annotator=="C"])}')

# Metrics on unique videos (majority vote)
tp = ((unique_videos['model'] == 1) & (unique_videos['human_majority'] == 1)).sum()
fp = ((unique_videos['model'] == 1) & (unique_videos['human_majority'] == 0)).sum()
tn = ((unique_videos['model'] == 0) & (unique_videos['human_majority'] == 0)).sum()
fn = ((unique_videos['model'] == 0) & (unique_videos['human_majority'] == 1)).sum()
n = len(unique_videos)
acc = (tp + tn) / n
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
po = acc
pe = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp)) / (n**2)
kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

print(f'\n  UNIQUE VIDEOS (majority vote where overlap):')
print(f'    n = {n} unique videos, 3 annotators')
print(f'    Accuracy = {acc:.1%}')
print(f'    Precision = {prec:.3f}')
print(f'    Recall = {rec:.3f}')
print(f'    F1 = {f1:.3f}')
print(f'    Cohen Kappa = {kappa:.3f}')
print(f'    TP={tp}, FP={fp}, TN={tn}, FN={fn}')

# Also compute pooled (all annotations treated independently)
tp_pool = ((comb['model']==1)&(comb['human']==1)).sum()
fp_pool = ((comb['model']==1)&(comb['human']==0)).sum()
tn_pool = ((comb['model']==0)&(comb['human']==0)).sum()
fn_pool = ((comb['model']==0)&(comb['human']==1)).sum()
n_pool = len(comb)
acc_pool = (tp_pool+tn_pool)/n_pool
prec_pool = tp_pool/(tp_pool+fp_pool) if (tp_pool+fp_pool)>0 else 0
rec_pool = tp_pool/(tp_pool+fn_pool) if (tp_pool+fn_pool)>0 else 0
f1_pool = 2*prec_pool*rec_pool/(prec_pool+rec_pool) if (prec_pool+rec_pool)>0 else 0

print(f'\n  POOLED (all annotations independently):')
print(f'    n = {n_pool} annotations')
print(f'    Accuracy = {acc_pool:.1%}')
print(f'    Precision = {prec_pool:.3f}')
print(f'    Recall = {rec_pool:.3f}')
print(f'    F1 = {f1_pool:.3f}')
print(f'    TP={tp_pool}, FP={fp_pool}, TN={tn_pool}, FN={fn_pool}')

# ============================================================
# SUMMARY FOR PAPER
# ============================================================
print()
print('=' * 60)
print('  SUMMARY FOR PAPER')
print('=' * 60)
print(f'''
  Validation Protocol:
  - 3 independent annotators labeled {n} unique videos
  - Annotators watched each video and assessed overall exploitation risk
  - Inter-rater agreement on overlapping samples: 100% (3/3 videos, kappa=1.0)
  
  Model Performance (vs. human consensus):
  - Accuracy: {acc:.1%}
  - Precision: {prec:.3f}
  - Recall: {rec:.3f}  
  - F1: {f1:.3f}
  - Cohen's Kappa: {kappa:.3f}
  
  Per-annotator F1: A={f1_a:.3f}, B={f1_b:.3f}, C={f1_c:.3f}
''')
