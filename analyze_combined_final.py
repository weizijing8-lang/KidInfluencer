import pandas as pd
import numpy as np

# Load all three annotators with updates applied
# Annotator A
df_a = pd.read_csv('/home/ubuntu/upload/pasted_content_6.txt', sep='\t', header=None)
cols_a = ['video_id', 'youtube_link', 'title', 'channel', 'views', 'sample_group',
          'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',
          'MODEL_exploitation_score', 'MODEL_overall_prediction', 'MODEL_confidence',
          'MODEL_n_dims_flagged', 'Human_Labeled', 'Note']
df_a.columns = cols_a[:len(df_a.columns)]
df_a['model_pred'] = (df_a['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_a['human_label'] = df_a['Human_Labeled'].astype(int)

# Annotator B (with updates)
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

# Annotator C
df_c = pd.read_csv('/home/ubuntu/upload/pasted_content_8.txt', sep='\t')
df_c['model_pred'] = (df_c['MODEL_overall_prediction'] == 'EXPLOITATIVE').astype(int)
df_c['human_label'] = df_c['Human Labeled'].astype(int)

# Inter-rater: find overlaps
overlap_ab = set(df_a['video_id']) & set(df_b['video_id'])
overlap_ac = set(df_a['video_id']) & set(df_c['video_id'])
overlap_bc = set(df_b['video_id']) & set(df_c['video_id'])

print('=== INTER-RATER RELIABILITY (UPDATED) ===')
print(f'Overlap A-B: {len(overlap_ab)} videos')
print(f'Overlap A-C: {len(overlap_ac)} videos')
print(f'Overlap B-C: {len(overlap_bc)} videos')

if len(overlap_ab) > 0:
    a_labels = df_a[df_a['video_id'].isin(overlap_ab)].set_index('video_id')['human_label']
    b_labels = df_b[df_b['video_id'].isin(overlap_ab)].set_index('video_id')['human_label']
    common = a_labels.index.intersection(b_labels.index)
    agree = (a_labels[common] == b_labels[common]).sum()
    print(f'  A vs B agreement: {agree}/{len(common)} = {agree/len(common)*100:.1f}%')
    for vid in common:
        if a_labels[vid] != b_labels[vid]:
            title = df_a[df_a['video_id']==vid]['title'].values[0][:50]
            print(f'    Disagree: {vid} | A={a_labels[vid]}, B={b_labels[vid]} | {title}')

if len(overlap_bc) > 0:
    b_labels2 = df_b[df_b['video_id'].isin(overlap_bc)].set_index('video_id')['human_label']
    c_labels = df_c[df_c['video_id'].isin(overlap_bc)].set_index('video_id')['human_label']
    common = b_labels2.index.intersection(c_labels.index)
    agree = (b_labels2[common] == c_labels[common]).sum()
    print(f'  B vs C agreement: {agree}/{len(common)} = {agree/len(common)*100:.1f}%')

# Combined metrics
print()
print('=' * 60)
print('  FINAL COMBINED METRICS FOR PAPER')
print('=' * 60)

adult_channels = ['brentrivera','crazygorilla','blippi','itsyeboi','jordanmatter',
                  'piersonwodzynski','rebeccazamolo','ronaldomg','itsrucka',
                  'jordynjones','gavinmagnus','itsjudyslife','jesssfam',
                  'thebramfam','thedashleys','samandnia','meetthemillers',
                  'family5vlogs','yawivlogs','bonniehoellein','babybus',
                  'kidssongs','salishmatter']

df_full = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws.csv')

all_annotations = []

# A (already kid-centric)
for _, row in df_a.iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': row['human_label'], 
                           'model': row['model_pred'], 'annotator': 'A', 'channel': row['channel']})

# B (filter + add model)
df_b_m = df_b.merge(df_full[['id','is_exploitative_ws']].rename(columns={'id':'video_id','is_exploitative_ws':'model_pred'}), on='video_id', how='left')
for _, row in df_b_m[~df_b_m['channel'].isin(adult_channels)].dropna(subset=['model_pred']).iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': int(row['human_label']),
                           'model': int(row['model_pred']), 'annotator': 'B', 'channel': row['channel']})

# C (filter)
for _, row in df_c[~df_c['channel'].isin(adult_channels)].iterrows():
    all_annotations.append({'video_id': row['video_id'], 'human': int(row['human_label']),
                           'model': int(row['model_pred']), 'annotator': 'C', 'channel': row['channel']})

comb = pd.DataFrame(all_annotations)
print(f'  Total annotations (kid-centric): {len(comb)}')
print(f'  Unique videos: {comb.video_id.nunique()}')
print(f'  Annotators: A={len(comb[comb.annotator=="A"])}, B={len(comb[comb.annotator=="B"])}, C={len(comb[comb.annotator=="C"])}')
print()

for ann in ['A', 'B', 'C']:
    sub = comb[comb['annotator'] == ann]
    if len(sub) == 0:
        continue
    tp = ((sub['model']==1)&(sub['human']==1)).sum()
    fp = ((sub['model']==1)&(sub['human']==0)).sum()
    tn = ((sub['model']==0)&(sub['human']==0)).sum()
    fn = ((sub['model']==0)&(sub['human']==1)).sum()
    n = len(sub)
    acc = (tp+tn)/n
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f'  Annotator {ann} (n={n}): Acc={acc:.1%}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}')

tp = ((comb['model']==1)&(comb['human']==1)).sum()
fp = ((comb['model']==1)&(comb['human']==0)).sum()
tn = ((comb['model']==0)&(comb['human']==0)).sum()
fn = ((comb['model']==0)&(comb['human']==1)).sum()
n = len(comb)
acc = (tp+tn)/n
prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
po = acc
pe = ((tp+fp)*(tp+fn) + (tn+fn)*(tn+fp)) / (n**2)
kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
print()
print(f'  COMBINED (all annotators, kid-centric):')
print(f'    n={n} annotations, {comb.video_id.nunique()} unique videos, 3 annotators')
print(f'    Accuracy = {acc:.1%}')
print(f'    Precision = {prec:.3f}')
print(f'    Recall = {rec:.3f}')
print(f'    F1 = {f1:.3f}')
print(f'    Cohen Kappa = {kappa:.3f}')
print(f'    TP={tp}, FP={fp}, TN={tn}, FN={fn}')
