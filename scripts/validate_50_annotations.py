"""
Validate multimodal pipeline predictions against 50 human annotations.
Computes accuracy, F1, precision, recall, Cohen's kappa per dimension and overall.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Load human annotations
human = pd.read_csv('/home/ubuntu/KidInfluencer/data/human_annotations_50.tsv', sep='\t')
print(f"Human annotations loaded: {len(human)} videos")

# Fill NaN with 0 for dimension columns
dim_cols_human = ['annotator_performative_labor', 'annotator_emotional_bait', 
                  'annotator_narrative_conflict', 'annotator_challenge_format',
                  'annotator_commercial_content', 'annotator_privacy_violation',
                  'annotator_overall_exploitative']
for col in dim_cols_human:
    human[col] = pd.to_numeric(human[col], errors='coerce').fillna(0).astype(int)

print(f"\nHuman label distribution (overall):")
print(f"  Exploitative: {human['annotator_overall_exploitative'].sum()}")
print(f"  Clean: {(human['annotator_overall_exploitative'] == 0).sum()}")

# Load vision classifications
# Load merged LLM+Vision data
merged_data = pd.read_csv('/home/ubuntu/KidInfluencer/data/merged_llm_vision.csv')
print(f"\nMerged LLM+Vision loaded: {len(merged_data)} videos")
print(f"Columns: {list(merged_data.columns)}")

# Deduplicate pipeline data before merging
merged_data = merged_data.drop_duplicates(subset='video_id', keep='first')
print(f"After dedup: {len(merged_data)} videos")

# Merge human with pipeline data
merged = human.merge(merged_data, on='video_id', how='left', suffixes=('', '_pipe'))
matched = merged['overall_exploitative'].notna().sum()
print(f"\nMatched with pipeline: {matched}/{len(human)}")

# Map dimension names to actual column names
all_cols = list(merged.columns)
print(f"\nAll merged columns: {all_cols}")

dim_mapping = {
    'performative_labor': {
        'human': 'annotator_performative_labor',
        'vision': 'performative_labor_vision',
        'llm': 'performative_labor_llm'
    },
    'emotional_bait': {
        'human': 'annotator_emotional_bait', 
        'vision': 'emotional_bait_vision',
        'llm': 'emotional_bait_llm'
    },
    'narrative_conflict': {
        'human': 'annotator_narrative_conflict',
        'vision': 'narrative_conflict_vision', 
        'llm': 'narrative_conflict_llm'
    },
    'challenge_format': {
        'human': 'annotator_challenge_format',
        'vision': 'challenge_format_vision',
        'llm': 'challenge_format_llm'
    },
    'commercial_content': {
        'human': 'annotator_commercial_content',
        'vision': 'commercial_content_vision',
        'llm': 'commercial_content_llm'
    },
    'privacy_violation': {
        'human': 'annotator_privacy_violation',
        'vision': 'privacy_violation_vision',
        'llm': 'privacy_violation_llm'
    }
}

# Compute combined scores and binary predictions
results = {}
for dim_name, cols in dim_mapping.items():
    human_col = cols['human']
    vision_col = cols['vision']
    llm_col = cols['llm']
    
    h = merged[human_col].values
    
    # Vision score (continuous 0-1)
    if vision_col in merged.columns:
        v = pd.to_numeric(merged[vision_col], errors='coerce').fillna(0).values
    else:
        v = np.zeros(len(merged))
    
    # LLM score (binary 0/1)
    if llm_col in merged.columns:
        l = pd.to_numeric(merged[llm_col], errors='coerce').fillna(0).values
    else:
        l = np.zeros(len(merged))
    
    # Combined score: 0.67 * vision + 0.33 * LLM
    combined = 0.67 * v + 0.33 * l
    pred = (combined >= 0.5).astype(int)
    
    # Also compute vision-only prediction
    v_pred = (v >= 0.5).astype(int)
    
    n_human_pos = int(h.sum())
    n_pred_pos = int(pred.sum())
    n_vision_pos = int(v_pred.sum())
    
    if n_human_pos > 0 and n_human_pos < len(h):
        acc = accuracy_score(h, pred)
        f1 = f1_score(h, pred, zero_division=0)
        prec = precision_score(h, pred, zero_division=0)
        rec = recall_score(h, pred, zero_division=0)
        kappa = cohen_kappa_score(h, pred)
        cm = confusion_matrix(h, pred, labels=[0,1])
        
        # Vision-only metrics
        v_acc = accuracy_score(h, v_pred)
        v_f1 = f1_score(h, v_pred, zero_division=0)
        v_kappa = cohen_kappa_score(h, v_pred)
    else:
        acc = f1 = prec = rec = kappa = 0
        v_acc = v_f1 = v_kappa = 0
        cm = None
    
    results[dim_name] = {
        'human_pos': n_human_pos, 'pred_pos': n_pred_pos,
        'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'kappa': kappa,
        'cm': cm,
        'vision_only_acc': v_acc, 'vision_only_f1': v_f1, 'vision_only_kappa': v_kappa
    }

# Overall exploitative
h_overall = merged['annotator_overall_exploitative'].values

# Use the pipeline's overall_exploitative score directly
if 'overall_exploitative' in merged.columns:
    pipe_overall = pd.to_numeric(merged['overall_exploitative'], errors='coerce').fillna(0).values
else:
    pipe_overall = np.zeros(len(merged))

# Also compute from individual dims: 0.67*vision + 0.33*llm for each, then average
v_dims = []
l_dims = []
for dim_name, cols in dim_mapping.items():
    if cols['vision'] in merged.columns:
        v_dims.append(pd.to_numeric(merged[cols['vision']], errors='coerce').fillna(0).values)
    if cols['llm'] in merged.columns:
        l_dims.append(pd.to_numeric(merged[cols['llm']], errors='coerce').fillna(0).values)

if v_dims:
    v_overall = np.mean(v_dims, axis=0)
else:
    v_overall = np.zeros(len(merged))
if l_dims:
    l_overall = np.mean(l_dims, axis=0)
else:
    l_overall = np.zeros(len(merged))

print(f"\nPipeline overall_exploitative available: {'overall_exploitative' in merged.columns}")

# Use pipeline's overall score directly
pred_overall = (pipe_overall >= 0.5).astype(int)
combined_overall = pipe_overall
v_pred_overall = (v_overall >= 0.5).astype(int)

# Overall metrics
overall_acc = accuracy_score(h_overall, pred_overall)
overall_f1 = f1_score(h_overall, pred_overall, zero_division=0)
overall_prec = precision_score(h_overall, pred_overall, zero_division=0)
overall_rec = recall_score(h_overall, pred_overall, zero_division=0)
overall_kappa = cohen_kappa_score(h_overall, pred_overall)
overall_cm = confusion_matrix(h_overall, pred_overall, labels=[0,1])

v_overall_acc = accuracy_score(h_overall, v_pred_overall)
v_overall_f1 = f1_score(h_overall, v_pred_overall, zero_division=0)
v_overall_kappa = cohen_kappa_score(h_overall, v_pred_overall)

print("\n" + "="*80)
print("VALIDATION RESULTS: 50 Human Annotations vs Pipeline Predictions")
print("="*80)

print(f"\n{'='*80}")
print("OVERALL EXPLOITATIVE")
print(f"{'='*80}")
print(f"  Human: {int(h_overall.sum())} exploit / {int((h_overall==0).sum())} clean")
print(f"  Pipeline pred: {int(pred_overall.sum())} exploit / {int((pred_overall==0).sum())} clean")
print(f"\n  Combined (0.67*Vision + 0.33*LLM):")
print(f"    Accuracy:  {overall_acc:.3f}")
print(f"    F1:        {overall_f1:.3f}")
print(f"    Precision: {overall_prec:.3f}")
print(f"    Recall:    {overall_rec:.3f}")
print(f"    Cohen's κ: {overall_kappa:.3f}")
print(f"    Confusion Matrix:")
print(f"                  Pred=0  Pred=1")
print(f"      Human=0     {overall_cm[0][0]:5d}   {overall_cm[0][1]:5d}")
print(f"      Human=1     {overall_cm[1][0]:5d}   {overall_cm[1][1]:5d}")

print(f"\n  Vision-only:")
print(f"    Accuracy:  {v_overall_acc:.3f}")
print(f"    F1:        {v_overall_f1:.3f}")
print(f"    Cohen's κ: {v_overall_kappa:.3f}")

print(f"\n{'='*80}")
print("PER-DIMENSION RESULTS")
print(f"{'='*80}")
for dim_name, r in results.items():
    print(f"\n  {dim_name}:")
    print(f"    Human positive: {r['human_pos']}, Pipeline positive: {r['pred_pos']}")
    print(f"    Combined: Acc={r['accuracy']:.3f}, F1={r['f1']:.3f}, κ={r['kappa']:.3f}")
    print(f"    Vision-only: Acc={r['vision_only_acc']:.3f}, F1={r['vision_only_f1']:.3f}, κ={r['vision_only_kappa']:.3f}")
    if r['cm'] is not None:
        print(f"    CM: TN={r['cm'][0][0]}, FP={r['cm'][0][1]}, FN={r['cm'][1][0]}, TP={r['cm'][1][1]}")

# Detailed disagreement analysis
print(f"\n{'='*80}")
print("DISAGREEMENT ANALYSIS (Overall)")
print(f"{'='*80}")

for i, row in merged.iterrows():
    h = int(row['annotator_overall_exploitative'])
    p = int(pred_overall[i])
    if h != p:
        title = row['title'] if 'title' in row else row.get('title_vision', 'N/A')
        v_score = v_overall[i]
        l_score = l_overall[i]
        c_score = combined_overall[i]
        error_type = "FALSE POSITIVE" if p == 1 else "FALSE NEGATIVE"
        notes = row.get('annotator_notes', '')
        print(f"\n  [{error_type}] {title[:60]}")
        print(f"    Human={h}, Pred={p} | Vision={v_score:.2f}, LLM={l_score:.2f}, Combined={c_score:.2f}")
        if pd.notna(notes) and notes:
            print(f"    Notes: {notes}")

# Save results to JSON
output = {
    'n_annotations': len(human),
    'overall': {
        'accuracy': overall_acc, 'f1': overall_f1, 'precision': overall_prec,
        'recall': overall_rec, 'kappa': overall_kappa,
        'human_exploit': int(h_overall.sum()), 'human_clean': int((h_overall==0).sum()),
        'vision_only_kappa': v_overall_kappa, 'vision_only_f1': v_overall_f1
    },
    'per_dimension': {}
}
for dim_name, r in results.items():
    output['per_dimension'][dim_name] = {
        'accuracy': r['accuracy'], 'f1': r['f1'], 'precision': r['precision'],
        'recall': r['recall'], 'kappa': r['kappa'],
        'human_pos': r['human_pos'], 'pred_pos': r['pred_pos']
    }

with open('/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v4/validation_50.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to validation_50.json")
