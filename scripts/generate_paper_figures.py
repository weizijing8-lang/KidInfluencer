"""
Generate publication-quality figures for AIES 2026 paper.
Combines clustering results + thumbnail CV analysis.
"""
import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = 'analysis_discovery/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
cluster_info = json.load(open('analysis_discovery/cluster_info.json'))
cluster_labels = json.load(open('analysis_discovery/cluster_labels_llm.json'))
thumb_cv = pd.read_csv('analysis_discovery/thumbnail_cv_results.csv')
videos_clusters = pd.read_csv('analysis_discovery/videos_with_clusters.csv')

label_map = {c['cluster_id']: c['category_name'] for c in cluster_labels}
boost_map = {c['cluster']: c['view_boost'] for c in cluster_info}
risk_map = {c['cluster_id']: c['manipulation_risk'] for c in cluster_labels}

# ============ FIGURE 1: View Boost by Cluster (with labels) ============
print("Generating Figure 1: View Boost by Cluster...")

sorted_clusters = sorted(cluster_info, key=lambda x: x['view_boost'], reverse=True)

fig, ax = plt.subplots(figsize=(12, 5))
categories = [label_map[c['cluster']] for c in sorted_clusters]
boosts = [c['view_boost'] * 100 for c in sorted_clusters]
risks = [risk_map[c['cluster']] for c in sorted_clusters]

# Color by manipulation risk
color_map = {'high': '#d32f2f', 'medium': '#f57c00', 'low': '#388e3c'}
colors = [color_map.get(r, '#999') for r in risks]

bars = ax.barh(range(len(categories)), boosts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories, fontsize=9)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('View Boost vs Dataset Median (%)', fontsize=11)
ax.set_title('Platform Reward by Content Pattern\n(Unsupervised Discovery via Sentence-BERT + K-Means)', fontsize=12)

# Add legend for risk levels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d32f2f', label='High Manipulation Risk'),
    Patch(facecolor='#f57c00', label='Medium Manipulation Risk'),
    Patch(facecolor='#388e3c', label='Low Manipulation Risk'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Truncate x-axis for readability (top 2 clusters are extreme outliers)
ax.set_xlim(-100, 1000)
# Add annotations for outliers
for i, (bar, boost) in enumerate(zip(bars, boosts)):
    if boost > 1000:
        ax.annotate(f'+{boost:,.0f}%', xy=(1000, i), fontsize=8, va='center', ha='left',
                   color='#d32f2f', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_view_boost_clusters.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig1_view_boost_clusters.pdf', bbox_inches='tight')
plt.close()

# ============ FIGURE 2: Thumbnail Visual Features by Cluster ============
print("Generating Figure 2: Thumbnail Features by Cluster...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Prepare cluster-level data
cluster_thumb_data = []
for k in range(15):
    cl = thumb_cv[thumb_cv['cluster'] == k]
    if len(cl) == 0:
        continue
    cluster_thumb_data.append({
        'cluster': k,
        'category': label_map[k],
        'face_pct': (cl['n_faces'] > 0).mean() * 100,
        'saturation': cl['mean_saturation'].mean(),
        'text_density': cl['text_density'].mean() * 100,
        'view_boost': boost_map[k] * 100,
        'risk': risk_map[k]
    })

ctd = pd.DataFrame(cluster_thumb_data)

# Panel A: Saturation vs View Boost
ax = axes[0]
colors_scatter = [color_map[r] for r in ctd['risk']]
ax.scatter(ctd['saturation'], ctd['view_boost'], c=colors_scatter, s=80, edgecolors='black', linewidth=0.5, zorder=5)
# Add labels for extreme points
for _, row in ctd.iterrows():
    if abs(row['view_boost']) > 100 or row['saturation'] > 115:
        ax.annotate(row['category'][:15], (row['saturation'], row['view_boost']),
                   fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Mean Thumbnail Saturation')
ax.set_ylabel('View Boost (%)')
ax.set_title('A) Color Saturation vs Platform Reward')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-100, 1000)

# Panel B: Face % vs View Boost
ax = axes[1]
ax.scatter(ctd['face_pct'], ctd['view_boost'], c=colors_scatter, s=80, edgecolors='black', linewidth=0.5, zorder=5)
for _, row in ctd.iterrows():
    if abs(row['view_boost']) > 100:
        ax.annotate(row['category'][:15], (row['face_pct'], row['view_boost']),
                   fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Face Detection Rate (%)')
ax.set_ylabel('View Boost (%)')
ax.set_title('B) Face Presence vs Platform Reward')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-100, 1000)

# Panel C: Text Density vs View Boost
ax = axes[2]
ax.scatter(ctd['text_density'], ctd['view_boost'], c=colors_scatter, s=80, edgecolors='black', linewidth=0.5, zorder=5)
for _, row in ctd.iterrows():
    if abs(row['view_boost']) > 100:
        ax.annotate(row['category'][:15], (row['text_density'], row['view_boost']),
                   fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Text/Edge Density (%)')
ax.set_ylabel('View Boost (%)')
ax.set_title('C) Visual Complexity vs Platform Reward')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-100, 1000)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_thumbnail_features.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig2_thumbnail_features.pdf', bbox_inches='tight')
plt.close()

# ============ FIGURE 3: Multimodal Heatmap ============
print("Generating Figure 3: Multimodal Feature Heatmap...")

# Create a heatmap of normalized features across clusters
features = ['face_pct', 'saturation', 'text_density', 'view_boost']
feature_labels = ['Face Detection\nRate (%)', 'Color\nSaturation', 'Text/Edge\nDensity (%)', 'View Boost\n(%)']

# Sort by view boost
ctd_sorted = ctd.sort_values('view_boost', ascending=False)

# Normalize each feature to 0-1 for heatmap
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = scaler.fit_transform(ctd_sorted[features])

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(feature_labels)))
ax.set_xticklabels(feature_labels, fontsize=9)
ax.set_yticks(range(len(ctd_sorted)))
ax.set_yticklabels([f"{row['category']}" for _, row in ctd_sorted.iterrows()], fontsize=9)

# Add text annotations
for i in range(len(ctd_sorted)):
    for j in range(len(features)):
        val = ctd_sorted.iloc[i][features[j]]
        if features[j] == 'view_boost':
            text = f'{val:+.0f}%'
        elif features[j] in ['face_pct', 'text_density']:
            text = f'{val:.1f}'
        else:
            text = f'{val:.0f}'
        color = 'white' if normalized[i, j] < 0.3 or normalized[i, j] > 0.7 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

ax.set_title('Multimodal Feature Heatmap by Content Cluster\n(Sorted by Platform Reward)', fontsize=11)
plt.colorbar(im, ax=ax, label='Normalized Value', shrink=0.8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_multimodal_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig3_multimodal_heatmap.pdf', bbox_inches='tight')
plt.close()

# ============ FIGURE 4: Content Volume vs Reward (Bubble Chart) ============
print("Generating Figure 4: Content Volume vs Reward...")

fig, ax = plt.subplots(figsize=(10, 7))

for _, row in ctd.iterrows():
    ci = [c for c in cluster_info if c['cluster'] == row['cluster']][0]
    size = ci['n_videos'] / 50  # Scale bubble size
    color = color_map[row['risk']]
    ax.scatter(ci['n_videos'], row['view_boost'], s=size, c=color, 
              alpha=0.7, edgecolors='black', linewidth=0.5)
    # Label
    if abs(row['view_boost']) > 50 or ci['n_videos'] > 4000:
        ax.annotate(row['category'][:20], (ci['n_videos'], row['view_boost']),
                   fontsize=8, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Number of Videos in Cluster', fontsize=11)
ax.set_ylabel('View Boost vs Median (%)', fontsize=11)
ax.set_title('Content Volume vs Platform Reward\n(Bubble size = cluster size)', fontsize=12)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-100, 800)
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_volume_vs_reward.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig4_volume_vs_reward.pdf', bbox_inches='tight')
plt.close()

# ============ FIGURE 5: Exploitation Risk Matrix ============
print("Generating Figure 5: Exploitation Risk Matrix...")

fig, ax = plt.subplots(figsize=(10, 7))

# X-axis: commercialization (saturation as proxy)
# Y-axis: view boost (platform reward)
# Color: manipulation risk
# Size: number of videos

for _, row in ctd.iterrows():
    ci = [c for c in cluster_info if c['cluster'] == row['cluster']][0]
    size = ci['n_videos'] / 30
    color = color_map[row['risk']]
    ax.scatter(row['saturation'], row['view_boost'], s=size, c=color,
              alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.annotate(row['category'][:18], (row['saturation'], row['view_boost']),
               fontsize=7, ha='center', va='bottom',
               xytext=(0, 5), textcoords='offset points')

ax.set_xlabel('Visual Commercialization (Thumbnail Saturation)', fontsize=11)
ax.set_ylabel('Platform Reward (View Boost %)', fontsize=11)
ax.set_title('Exploitation Risk Matrix: Commercialization × Platform Incentive', fontsize=12)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=ctd['saturation'].median(), color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-100, 800)

# Quadrant labels
ax.text(0.95, 0.95, 'HIGH RISK\n(High commercial +\nHigh reward)', transform=ax.transAxes,
       fontsize=9, ha='right', va='top', color='#d32f2f', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='#ffcdd2', alpha=0.5))
ax.text(0.05, 0.05, 'LOW RISK\n(Low commercial +\nLow reward)', transform=ax.transAxes,
       fontsize=9, ha='left', va='bottom', color='#388e3c', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='#c8e6c9', alpha=0.5))

ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_risk_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig5_risk_matrix.pdf', bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}/")
print("Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('fig'):
        print(f"  {f}")
