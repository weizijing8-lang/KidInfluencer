"""
Visualize content drift time series for all channels.
Creates publication-quality plots showing:
1. Rolling drift score over time for each channel
2. Comparison bar chart of mean drift scores
3. Top exploitative titles per channel
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

import os

RESULTS_DIR = "/home/ubuntu/pilot/results"
OUTPUT_DIR = "/home/ubuntu/pilot/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

channels = {
    'acefamily': ('The ACE Family', '#e74c3c', 'Family'),
    'ryansworld': ("Ryan's World", '#3498db', 'Family'),
    'familyfunpack': ('Family Fun Pack', '#2ecc71', 'Family'),
    'bratayley': ('Bratayley', '#9b59b6', 'Family'),
    'caseyneistat': ('Casey Neistat', '#95a5a6', 'Control'),
    'markwiens': ('Mark Wiens', '#7f8c8d', 'Control'),
}

# ============================================================
# Figure 1: Rolling Drift Score Over Time (All Channels)
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=False, sharey=True)
axes = axes.flatten()

for idx, (key, (label, color, ch_type)) in enumerate(channels.items()):
    ax = axes[idx]
    filepath = os.path.join(RESULTS_DIR, f"{key}_drift.csv")
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    
    # Plot individual video drift scores (light)
    ax.scatter(df['video_order'], df['drift_score'], alpha=0.1, s=3, color=color)
    
    # Plot rolling average (bold)
    ax.plot(df['video_order'], df['drift_rolling'], color=color, linewidth=2, label='Rolling avg (w=20)')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Mark top 5 most "exploitative" titles
    top5 = df.nlargest(5, 'drift_score')
    for _, row in top5.iterrows():
        ax.annotate('', xy=(row['video_order'], row['drift_score']),
                    xytext=(row['video_order'], row['drift_score'] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    type_tag = "[FAMILY]" if ch_type == "Family" else "[CONTROL]"
    ax.set_title(f"{type_tag} {label} (n={len(df)})", fontsize=12, fontweight='bold')
    ax.set_xlabel('Video Order (oldest → newest)')
    ax.set_ylabel('Exploitation Drift Score')
    ax.set_ylim(-0.4, 0.5)

plt.suptitle('Content Exploitation Drift Over Time\n(Higher = More Exploitative Title Language)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'drift_timeseries_all.png'), bbox_inches='tight', dpi=150)
print("Saved: drift_timeseries_all.png")
plt.close()

# ============================================================
# Figure 2: Mean Drift Score Comparison (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

labels_list = []
means = []
stds = []
colors = []

for key, (label, color, ch_type) in channels.items():
    filepath = os.path.join(RESULTS_DIR, f"{key}_drift.csv")
    if not os.path.exists(filepath):
        continue
    df = pd.read_csv(filepath)
    type_tag = "[F]" if ch_type == "Family" else "[C]"
    labels_list.append(f"{type_tag} {label}")
    means.append(df['drift_score'].mean())
    stds.append(df['drift_score'].std())
    colors.append(color)

# Sort by mean
sorted_idx = np.argsort(means)[::-1]
labels_sorted = [labels_list[i] for i in sorted_idx]
means_sorted = [means[i] for i in sorted_idx]
stds_sorted = [stds[i] for i in sorted_idx]
colors_sorted = [colors[i] for i in sorted_idx]

bars = ax.barh(range(len(labels_sorted)), means_sorted, xerr=stds_sorted,
               color=colors_sorted, alpha=0.8, capsize=5)
ax.set_yticks(range(len(labels_sorted)))
ax.set_yticklabels(labels_sorted, fontsize=11)
ax.set_xlabel('Mean Exploitation Drift Score', fontsize=12)
ax.set_title('Mean Exploitation Drift Score by Channel\n[F] = Family Channel, [C] = Control (Adult-Only)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)

# Add value labels
for i, (m, s) in enumerate(zip(means_sorted, stds_sorted)):
    ax.text(m + s + 0.01, i, f'{m:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'drift_comparison_bar.png'), bbox_inches='tight', dpi=150)
print("Saved: drift_comparison_bar.png")
plt.close()

# ============================================================
# Figure 3: Top Exploitative Titles per Channel
# ============================================================
print("\n" + "="*80)
print("TOP 5 MOST 'EXPLOITATIVE' TITLES PER CHANNEL")
print("="*80)

for key, (label, color, ch_type) in channels.items():
    filepath = os.path.join(RESULTS_DIR, f"{key}_drift.csv")
    if not os.path.exists(filepath):
        continue
    df = pd.read_csv(filepath)
    top5 = df.nlargest(5, 'drift_score')[['title', 'drift_score', 'view_count']]
    print(f"\n--- {label} ({ch_type}) ---")
    for _, row in top5.iterrows():
        views = f"{row['view_count']:,.0f}" if row['view_count'] > 0 else "N/A"
        print(f"  [{row['drift_score']:.3f}] {row['title'][:80]} (views: {views})")

# ============================================================
# Figure 4: ACE Family Deep Dive - Drift Over Time with Annotations
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

df_ace = pd.read_csv(os.path.join(RESULTS_DIR, 'acefamily_drift.csv'))

ax.scatter(df_ace['video_order'], df_ace['drift_score'], alpha=0.15, s=8, color='#e74c3c')
ax.plot(df_ace['video_order'], df_ace['drift_rolling'], color='#e74c3c', linewidth=2.5)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Annotate top 3 most exploitative videos
top3 = df_ace.nlargest(3, 'drift_score')
for _, row in top3.iterrows():
    title_short = row['title'][:45] + "..." if len(str(row['title'])) > 45 else row['title']
    ax.annotate(title_short, 
                xy=(row['video_order'], row['drift_score']),
                xytext=(row['video_order'] + 20, row['drift_score'] + 0.05),
                fontsize=8, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

ax.set_xlabel('Video Order (oldest → newest)', fontsize=12)
ax.set_ylabel('Exploitation Drift Score', fontsize=12)
ax.set_title('The ACE Family: Content Exploitation Drift Over Time\n(Rolling Average, Window=20)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(-0.3, 0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'acefamily_deep_dive.png'), bbox_inches='tight', dpi=150)
print("\nSaved: acefamily_deep_dive.png")
plt.close()

print("\nAll figures saved to /home/ubuntu/pilot/figures/")
