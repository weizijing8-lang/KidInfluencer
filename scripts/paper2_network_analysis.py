"""
Paper 2: The Industrial Structure Behind Kidfluencers
=====================================================
RQ: Is the collaboration network among family channels driven by 
    organic relationships or by shared management infrastructure?
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats

BASE_DIR = '/home/ubuntu/KidInfluencer'
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
RESULTS_DIR = os.path.join(BASE_DIR, 'data/paper2_results')
FIG_DIR = os.path.join(BASE_DIR, 'figures_paper2')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("="*60)
print("PAPER 2: INDUSTRIAL STRUCTURE BEHIND KIDFLUENCERS")
print("="*60)

# ============================================================
# 1. MCN affiliations
# ============================================================
mcn_data = {
    'brentrivera': {'mcn': 'Amp Studios', 'controversy': 'LOW'},
    'piersonwodzynski': {'mcn': 'Amp Studios', 'controversy': 'LOW'},
    'rebeccazamolo': {'mcn': 'Underscore/CAA', 'controversy': 'LOW'},
    'jordanmatter': {'mcn': 'Underscore', 'controversy': 'MEDIUM'},
    'ryansworld': {'mcn': 'pocket.watch', 'controversy': 'MEDIUM'},
    'cocomelon': {'mcn': 'Moonbug', 'controversy': 'LOW'},
    'bratayley': {'mcn': 'Maker/Disney', 'controversy': 'HIGH'},
    'dailybumps': {'mcn': 'Maker/Disney', 'controversy': 'MEDIUM'},
    'familyfunpack': {'mcn': 'Studio71', 'controversy': 'MEDIUM'},
    'acefamily': {'mcn': 'MN2S', 'controversy': 'HIGH'},
    'piperrockelle': {'mcn': 'Independent', 'controversy': 'HIGH'},
    'labrantfam': {'mcn': 'CAA/Digital Dept', 'controversy': 'HIGH'},
    'vladandniki': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'familyfizz': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'ehbee': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'tannerites': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'thesacconejolys': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'kkandbabyj': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'bonniehoellein': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'theleray': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'theweisslife': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'everleighrose': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'itsyeboi': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'andrewdavila': {'mcn': 'Self-managed', 'controversy': 'LOW'},
    'thesuperherobuddy': {'mcn': 'Self-managed', 'controversy': 'LOW'},
}

# ============================================================
# 2. Load data - use short names (lowercase) consistently
# ============================================================
print("\n--- Loading data ---")

# Within-family profiles (has exploitation scores)
df_profiles = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/within_family_profiles.csv'))
print(f"Within-family profiles: {len(df_profiles)} channels")

# Full video data for collaboration detection
df_all = pd.read_csv(os.path.join(BASE_DIR, 'data/results_v4/full_results_v4.csv'))
df_all = df_all.rename(columns={'channel_short_name': 'channel', 'channel_category': 'category'})
df_fam = df_all[df_all['category'] == 'family'].copy()
print(f"Family videos: {len(df_fam)}")

# Get all channel short names
from channel_list import CHANNELS
family_short_names = set(name for name, handle, cat in CHANNELS if cat == 'family')
all_short_names = set(name for name, handle, cat in CHANNELS)
print(f"Family channels in list: {len(family_short_names)}")

# ============================================================
# 3. Detect collaborations from titles + descriptions
# ============================================================
print("\n--- Detecting collaborations ---")

collab_edges = defaultdict(int)

for _, row in df_fam.iterrows():
    ch = row['channel']
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    
    for other in all_short_names:
        if other == ch:
            continue
        if len(other) < 4:
            continue
        if other.lower() in text:
            pair = tuple(sorted([ch, other]))
            collab_edges[pair] += 1

# Filter to family-family edges only, with minimum 2 co-appearances
edges = []
for (a, b), count in collab_edges.items():
    if a in family_short_names and b in family_short_names and count >= 2:
        a_mcn = mcn_data.get(a, {}).get('mcn', 'Self-managed')
        b_mcn = mcn_data.get(b, {}).get('mcn', 'Self-managed')
        same_mcn = (a_mcn == b_mcn) and a_mcn not in ['Self-managed', 'Independent']
        edges.append({
            'source': a, 'target': b, 'weight': count,
            'same_mcn': same_mcn,
            'source_mcn': a_mcn, 'target_mcn': b_mcn,
        })

df_edges = pd.DataFrame(edges).sort_values('weight', ascending=False) if edges else pd.DataFrame()
print(f"Family-family edges (>=2): {len(df_edges)}")

if len(df_edges) > 0:
    print(f"Same-MCN edges: {df_edges['same_mcn'].sum()}")
    print(f"\nTop 15 collaboration pairs:")
    for _, row in df_edges.head(15).iterrows():
        tag = " [SAME MCN]" if row['same_mcn'] else ""
        print(f"  {row['source']:25s} ↔ {row['target']:25s}: {row['weight']:4d}{tag}")

df_edges.to_csv(os.path.join(RESULTS_DIR, 'family_collab_edges.csv'), index=False)

# ============================================================
# 4. Compute network metrics
# ============================================================
print("\n--- Computing network metrics ---")

degree = defaultdict(set)
weighted_degree = defaultdict(int)
for _, row in df_edges.iterrows():
    degree[row['source']].add(row['target'])
    degree[row['target']].add(row['source'])
    weighted_degree[row['source']] += row['weight']
    weighted_degree[row['target']] += row['weight']

# Build channel-level table
rows = []
for ch in sorted(family_short_names):
    mcn_info = mcn_data.get(ch, {})
    
    # Get exploitation score from profiles
    exploit = 0
    match = df_profiles[df_profiles['channel'] == ch]
    if len(match) > 0:
        exploit = match['content_exploit'].values[0]
    
    mcn = mcn_info.get('mcn', 'Self-managed')
    rows.append({
        'channel': ch,
        'mcn': mcn,
        'mcn_type': 'Professional' if mcn not in ['Self-managed', 'Independent'] else 'Self-managed',
        'controversy': mcn_info.get('controversy', 'LOW'),
        'degree': len(degree.get(ch, set())),
        'weighted_degree': weighted_degree.get(ch, 0),
        'exploit_score': exploit,
    })

df_net = pd.DataFrame(rows).sort_values('weighted_degree', ascending=False)
print("\nChannel network summary:")
print(df_net[['channel', 'mcn', 'degree', 'weighted_degree', 'exploit_score', 'controversy']].to_string(index=False))

df_net.to_csv(os.path.join(RESULTS_DIR, 'channel_network_mcn.csv'), index=False)

# ============================================================
# 5. Statistical tests
# ============================================================
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

# Test 1: Same-MCN pairs collaborate more?
if len(df_edges) > 0:
    same = df_edges[df_edges['same_mcn']]['weight']
    diff = df_edges[~df_edges['same_mcn']]['weight']
    print(f"\nSame-MCN pairs: n={len(same)}, mean={same.mean():.1f}")
    print(f"Diff-MCN pairs: n={len(diff)}, mean={diff.mean():.1f}")
    if len(same) > 0 and len(diff) > 0:
        u, p = stats.mannwhitneyu(same, diff, alternative='greater')
        print(f"Mann-Whitney (same > diff): U={u:.1f}, p={p:.4f}")

# Test 2: Professional MCN vs Self-managed
pro = df_net[df_net['mcn_type'] == 'Professional']
self_m = df_net[df_net['mcn_type'] == 'Self-managed']
print(f"\nProfessional MCN (n={len(pro)}):")
print(f"  Mean exploit: {pro['exploit_score'].mean():.4f}")
print(f"  Mean degree: {pro['degree'].mean():.2f}")
print(f"Self-managed (n={len(self_m)}):")
print(f"  Mean exploit: {self_m['exploit_score'].mean():.4f}")
print(f"  Mean degree: {self_m['degree'].mean():.2f}")

if len(pro) > 1 and len(self_m) > 1:
    u_e, p_e = stats.mannwhitneyu(pro['exploit_score'], self_m['exploit_score'], alternative='greater')
    u_d, p_d = stats.mannwhitneyu(pro['degree'], self_m['degree'], alternative='greater')
    print(f"Exploit: Professional > Self-managed: p={p_e:.4f}")
    print(f"Degree: Professional > Self-managed: p={p_d:.4f}")

# Test 3: Degree ↔ Exploitation correlation
connected = df_net[df_net['degree'] > 0]
if len(connected) > 2:
    r, p = stats.pearsonr(connected['degree'], connected['exploit_score'])
    print(f"\nDegree ↔ Exploitation (connected channels only): r={r:.4f}, p={p:.4f}")

# Test 4: Controversy ↔ Exploitation
cont_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
df_net['controversy_num'] = df_net['controversy'].map(cont_map)
r_c, p_c = stats.spearmanr(df_net['controversy_num'], df_net['exploit_score'])
print(f"Controversy ↔ Exploitation (Spearman): r={r_c:.4f}, p={p_c:.4f}")

# ============================================================
# 6. FIGURES
# ============================================================
print("\n--- Generating figures ---")
plt.style.use('seaborn-v0_8-whitegrid')

mcn_colors = {
    'Amp Studios': '#e74c3c',
    'Underscore/CAA': '#3498db',
    'Underscore': '#3498db',
    'pocket.watch': '#2ecc71',
    'Moonbug': '#9b59b6',
    'Maker/Disney': '#f39c12',
    'Studio71': '#1abc9c',
    'MN2S': '#e67e22',
    'CAA/Digital Dept': '#34495e',
    'Independent': '#95a5a6',
    'Self-managed': '#bdc3c7',
}

# --- Fig 1: Network graph ---
fig, ax = plt.subplots(figsize=(14, 12))

# Layout: place channels in a circle, high-degree closer to center
np.random.seed(42)
n = len(df_net)
df_sorted_net = df_net.sort_values('degree', ascending=False)

pos = {}
for i, (_, row) in enumerate(df_sorted_net.iterrows()):
    angle = 2 * np.pi * i / n
    radius = 4.0 - min(row['degree'], 8) * 0.35
    pos[row['channel']] = (radius * np.cos(angle), radius * np.sin(angle))

# Draw edges
for _, edge in df_edges.iterrows():
    if edge['source'] in pos and edge['target'] in pos:
        x = [pos[edge['source']][0], pos[edge['target']][0]]
        y = [pos[edge['source']][1], pos[edge['target']][1]]
        alpha = min(0.7, edge['weight'] / 50)
        width = min(5, edge['weight'] / 20)
        color = '#e74c3c' if edge['same_mcn'] else '#aaaaaa'
        ax.plot(x, y, color=color, alpha=alpha, linewidth=width, zorder=1)

# Draw nodes
for _, row in df_net.iterrows():
    if row['channel'] in pos:
        x, y = pos[row['channel']]
        color = mcn_colors.get(row['mcn'], '#bdc3c7')
        size = 150 + row['weighted_degree'] * 3
        ec = '#e74c3c' if row['controversy'] == 'HIGH' else '#f39c12' if row['controversy'] == 'MEDIUM' else '#2ecc71'
        ew = 3 if row['controversy'] == 'HIGH' else 2
        ax.scatter(x, y, s=size, c=color, edgecolors=ec, linewidths=ew, zorder=2)
        fs = 7 + min(row['degree'], 5)
        fw = 'bold' if row['degree'] > 2 else 'normal'
        ax.annotate(row['channel'], (x, y), textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=fs, fontweight=fw)

# Legends
patches = [mpatches.Patch(color=c, label=m) for m, c in mcn_colors.items() if m != 'Self-managed']
patches.append(mpatches.Patch(color='#bdc3c7', label='Self-managed'))
ax.legend(handles=patches, loc='upper left', title='MCN / Agency', fontsize=8, title_fontsize=9)
ax.annotate('Border: Red=HIGH controversy | Orange=MEDIUM | Green=LOW\nEdge: Red=Same MCN | Gray=Different MCN',
           xy=(0.02, 0.02), xycoords='axes fraction', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_title('Family Channel Collaboration Network\nColored by MCN/Agency Affiliation', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_network_mcn.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig1_network_mcn.png")

# --- Fig 2: Channel ranking with MCN + controversy ---
fig, ax = plt.subplots(figsize=(12, 10))
df_rank = df_net.sort_values('exploit_score', ascending=True)
y_pos = range(len(df_rank))
colors = [mcn_colors.get(row['mcn'], '#bdc3c7') for _, row in df_rank.iterrows()]
ec = ['#e74c3c' if row['controversy'] == 'HIGH' else '#f39c12' if row['controversy'] == 'MEDIUM' else '#2ecc71'
      for _, row in df_rank.iterrows()]

bars = ax.barh(y_pos, df_rank['exploit_score'], color=colors, edgecolor=ec, linewidth=2, height=0.7)

for i, (_, row) in enumerate(df_rank.iterrows()):
    ax.text(-0.003, i, row['channel'], ha='right', va='center', fontsize=8, fontweight='bold')
    mcn_label = row['mcn'] if row['mcn'] != 'Self-managed' else ''
    if mcn_label:
        ax.text(row['exploit_score'] + 0.002, i, f"[{mcn_label}]", ha='left', va='center', fontsize=7,
               fontstyle='italic', color='#555555')

ax.set_yticks([])
ax.set_xlabel('Mean Exploitation Score', fontsize=11)
ax.set_title('Family Channels: Exploitation Score with MCN Affiliation\n(Border color = Controversy level)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_ranking_mcn.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig2_ranking_mcn.png")

# --- Fig 3: Professional vs Self-managed comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_i, (col, label) in enumerate([
    ('exploit_score', 'Exploitation Score'),
    ('degree', 'Network Degree'),
    ('weighted_degree', 'Weighted Degree')
]):
    ax = axes[ax_i]
    data_pro = pro[col].values
    data_self = self_m[col].values
    
    bp = ax.boxplot([data_pro, data_self], labels=['Professional\nMCN', 'Self-\nmanaged'],
                   patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('#3498db')
    bp['boxes'][1].set_alpha(0.5)
    
    # Overlay points
    for i, d in enumerate([data_pro, data_self]):
        jitter = np.random.normal(0, 0.04, len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d, alpha=0.7, s=40, zorder=3,
                  color=['#e74c3c', '#3498db'][i])
    
    if len(data_pro) > 1 and len(data_self) > 1:
        u, p = stats.mannwhitneyu(data_pro, data_self, alternative='greater')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.set_title(f'{label}\np={p:.4f} {sig}', fontsize=10)
    ax.set_ylabel(label, fontsize=9)

plt.suptitle('Professional MCN vs Self-Managed Family Channels', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_mcn_vs_selfmanaged.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig3_mcn_vs_selfmanaged.png")

# --- Fig 4: Controversy vs Exploitation ---
fig, ax = plt.subplots(figsize=(8, 6))
for level, num in [('LOW', 1), ('MEDIUM', 2), ('HIGH', 3)]:
    sub = df_net[df_net['controversy'] == level]
    color = '#2ecc71' if level == 'LOW' else '#f39c12' if level == 'MEDIUM' else '#e74c3c'
    jitter = np.random.normal(0, 0.08, len(sub))
    ax.scatter(num + jitter, sub['exploit_score'], s=100, c=color, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.errorbar(num, sub['exploit_score'].mean(), yerr=sub['exploit_score'].sem(), 
               fmt='D', color='black', markersize=10, capsize=5, zorder=5)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['LOW\n(no major issues)', 'MEDIUM\n(FTC complaints,\ncriticism)', 'HIGH\n(lawsuits, abuse,\nlegal action)'])
ax.set_ylabel('Mean Exploitation Score', fontsize=11)
ax.set_title('Controversy Level vs. Exploitation Score\n(Within Family Channels)', fontsize=13, fontweight='bold')
r_s, p_s = stats.spearmanr(df_net['controversy_num'], df_net['exploit_score'])
ax.annotate(f'Spearman r = {r_s:.3f}, p = {p_s:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_controversy_vs_exploit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig4_controversy_vs_exploit.png")

print("\nDone! All results saved.")
