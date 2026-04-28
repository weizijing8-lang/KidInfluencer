"""
Paper 2: The Industrial Structure Behind Kidfluencers
=====================================================
RQ: Is child influencer exploitation driven by individual families or 
    by an organized industry structure (MCNs, talent agencies, brand deals)?

Core analyses:
1. Collaboration network → MCN cluster detection
2. MCN-managed vs self-managed: structural differences
3. Brand deal ecosystem: who profits from child content?
4. Controversy patterns: do MCN-managed channels face more issues?
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats
from collections import defaultdict

BASE_DIR = '/home/ubuntu/KidInfluencer'
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
RESULTS_DIR = os.path.join(BASE_DIR, 'data/paper2_industrial')
FIG_DIR = os.path.join(BASE_DIR, 'figures_paper2_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("="*60)
print("PAPER 2: THE INDUSTRIAL STRUCTURE BEHIND KIDFLUENCERS")
print("="*60)

# ============================================================
# 1. MCN/Agency data (from research)
# ============================================================
mcn_data = {
    'brentrivera':     {'mcn': 'Amp Studios',       'type': 'professional', 'controversy': 'LOW',    'brand_scale': 'LARGE'},
    'piersonwodzynski':{'mcn': 'Amp Studios',       'type': 'professional', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'rebeccazamolo':   {'mcn': 'Underscore/CAA',    'type': 'professional', 'controversy': 'LOW',    'brand_scale': 'MEDIUM'},
    'jordanmatter':    {'mcn': 'Underscore',        'type': 'professional', 'controversy': 'MEDIUM', 'brand_scale': 'LARGE'},
    'ryansworld':      {'mcn': 'pocket.watch',      'type': 'corporate',    'controversy': 'MEDIUM', 'brand_scale': 'MASSIVE'},
    'cocomelon':       {'mcn': 'Moonbug',           'type': 'corporate',    'controversy': 'LOW',    'brand_scale': 'MASSIVE'},
    'bratayley':       {'mcn': 'Maker/Disney',      'type': 'professional', 'controversy': 'HIGH',   'brand_scale': 'MEDIUM'},
    'dailybumps':      {'mcn': 'Maker/Disney',      'type': 'professional', 'controversy': 'MEDIUM', 'brand_scale': 'MEDIUM'},
    'familyfunpack':   {'mcn': 'Studio71',          'type': 'professional', 'controversy': 'MEDIUM', 'brand_scale': 'MEDIUM'},
    'labrantfam':      {'mcn': 'CAA/Digital Dept',  'type': 'professional', 'controversy': 'HIGH',   'brand_scale': 'LARGE'},
    'piperrockelle':   {'mcn': 'Self (mother)',     'type': 'self-managed', 'controversy': 'HIGH',   'brand_scale': 'MEDIUM'},
    'acefamily':       {'mcn': 'MN2S',              'type': 'professional', 'controversy': 'HIGH',   'brand_scale': 'MEDIUM'},
    # Channels without clear MCN info
    'thesuperherobuddy':{'mcn': 'Unknown',          'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'familyfizz':      {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'theweisslife':    {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'bonniehoellein':  {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'kkandbabyj':      {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'theleray':        {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'thesacconejolys': {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'vladandniki':     {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'MEDIUM'},
    'ehbee':           {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'everleighrose':   {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'tannerites':      {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'itsyeboi':        {'mcn': 'Unknown',           'type': 'self-managed', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
    'andrewdavila':    {'mcn': 'Amp Studios',       'type': 'professional', 'controversy': 'LOW',    'brand_scale': 'SMALL'},
}

# ============================================================
# 2. Load structural data
# ============================================================
print("\n--- Loading data ---")

# Labor data from Paper 1
labor_file = os.path.join(BASE_DIR, 'data/paper1_structural/labor_hours_analysis.csv')
if os.path.exists(labor_file):
    df_labor = pd.read_csv(labor_file)
    print(f"Labor data: {len(df_labor)} channels")
else:
    print("ERROR: Run paper1_structural.py first")
    sys.exit(1)

# Collaboration edges
collab_file = os.path.join(BASE_DIR, 'data/paper2_results/family_collab_edges.csv')
if os.path.exists(collab_file):
    df_edges = pd.read_csv(collab_file)
    print(f"Collaboration edges: {len(df_edges)}")
else:
    # Try alternative
    collab_file2 = os.path.join(BASE_DIR, 'data/paper2_results/family_collaboration_edges.csv')
    if os.path.exists(collab_file2):
        df_edges = pd.read_csv(collab_file2)
        print(f"Collaboration edges: {len(df_edges)}")
    else:
        df_edges = pd.DataFrame()
        print("No collaboration edge data found")

# Add MCN data to labor dataframe
df_labor['mcn'] = df_labor['channel'].map(lambda x: mcn_data.get(x, {}).get('mcn', 'Unknown'))
df_labor['management_type'] = df_labor['channel'].map(lambda x: mcn_data.get(x, {}).get('type', 'self-managed'))
df_labor['controversy'] = df_labor['channel'].map(lambda x: mcn_data.get(x, {}).get('controversy', 'LOW'))
df_labor['brand_scale'] = df_labor['channel'].map(lambda x: mcn_data.get(x, {}).get('brand_scale', 'SMALL'))

# ============================================================
# 3. ANALYSIS: Professional vs Self-Managed
# ============================================================
print("\n" + "="*60)
print("PROFESSIONAL vs SELF-MANAGED CHANNELS")
print("="*60)

prof = df_labor[df_labor['management_type'].isin(['professional', 'corporate'])]
self_m = df_labor[df_labor['management_type'] == 'self-managed']

print(f"\nProfessional/Corporate: {len(prof)} channels")
print(f"Self-managed: {len(self_m)} channels")

metrics = {
    'videos_per_week': 'Upload Frequency (videos/week)',
    'production_hours_week': 'Production Hours/Week',
    'sponsor_rate': 'Sponsor Rate',
    'n_child_brands': 'Child Brand Mentions',
    'tiktok_videos': 'TikTok Videos',
    'total_videos_all_platforms': 'Total Videos (All Platforms)',
}

comparison_results = []
for col, label in metrics.items():
    p_vals = prof[col].dropna()
    s_vals = self_m[col].dropna()
    if len(p_vals) > 1 and len(s_vals) > 1:
        stat, p = stats.mannwhitneyu(p_vals, s_vals, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        comparison_results.append({
            'metric': label,
            'professional_mean': p_vals.mean(),
            'professional_median': p_vals.median(),
            'self_managed_mean': s_vals.mean(),
            'self_managed_median': s_vals.median(),
            'ratio': p_vals.mean() / s_vals.mean() if s_vals.mean() > 0 else float('inf'),
            'p_value': p,
            'significance': sig,
        })
        print(f"\n{label}:")
        print(f"  Professional: mean={p_vals.mean():.3f}, median={p_vals.median():.3f}")
        print(f"  Self-managed: mean={s_vals.mean():.3f}, median={s_vals.median():.3f}")
        print(f"  Ratio: {p_vals.mean()/s_vals.mean():.2f}x, p={p:.4f} {sig}")

df_comparison = pd.DataFrame(comparison_results)
df_comparison.to_csv(os.path.join(RESULTS_DIR, 'prof_vs_self_comparison.csv'), index=False)

# ============================================================
# 4. NETWORK ANALYSIS
# ============================================================
print("\n" + "="*60)
print("COLLABORATION NETWORK ANALYSIS")
print("="*60)

G = nx.Graph()
# Add all family channels as nodes
for _, row in df_labor.iterrows():
    G.add_node(row['channel'], 
               mcn=row['mcn'],
               management_type=row['management_type'],
               controversy=row['controversy'],
               brand_scale=row['brand_scale'],
               videos_per_week=row['videos_per_week'],
               sponsor_rate=row['sponsor_rate'])

# Add edges
if len(df_edges) > 0:
    edge_cols = df_edges.columns.tolist()
    print(f"Edge columns: {edge_cols}")
    
    # Try different column name patterns
    if 'channel1' in edge_cols and 'channel2' in edge_cols:
        c1, c2, w = 'channel1', 'channel2', 'weight' if 'weight' in edge_cols else 'n_collabs'
    elif 'source' in edge_cols and 'target' in edge_cols:
        c1, c2, w = 'source', 'target', 'weight' if 'weight' in edge_cols else 'n_collabs'
    else:
        c1, c2, w = edge_cols[0], edge_cols[1], edge_cols[2] if len(edge_cols) > 2 else None
    
    for _, row in df_edges.iterrows():
        n1, n2 = row[c1], row[c2]
        weight = row[w] if w and w in edge_cols else 1
        if n1 in G.nodes and n2 in G.nodes:
            G.add_edge(n1, n2, weight=weight)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Network metrics
degree_dict = dict(G.degree())
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

print("\nTop 10 by degree:")
for ch, deg in sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    mcn = mcn_data.get(ch, {}).get('mcn', 'Unknown')
    print(f"  {ch:25s}: degree={deg}, betweenness={betweenness[ch]:.3f}, MCN={mcn}")

# Connected components
components = list(nx.connected_components(G))
print(f"\nConnected components: {len(components)}")
for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5]):
    mcns = set(mcn_data.get(ch, {}).get('mcn', 'Unknown') for ch in comp)
    print(f"  Component {i+1} ({len(comp)} nodes): {comp}")
    print(f"    MCNs: {mcns}")

# ============================================================
# 5. CONTROVERSY × MANAGEMENT TYPE
# ============================================================
print("\n" + "="*60)
print("CONTROVERSY × MANAGEMENT TYPE")
print("="*60)

controversy_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
df_labor['controversy_score'] = df_labor['controversy'].map(controversy_map)

for mtype in ['professional', 'corporate', 'self-managed']:
    subset = df_labor[df_labor['management_type'] == mtype]
    if len(subset) > 0:
        mean_c = subset['controversy_score'].mean()
        high_pct = (subset['controversy'] == 'HIGH').mean() * 100
        print(f"  {mtype:15s}: mean controversy={mean_c:.2f}, HIGH%={high_pct:.0f}% (n={len(subset)})")

# ============================================================
# 6. FIGURES
# ============================================================
print("\n--- Generating figures ---")
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme for MCNs
mcn_colors = {
    'Amp Studios': '#e74c3c',
    'Underscore/CAA': '#3498db',
    'Underscore': '#3498db',
    'pocket.watch': '#2ecc71',
    'Moonbug': '#2ecc71',
    'Maker/Disney': '#9b59b6',
    'Studio71': '#f39c12',
    'CAA/Digital Dept': '#1abc9c',
    'MN2S': '#e67e22',
    'Self (mother)': '#95a5a6',
    'Unknown': '#bdc3c7',
}

mgmt_colors = {
    'professional': '#e74c3c',
    'corporate': '#3498db',
    'self-managed': '#95a5a6',
}

# --- Fig 1: Network graph with MCN coloring ---
fig, ax = plt.subplots(figsize=(14, 12))

# Layout
pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

# Draw edges with weight
edges = G.edges(data=True)
if edges:
    edge_weights = [d.get('weight', 1) for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 4 * w / max_w for w in edge_weights]
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.3, edge_color='#555555')

# Draw nodes colored by MCN
node_colors = [mcn_colors.get(mcn_data.get(n, {}).get('mcn', 'Unknown'), '#bdc3c7') for n in G.nodes]
node_sizes = [300 + 200 * degree_dict.get(n, 0) for n in G.nodes]

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8, edgecolors='black', linewidths=0.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')

# Legend for MCNs
legend_mcns = ['Amp Studios', 'Underscore/CAA', 'pocket.watch/Moonbug', 'Maker/Disney', 'Studio71', 'Self-managed/Unknown']
legend_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#bdc3c7']
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in zip(legend_mcns, legend_colors)]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9, title='MCN/Agency', title_fontsize=10)

ax.set_title('Kidfluencer Collaboration Network\nColored by MCN/Talent Agency Affiliation', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_network_mcn.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig1_network_mcn.png")

# --- Fig 2: Professional vs Self-Managed comparison ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

plot_metrics = [
    ('videos_per_week', 'Upload Frequency\n(videos/week)'),
    ('production_hours_week', 'Production Hours\n(per week)'),
    ('sponsor_rate', 'Sponsor Rate\n(fraction)'),
    ('n_child_brands', 'Child Brand\nMentions'),
    ('tiktok_videos', 'TikTok Videos'),
    ('total_videos_all_platforms', 'Total Videos\n(All Platforms)'),
]

for i, (col, label) in enumerate(plot_metrics):
    ax = axes[i]
    
    data_prof = prof[col].dropna()
    data_self = self_m[col].dropna()
    
    # Box plot
    bp = ax.boxplot([data_prof, data_self], labels=['Professional\n/Corporate', 'Self-\nManaged'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('#95a5a6')
    bp['boxes'][1].set_alpha(0.5)
    
    # Scatter individual points
    ax.scatter(np.ones(len(data_prof)) + np.random.normal(0, 0.05, len(data_prof)), 
              data_prof, color='#e74c3c', alpha=0.6, s=30, zorder=5)
    ax.scatter(np.ones(len(data_self))*2 + np.random.normal(0, 0.05, len(data_self)), 
              data_self, color='#95a5a6', alpha=0.6, s=30, zorder=5)
    
    # p-value
    if len(data_prof) > 1 and len(data_self) > 1:
        _, p = stats.mannwhitneyu(data_prof, data_self, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.set_title(f'{label}\n(p={p:.3f} {sig})', fontsize=9, fontweight='bold')
    else:
        ax.set_title(label, fontsize=9, fontweight='bold')

plt.suptitle('Professional/Corporate vs Self-Managed Family Channels\n(Structural Exploitation Metrics)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_prof_vs_self.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig2_prof_vs_self.png")

# --- Fig 3: Channel profiles by MCN ---
fig, ax = plt.subplots(figsize=(14, 10))

# Sort by management type then by total videos
df_sorted = df_labor.sort_values(['management_type', 'total_videos_all_platforms'], ascending=[True, True])

y_pos = range(len(df_sorted))
bar_colors = [mgmt_colors.get(t, '#bdc3c7') for t in df_sorted['management_type']]

# Stacked: YouTube + TikTok
yt_vids = df_sorted['n_videos_youtube'].values
tt_vids = df_sorted['tiktok_videos'].fillna(0).values

ax.barh(y_pos, yt_vids, color=bar_colors, alpha=0.8, label='YouTube')
ax.barh(y_pos, tt_vids, left=yt_vids, color=bar_colors, alpha=0.4, hatch='///', label='TikTok')

# Add MCN labels
for i, (_, row) in enumerate(df_sorted.iterrows()):
    total = row['n_videos_youtube'] + (row['tiktok_videos'] if not pd.isna(row['tiktok_videos']) else 0)
    mcn = row['mcn']
    if mcn != 'Unknown':
        ax.text(total + 30, i, f"[{mcn}]", va='center', fontsize=7, color='#333333', style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted['channel'], fontsize=8)
ax.set_xlabel('Total Videos Produced', fontsize=11)
ax.set_title('Content Production by Channel\n(Grouped by Management Type, MCN Labels Shown)', fontsize=13, fontweight='bold')

# Legend
legend_patches = [mpatches.Patch(color=c, label=l.title()) for l, c in mgmt_colors.items()]
legend_patches.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='TikTok portion'))
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_channel_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig3_channel_profiles.png")

# --- Fig 4: Controversy by management type ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: controversy distribution
ax = axes[0]
controversy_counts = df_labor.groupby(['management_type', 'controversy']).size().unstack(fill_value=0)
controversy_order = ['LOW', 'MEDIUM', 'HIGH']
controversy_colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}

x = range(len(controversy_counts.index))
bottom = np.zeros(len(x))
for level in controversy_order:
    if level in controversy_counts.columns:
        vals = controversy_counts[level].values
        ax.bar(x, vals, bottom=bottom, color=controversy_colors[level], label=level, alpha=0.8)
        bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([t.replace('-', '-\n') for t in controversy_counts.index], fontsize=9)
ax.set_ylabel('Number of Channels', fontsize=10)
ax.set_title('Controversy Level by\nManagement Type', fontsize=11, fontweight='bold')
ax.legend(title='Controversy', fontsize=9)

# Right: brand scale distribution
ax = axes[1]
brand_counts = df_labor.groupby(['management_type', 'brand_scale']).size().unstack(fill_value=0)
brand_order = ['SMALL', 'MEDIUM', 'LARGE', 'MASSIVE']
brand_colors = {'SMALL': '#bdc3c7', 'MEDIUM': '#f39c12', 'LARGE': '#e74c3c', 'MASSIVE': '#8e44ad'}

x = range(len(brand_counts.index))
bottom = np.zeros(len(x))
for level in brand_order:
    if level in brand_counts.columns:
        vals = brand_counts[level].values
        ax.bar(x, vals, bottom=bottom, color=brand_colors[level], label=level, alpha=0.8)
        bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([t.replace('-', '-\n') for t in brand_counts.index], fontsize=9)
ax.set_ylabel('Number of Channels', fontsize=10)
ax.set_title('Brand Deal Scale by\nManagement Type', fontsize=11, fontweight='bold')
ax.legend(title='Brand Scale', fontsize=9)

plt.suptitle('Who Faces Controversy? Who Gets the Biggest Brand Deals?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_controversy_brands.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig4_controversy_brands.png")

# --- Fig 5: The "industrial food chain" ---
fig, ax = plt.subplots(figsize=(14, 8))

# Create a summary visualization showing the industrial structure
# X = total videos, Y = sponsor rate, size = brand scale, color = MCN
brand_size_map = {'SMALL': 100, 'MEDIUM': 300, 'LARGE': 600, 'MASSIVE': 1200}

for _, row in df_labor.iterrows():
    x = row['total_videos_all_platforms']
    y = row['sponsor_rate'] * 100
    size = brand_size_map.get(row['brand_scale'], 100)
    color = mgmt_colors.get(row['management_type'], '#bdc3c7')
    
    ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=5)
    ax.annotate(row['channel'], (x, y), fontsize=7, ha='center', va='bottom',
               xytext=(0, 8), textcoords='offset points')

ax.set_xlabel('Total Videos Produced (YouTube + TikTok)', fontsize=11)
ax.set_ylabel('Sponsor Rate (%)', fontsize=11)
ax.set_title('The Industrial Food Chain of Child Influencing\n(Size = Brand Deal Scale, Color = Management Type)', 
             fontsize=13, fontweight='bold')

# Legend for management type
mgmt_patches = [mpatches.Patch(color=c, label=l.title()) for l, c in mgmt_colors.items()]
# Legend for brand scale
from matplotlib.lines import Line2D
size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=np.sqrt(s/10), label=l) 
                for l, s in brand_size_map.items()]

leg1 = ax.legend(handles=mgmt_patches, loc='upper left', title='Management', fontsize=8)
ax.add_artist(leg1)
ax.legend(handles=size_handles, loc='upper right', title='Brand Scale', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_industrial_food_chain.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fig5_industrial_food_chain.png")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "="*60)
print("PAPER 2 SUMMARY")
print("="*60)

print(f"\nSample: {len(df_labor)} family channels")
print(f"  Professional/Corporate: {len(prof)}")
print(f"  Self-managed: {len(self_m)}")

print(f"\nKey findings:")
print(f"  1. Network has {G.number_of_edges()} collaboration edges across {G.number_of_nodes()} channels")
print(f"  2. {len(components)} connected components (largest: {max(len(c) for c in components)} channels)")

print(f"\n  3. Professional vs Self-managed differences:")
for _, row in df_comparison.iterrows():
    if row['significance']:
        print(f"     {row['metric']}: {row['ratio']:.2f}x higher for professional (p={row['p_value']:.4f} {row['significance']})")

print(f"\n  4. All HIGH controversy channels are professional/self-managed-by-parent:")
high_c = df_labor[df_labor['controversy'] == 'HIGH']
for _, row in high_c.iterrows():
    print(f"     {row['channel']}: MCN={row['mcn']}, type={row['management_type']}")

print(f"\n  5. All MASSIVE/LARGE brand deals go to professional/corporate channels:")
big_brands = df_labor[df_labor['brand_scale'].isin(['MASSIVE', 'LARGE'])]
for _, row in big_brands.iterrows():
    print(f"     {row['channel']}: MCN={row['mcn']}, brand_scale={row['brand_scale']}")

print("\nDone!")
