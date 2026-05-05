#!/usr/bin/env python3
"""Generate publication-quality figures for AIES 2026 paper.
Uses Snorkel pipeline v2 results (6-dim + weak supervision)."""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = '/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v2'
FIGURES_DIR = '/home/ubuntu/KidInfluencer/analysis_discovery/paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
print("Loading results...")
scored = pd.read_csv(f'{RESULTS_DIR}/videos_with_exploitation_scores.csv')
ch_df = pd.read_csv(f'{RESULTS_DIR}/channel_exploitation_analysis.csv')
dim_boost = pd.read_csv(f'{RESULTS_DIR}/within_channel_boost_by_dimension.csv')
lf_weights = pd.read_csv(f'{RESULTS_DIR}/lf_weights.csv')
lf_analysis = pd.read_csv(f'{RESULTS_DIR}/lf_analysis.csv', index_col=0)

with open(f'{RESULTS_DIR}/summary_statistics.json') as f:
    summary = json.load(f)

valid = scored[scored['viewCount'].notna() & (scored['viewCount'] > 0)].copy()
valid['log_views'] = np.log10(valid['viewCount'])

dimensions = ['performative', 'emotional_bait', 'narrative_conflict', 
              'challenge_format', 'commercial_content', 'privacy_violation']

dim_labels = {
    'performative': 'Performative\nLabor',
    'emotional_bait': 'Emotional\nBait',
    'narrative_conflict': 'Narrative\nConflict',
    'challenge_format': 'Challenge\nFormat',
    'commercial_content': 'Commercial\nContent',
    'privacy_violation': 'Privacy\nViolation',
}

print(f"Dataset: {len(scored)} videos, {scored['channel_short_name'].nunique()} channels")

# ============================================================
# FIGURE 1: Pipeline Overview (Dimension Prevalence + Score Distribution)
# ============================================================
print("\nGenerating Figure 1: Dimension prevalence and score distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Dimension prevalence
ax = axes[0]
prev = scored[dimensions].mean().sort_values(ascending=True) * 100
colors_prev = sns.color_palette("Blues_d", len(prev))
bars = ax.barh(range(len(prev)), prev.values, color=colors_prev, edgecolor='white', height=0.65)
ax.set_yticks(range(len(prev)))
ax.set_yticklabels([dim_labels.get(d, d) for d in prev.index], fontsize=10)
ax.set_xlabel('Prevalence in Sample (%)')
ax.set_title('(a) Exploitation Dimension Prevalence')
for i, val in enumerate(prev.values):
    ax.text(val + 0.8, i, f'{val:.1f}%', va='center', fontsize=9)
ax.set_xlim(0, max(prev.values) * 1.2)

# Panel B: Exploitation score distribution
ax = axes[1]
ax.hist(scored['exploitation_score'], bins=40, color='steelblue', edgecolor='white', alpha=0.85, density=True)
ax.axvline(scored['exploitation_score'].median(), color='#e74c3c', linestyle='--', linewidth=2,
           label=f"Median = {scored['exploitation_score'].median():.3f}")
ax.axvline(scored['exploitation_score'].mean(), color='#f39c12', linestyle=':', linewidth=2,
           label=f"Mean = {scored['exploitation_score'].mean():.3f}")
ax.set_xlabel('Exploitation Score P(exploit)')
ax.set_ylabel('Density')
ax.set_title('(b) Snorkel Label Model Score Distribution')
ax.legend(frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_prevalence_and_scores.png')
plt.savefig(f'{FIGURES_DIR}/fig1_prevalence_and_scores.pdf')
plt.close()
print("  Saved fig1")

# ============================================================
# FIGURE 2: Within-Channel View Boost by Dimension (KEY FINDING)
# ============================================================
print("Generating Figure 2: Within-channel view boost...")

fig, ax = plt.subplots(figsize=(8, 5))
dim_boost_sorted = dim_boost.sort_values('mean_within_boost', ascending=True)

colors_boost = []
for _, row in dim_boost_sorted.iterrows():
    if row['p_value'] < 0.001:
        colors_boost.append('#27ae60')
    elif row['p_value'] < 0.01:
        colors_boost.append('#2ecc71')
    elif row['p_value'] < 0.05:
        colors_boost.append('#82e0aa')
    else:
        colors_boost.append('#bdc3c7')

bars = ax.barh(range(len(dim_boost_sorted)), dim_boost_sorted['mean_within_boost'], 
               color=colors_boost, edgecolor='white', height=0.6)
ax.set_yticks(range(len(dim_boost_sorted)))
ax.set_yticklabels([dim_labels.get(d, d) for d in dim_boost_sorted['dimension']], fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean Within-Channel View Boost (%)')
ax.set_title('Algorithmic Reward for Exploitation Dimensions\n(Within-Channel Median View Comparison)')

for i, (_, row) in enumerate(dim_boost_sorted.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
    x_pos = row['mean_within_boost']
    offset = 5 if x_pos >= 0 else -5
    ha = 'left' if x_pos >= 0 else 'right'
    ax.text(x_pos + offset, i, f"p={row['p_value']:.3f} {sig}", va='center', fontsize=9, ha=ha)

ax.text(0.02, 0.98, 'Green = statistically significant (p<0.05)\nGray = not significant',
        transform=ax.transAxes, fontsize=8, va='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_within_channel_boost.png')
plt.savefig(f'{FIGURES_DIR}/fig2_within_channel_boost.pdf')
plt.close()
print("  Saved fig2")

# ============================================================
# FIGURE 3: Labeling Function Coverage and Weights
# ============================================================
print("Generating Figure 3: LF analysis...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

cat_colors = {'LLM': '#3498db', 'Rule': '#e74c3c', 'CV/Vision': '#27ae60'}

# Panel A: Coverage
ax = axes[0]
lf_names = lf_analysis.index.tolist()
categories = ['LLM']*6 + ['Rule']*9 + ['CV/Vision']*3
coverage = lf_analysis['Coverage'].values
colors_lf = [cat_colors[c] for c in categories]

y_pos = range(len(lf_names))
ax.barh(y_pos, coverage, color=colors_lf, alpha=0.8, edgecolor='white', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace('lf_', '').replace('_', ' ') for n in lf_names], fontsize=8)
ax.set_xlabel('Coverage (fraction of dataset)')
ax.set_title('(a) Labeling Function Coverage')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in ['LLM', 'Rule', 'CV/Vision']]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

# Panel B: Learned weights
ax = axes[1]
weight_sorted = lf_weights.sort_values('weight', ascending=True)
w_colors = [cat_colors[c] for c in weight_sorted['category']]
ax.barh(range(len(weight_sorted)), weight_sorted['weight'], color=w_colors, alpha=0.8, edgecolor='white', height=0.7)
ax.set_yticks(range(len(weight_sorted)))
ax.set_yticklabels([n.replace('lf_', '').replace('_', ' ') for n in weight_sorted['lf_name']], fontsize=8)
ax.set_xlabel('Learned Weight (Label Model)')
ax.set_title('(b) Label Model Learned Weights')
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig3_lf_analysis.png')
plt.savefig(f'{FIGURES_DIR}/fig3_lf_analysis.pdf')
plt.close()
print("  Saved fig3")

# ============================================================
# FIGURE 4: Exploitation Score vs Views (scatter + regression)
# ============================================================
print("Generating Figure 4: Score vs views...")

fig, ax = plt.subplots(figsize=(8, 6))

# Bin by exploitation score deciles
valid['score_bin'] = pd.qcut(valid['exploitation_score'], q=10, labels=False, duplicates='drop')
bin_means = valid.groupby('score_bin').agg(
    mean_score=('exploitation_score', 'mean'),
    mean_log_views=('log_views', 'mean'),
    se_log_views=('log_views', 'sem'),
    n=('log_views', 'count')
).reset_index()

# Background scatter
ax.scatter(valid['exploitation_score'], valid['log_views'], alpha=0.08, s=8, c='gray')

# Binned means with error bars
ax.errorbar(bin_means['mean_score'], bin_means['mean_log_views'],
           yerr=bin_means['se_log_views'] * 1.96,
           fmt='o-', color='#e74c3c', markersize=8, linewidth=2, capsize=4,
           label='Decile means (95% CI)')

# Regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(valid['exploitation_score'], valid['log_views'])
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, slope * x_line + intercept, '--', color='#2c3e50', linewidth=1.5, alpha=0.7,
        label=f'OLS: slope={slope:.2f}')

corr, pval = stats.spearmanr(valid['exploitation_score'], valid['log_views'])
ax.set_xlabel('Exploitation Score')
ax.set_ylabel('log₁₀(View Count)')
ax.set_title(f'Exploitation Score vs. Video Popularity\n(Spearman ρ = {corr:.3f}, p < 10⁻³⁹, n = {len(valid):,})')
ax.legend(frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig4_score_vs_views.png')
plt.savefig(f'{FIGURES_DIR}/fig4_score_vs_views.pdf')
plt.close()
print("  Saved fig4")

# ============================================================
# FIGURE 5: Channel-level analysis
# ============================================================
print("Generating Figure 5: Channel-level analysis...")

fig, ax = plt.subplots(figsize=(10, 7))

sig_mask = ch_df['p_value'] < 0.05
ax.scatter(ch_df[~sig_mask]['mean_exploit_score'], 
          np.log10(ch_df[~sig_mask]['high_exploit_median_views']),
          s=ch_df[~sig_mask]['n']*3, alpha=0.5, c='#bdc3c7', edgecolor='white', label='p >= 0.05')
ax.scatter(ch_df[sig_mask]['mean_exploit_score'], 
          np.log10(ch_df[sig_mask]['high_exploit_median_views']),
          s=ch_df[sig_mask]['n']*3, alpha=0.7, c='#e74c3c', edgecolor='white', label='p < 0.05')

for _, row in ch_df.iterrows():
    ax.annotate(row['channel'],
               (row['mean_exploit_score'], np.log10(row['high_exploit_median_views'])),
               fontsize=6, alpha=0.6)

ax.set_xlabel('Mean Exploitation Score')
ax.set_ylabel('log₁₀(Median Views for High-Exploit Videos)')
ax.set_title('Channel-Level Exploitation vs. Popularity\n(bubble size = sample size; red = significant within-channel boost)')
ax.legend(frameon=True, fancybox=True, markerscale=0.5)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig5_channel_analysis.png')
plt.savefig(f'{FIGURES_DIR}/fig5_channel_analysis.pdf')
plt.close()
print("  Saved fig5")

# ============================================================
# ADDITIONAL STATISTICS FOR PAPER
# ============================================================
print("\n" + "=" * 60)
print("ADDITIONAL STATISTICS FOR PAPER")
print("=" * 60)

# Effect size (Cohen's d) for each dimension
print("\n--- Effect Sizes (Cohen's d) ---")
effect_sizes = {}
for dim in dimensions:
    dim_1_views = valid[valid[dim] == 1]['log_views']
    dim_0_views = valid[valid[dim] == 0]['log_views']
    if len(dim_1_views) > 5 and len(dim_0_views) > 5:
        pooled_std = np.sqrt(
            ((len(dim_1_views)-1)*dim_1_views.std()**2 + (len(dim_0_views)-1)*dim_0_views.std()**2) /
            (len(dim_1_views) + len(dim_0_views) - 2)
        )
        cohens_d = (dim_1_views.mean() - dim_0_views.mean()) / pooled_std
        u_stat, u_pval = stats.mannwhitneyu(dim_1_views, dim_0_views, alternative='two-sided')
        effect_sizes[dim] = {'cohens_d': cohens_d, 'mann_whitney_p': u_pval}
        print(f"  {dim}: Cohen's d = {cohens_d:.3f}, Mann-Whitney p = {u_pval:.4e}")

# OLS with channel fixed effects
print("\n--- OLS Regression (log_views ~ dimensions + channel FE) ---")
from numpy.linalg import lstsq

channel_dummies = pd.get_dummies(valid['channel_short_name'], prefix='ch', drop_first=True).astype(float)
X_df = pd.concat([valid[dimensions].reset_index(drop=True).astype(float), channel_dummies.reset_index(drop=True)], axis=1)
X = np.column_stack([np.ones(len(X_df)), X_df.values.astype(float)])
y = valid['log_views'].values.astype(float)

coeffs, residuals, rank, sv = lstsq(X, y, rcond=None)
y_pred = X @ coeffs
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r_squared = 1 - ss_res / ss_tot
n = len(y)
p = X.shape[1]
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print(f"  R² = {r_squared:.4f}, Adjusted R² = {adj_r_squared:.4f}")
print(f"  Dimension coefficients (with channel fixed effects):")
dim_coeffs = {}
for i, dim in enumerate(dimensions):
    dim_coeffs[dim] = float(coeffs[i+1])
    print(f"    {dim}: beta = {coeffs[i+1]:.4f}")

# Standard errors
mse = ss_res / (n - p)
try:
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    se_beta = np.sqrt(var_beta)
    print(f"\n  Standard errors and t-statistics:")
    for i, dim in enumerate(dimensions):
        t_val = coeffs[i+1] / se_beta[i+1]
        p_val_t = 2 * (1 - stats.t.cdf(abs(t_val), n - p))
        sig = "***" if p_val_t < 0.001 else "**" if p_val_t < 0.01 else "*" if p_val_t < 0.05 else ""
        print(f"    {dim}: beta={coeffs[i+1]:.4f}, SE={se_beta[i+1]:.4f}, t={t_val:.2f}, p={p_val_t:.4f} {sig}")
except np.linalg.LinAlgError:
    print("  (Could not compute standard errors - singular matrix)")

# Save extended stats
extended_stats = {
    'ols_r_squared': float(r_squared),
    'ols_adj_r_squared': float(adj_r_squared),
    'dimension_coefficients': dim_coeffs,
    'effect_sizes': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in effect_sizes.items()},
    'n_valid_views': int(len(valid)),
    'mean_views': float(valid['viewCount'].mean()),
    'median_views': float(valid['viewCount'].median()),
    'spearman_rho': float(corr),
    'spearman_p': float(pval),
}
with open(f'{FIGURES_DIR}/extended_statistics.json', 'w') as f:
    json.dump(extended_stats, f, indent=2)

print(f"\nAll figures saved to {FIGURES_DIR}/")
print("Done!")
