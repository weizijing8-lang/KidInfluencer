"""
Generate publication-quality figures for Paper 1:
"A Computational Analysis of Kidfluencer Content Strategies"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Output directory
import os
outdir = '/home/ubuntu/KidInfluencer/docs/figures'
os.makedirs(outdir, exist_ok=True)

# Load data
nlp = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/nlp/title_nlp_features.csv')
cv = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/thumbnails/thumbnail_cv_features.csv')
vid = pd.read_csv('/home/ubuntu/KidInfluencer/data/combined_videos.csv')

# Merge
df = vid.merge(nlp, on='video_id', how='inner', suffixes=('', '_nlp'))
df = df.merge(cv, on='video_id', how='inner', suffixes=('', '_cv'))
df = df[df['views'] > 0].copy()
df['log_views'] = np.log(df['views'])

print(f"Dataset: {len(df)} videos")

# ============================================================
# FIGURE 1: Prevalence of Content Strategies (Bar Chart)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# NLP strategies
nlp_strategies = {
    'ALL CAPS\nwords': (df['caps_word_count'] > 0).mean() * 100,
    'Exclamation\nmarks': (df['exclamation_count'] > 0).mean() * 100,
    'Emojis': (df['emoji_count'] > 0).mean() * 100,
    'First-person\npronouns': df['has_first_person'].mean() * 100,
    'Challenge\nkeywords': df['has_challenge'].mean() * 100,
}

bars1 = axes[0].bar(nlp_strategies.keys(), nlp_strategies.values(), 
                     color='#4C72B0', alpha=0.85, edgecolor='white', linewidth=0.5)
axes[0].set_ylabel('Percentage of Videos (%)')
axes[0].set_title('(a) Title Strategies (NLP)')
axes[0].set_ylim(0, 55)
for bar, val in zip(bars1, nlp_strategies.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# CV strategies
cv_strategies = {
    'Faces\npresent': (df['num_faces'] > 0).mean() * 100,
    'Open\nmouth': df['has_open_mouth'].mean() * 100,
    'Text\noverlay': df['has_text_overlay'].mean() * 100,
    'High\nbrightness': (df['brightness'] > df['brightness'].median()).mean() * 100,
    'High\ncolorfulness': (df['colorfulness'] > df['colorfulness'].median()).mean() * 100,
}

bars2 = axes[1].bar(cv_strategies.keys(), cv_strategies.values(), 
                     color='#DD8452', alpha=0.85, edgecolor='white', linewidth=0.5)
axes[1].set_ylabel('Percentage of Videos (%)')
axes[1].set_title('(b) Thumbnail Strategies (CV)')
axes[1].set_ylim(0, 100)
for bar, val in zip(bars2, cv_strategies.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{outdir}/fig1_strategy_prevalence.png')
plt.close()
print("Figure 1 saved: Strategy prevalence")

# ============================================================
# FIGURE 2: Clickbait Strategy Impact on Views (Box + Violin)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

comparisons = [
    ('has_open_mouth', 'Open Mouth in Thumbnail', axes[0, 0]),
    ('has_text_overlay', 'Text Overlay in Thumbnail', axes[0, 1]),
    (None, 'ALL CAPS Words in Title', axes[1, 0]),  # custom
    (None, 'Exclamation Marks in Title', axes[1, 1]),  # custom
]

# Open mouth
for ax_idx, (col, title, ax) in enumerate(comparisons):
    if col:
        data_yes = df[df[col] == 1]['log_views']
        data_no = df[df[col] == 0]['log_views']
    elif 'CAPS' in title:
        data_yes = df[df['caps_word_count'] > 0]['log_views']
        data_no = df[df['caps_word_count'] == 0]['log_views']
    else:
        data_yes = df[df['exclamation_count'] > 0]['log_views']
        data_no = df[df['exclamation_count'] == 0]['log_views']
    
    parts = ax.violinplot([data_no, data_yes], positions=[0, 1], showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.3)
    parts['bodies'][0].set_facecolor('#4C72B0')
    parts['bodies'][1].set_facecolor('#DD8452')
    
    bp = ax.boxplot([data_no, data_yes], positions=[0, 1], widths=0.2, 
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#4C72B0')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#DD8452')
    bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Log Views')
    ax.set_title(title)
    
    # Add median annotation
    med_no = data_no.median()
    med_yes = data_yes.median()
    pct_diff = (np.exp(med_yes) - np.exp(med_no)) / np.exp(med_no) * 100
    ax.text(0.5, 0.95, f'+{pct_diff:.0f}% median views', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold', color='#C44E52',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{outdir}/fig2_clickbait_impact.png')
plt.close()
print("Figure 2 saved: Clickbait impact")

# ============================================================
# FIGURE 3: Hierarchical Regression R² (Stacked Bar)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))

models = ['Model 1\n(Controls)', 'Model 2\n(+ NLP)', 'Model 3\n(+ NLP + CV)']
r2_controls = [0.5685, 0.5685, 0.5685]
r2_nlp = [0, 0.0215, 0.0215]
r2_cv = [0, 0, 0.0281]

x = np.arange(len(models))
width = 0.5

bars1 = ax.bar(x, r2_controls, width, label='Controls (log_subs + duration)', color='#4C72B0', alpha=0.85)
bars2 = ax.bar(x, r2_nlp, width, bottom=r2_controls, label='NLP Features (ΔR² = 0.022)', color='#55A868', alpha=0.85)
bars3 = ax.bar(x, r2_cv, width, bottom=[a+b for a,b in zip(r2_controls, r2_nlp)], 
               label='CV Features (ΔR² = 0.028)', color='#DD8452', alpha=0.85)

ax.set_ylabel('R²')
ax.set_title('Hierarchical Regression: Incremental Variance Explained')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.75)
ax.legend(loc='upper left')

# Add R² labels on top
for i, (c, n, cv_val) in enumerate(zip(r2_controls, r2_nlp, r2_cv)):
    total = c + n + cv_val
    ax.text(i, total + 0.01, f'R² = {total:.4f}', ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_hierarchical_r2.png')
plt.close()
print("Figure 3 saved: Hierarchical R²")

# ============================================================
# FIGURE 4: Significant Regression Coefficients (Forest Plot)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# From Model 3 results
coefs = {
    'First-person language': (0.4435, 0.060, '***'),
    'Exclamation marks': (0.0840, 0.033, '**'),
    'Number of faces': (0.0426, 0.012, '***'),
    'Brightness': (0.0091, 0.001, '***'),
    'Edge density': (6.8357, 0.783, '***'),
    'Duration (min)': (0.0116, 0.001, '***'),
    'Emoji count': (-0.1084, 0.037, '**'),
    'Family word': (-0.1392, 0.056, '*'),
    'Question marks': (-0.3497, 0.067, '***'),
    'Prank keyword': (-0.8810, 0.312, '**'),
    'Contrast': (-0.0141, 0.002, '***'),
}

# Sort by coefficient
sorted_coefs = sorted(coefs.items(), key=lambda x: x[1][0])
labels = [k for k, v in sorted_coefs]
values = [v[0] for k, v in sorted_coefs]
errors = [1.96 * v[1] for k, v in sorted_coefs]
sigs = [v[2] for k, v in sorted_coefs]

colors = ['#C44E52' if v < 0 else '#55A868' for v in values]

y_pos = np.arange(len(labels))
ax.barh(y_pos, values, xerr=errors, color=colors, alpha=0.75, edgecolor='white', 
        linewidth=0.5, capsize=3)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f'{l} {s}' for l, s in zip(labels, sigs)])
ax.set_xlabel('Coefficient (effect on log views)')
ax.set_title('Significant Predictors of Video Views (OLS, p < 0.05)')

# Note about edge density scale
ax.text(0.98, 0.02, 'Note: Edge density is on [0,1] scale\n*** p<0.001, ** p<0.01, * p<0.05', 
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(f'{outdir}/fig4_forest_plot.png')
plt.close()
print("Figure 4 saved: Forest plot")

# ============================================================
# FIGURE 5: Channel-level clickbait intensity distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

clickbait_scores = pd.read_csv('/home/ubuntu/KidInfluencer/analysis_v3/combined/channel_clickbait_scores.csv')

ax.hist(clickbait_scores['clickbait_intensity'], bins=25, color='#4C72B0', alpha=0.7, edgecolor='white')
ax.axvline(x=clickbait_scores['clickbait_intensity'].mean(), color='#C44E52', linestyle='--', linewidth=2,
           label=f'Mean = {clickbait_scores["clickbait_intensity"].mean():.3f}')
ax.axvline(x=clickbait_scores['clickbait_intensity'].quantile(0.75), color='#DD8452', linestyle=':', linewidth=2,
           label=f'75th percentile = {clickbait_scores["clickbait_intensity"].quantile(0.75):.3f}')
ax.set_xlabel('Clickbait Intensity Score')
ax.set_ylabel('Number of Channels')
ax.set_title('Distribution of Channel-Level Clickbait Intensity')
ax.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/fig5_clickbait_distribution.png')
plt.close()
print("Figure 5 saved: Clickbait distribution")

# ============================================================
# FIGURE 6: Correlation between NLP and CV features
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

feature_cols = ['caps_ratio', 'exclamation_count', 'emoji_count', 'has_first_person',
                'num_faces', 'has_open_mouth', 'has_text_overlay', 'brightness', 
                'colorfulness', 'edge_density', 'contrast']
feature_labels = ['CAPS ratio', 'Exclamation', 'Emoji', 'First person',
                  'Num faces', 'Open mouth', 'Text overlay', 'Brightness',
                  'Colorfulness', 'Edge density', 'Contrast']

corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.5, vmax=0.5, square=True, ax=ax,
            xticklabels=feature_labels, yticklabels=feature_labels,
            annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix (NLP + CV)')
plt.tight_layout()
plt.savefig(f'{outdir}/fig6_correlation_matrix.png')
plt.close()
print("Figure 6 saved: Correlation matrix")

print("\n=== ALL FIGURES GENERATED ===")
print(f"Output directory: {outdir}")
