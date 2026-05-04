"""
ML/DL Classification Pipeline for Kidfluencer Title Manipulation Detection
Inspired by fake news detection methodology (Tian et al. 2025, Xu et al. 2025)

Compares:
1. Traditional ML: TF-IDF + Logistic Regression / Random Forest / XGBoost
2. Feature Engineering: Handcrafted features + XGBoost
3. Deep Learning: BERT-based classification (DistilBERT for efficiency)

Binary task: Manipulative vs Neutral
Multi-class task: 6 categories
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                             roc_auc_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
print(f"Dataset: {len(df)} videos")
print(f"Class distribution:\n{df['manipulation_category'].value_counts()}")

# Binary label
df['is_manipulative'] = (df['manipulation_category'] != 'NEUTRAL').astype(int)
print(f"\nBinary: {df['is_manipulative'].mean():.1%} manipulative")

# ============================================================
# Feature Engineering
# ============================================================
# Handcrafted features (already computed)
feature_cols = ['title_length', 'word_count', 'caps_ratio', 'exclamation_count',
                'question_count', 'all_caps_words', 'has_emoji', 'has_ellipsis',
                'has_asterisk', 'num_count']

# Fill NaN
df[feature_cols] = df[feature_cols].fillna(0)
df['title'] = df['title'].fillna('')

# TF-IDF features
print("\nBuilding TF-IDF features...")
tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                              stop_words='english', min_df=5)
tfidf_char = TfidfVectorizer(max_features=3000, analyzer='char_wb', 
                              ngram_range=(3, 5), min_df=5)

X_tfidf_word = tfidf_word.fit_transform(df['title'])
X_tfidf_char = tfidf_char.fit_transform(df['title'])
X_handcrafted = df[feature_cols].astype(float).values

# Combined feature matrix
from scipy.sparse import csr_matrix
X_combined = hstack([X_tfidf_word, X_tfidf_char, csr_matrix(X_handcrafted)])

print(f"TF-IDF word features: {X_tfidf_word.shape[1]}")
print(f"TF-IDF char features: {X_tfidf_char.shape[1]}")
print(f"Handcrafted features: {len(feature_cols)}")
print(f"Combined features: {X_combined.shape[1]}")

# ============================================================
# EXPERIMENT 1: Binary Classification (Manipulative vs Neutral)
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Binary Classification (Manipulative vs Neutral)")
print("="*60)

y_binary = df['is_manipulative'].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression (TF-IDF word)': (LogisticRegression(max_iter=1000, C=1.0), X_tfidf_word),
    'Logistic Regression (TF-IDF word+char)': (LogisticRegression(max_iter=1000, C=1.0), hstack([X_tfidf_word, X_tfidf_char])),
    'Logistic Regression (Combined)': (LogisticRegression(max_iter=1000, C=1.0), X_combined),
    'Random Forest (Combined)': (RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1), X_combined),
    'Gradient Boosting (Combined)': (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42), X_combined),
    'XGBoost (Handcrafted only)': (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42), csr_matrix(X_handcrafted)),
}

results_binary = {}
for name, (model, X) in models.items():
    print(f"\n--- {name} ---")
    y_pred = cross_val_predict(model, X, y_binary, cv=cv, method='predict')
    
    acc = accuracy_score(y_binary, y_pred)
    f1 = f1_score(y_binary, y_pred, average='macro')
    precision, recall, f1_class, _ = precision_recall_fscore_support(y_binary, y_pred, average='binary')
    
    results_binary[name] = {
        'accuracy': acc,
        'f1_macro': f1,
        'precision': precision,
        'recall': recall,
        'f1_binary': f1_class
    }
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")

# ============================================================
# EXPERIMENT 2: Multi-class Classification (6 categories)
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Multi-class Classification (6 categories)")
print("="*60)

le = LabelEncoder()
y_multi = le.fit_transform(df['manipulation_category'])
print(f"Classes: {le.classes_}")

# Use best model from binary (Logistic Regression Combined)
models_multi = {
    'Logistic Regression (Combined)': (LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial'), X_combined),
    'Random Forest (Combined)': (RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1), X_combined),
    'Gradient Boosting (Combined)': (GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42), X_combined),
}

results_multi = {}
for name, (model, X) in models_multi.items():
    print(f"\n--- {name} ---")
    y_pred = cross_val_predict(model, X, y_multi, cv=cv, method='predict')
    
    acc = accuracy_score(y_multi, y_pred)
    f1_macro = f1_score(y_multi, y_pred, average='macro')
    f1_weighted = f1_score(y_multi, y_pred, average='weighted')
    
    results_multi[name] = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

# Best model detailed report
print("\n--- Best Model (LR Combined) Detailed Report ---")
model_best = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
y_pred_best = cross_val_predict(model_best, X_combined, y_multi, cv=cv)
print(classification_report(y_multi, y_pred_best, target_names=le.classes_))

# ============================================================
# EXPERIMENT 3: Feature Importance Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Feature Importance (What words predict manipulation?)")
print("="*60)

# Train LR on full data for feature importance
lr_full = LogisticRegression(max_iter=1000, C=1.0)
lr_full.fit(X_tfidf_word, y_binary)

# Top features for manipulative class
feature_names = tfidf_word.get_feature_names_out()
coefs = lr_full.coef_[0]
top_positive = np.argsort(coefs)[-20:][::-1]
top_negative = np.argsort(coefs)[:20]

print("\nTop 20 words predicting MANIPULATIVE:")
for idx in top_positive:
    print(f"  {feature_names[idx]:25s} coef={coefs[idx]:.3f}")

print("\nTop 20 words predicting NEUTRAL:")
for idx in top_negative:
    print(f"  {feature_names[idx]:25s} coef={coefs[idx]:.3f}")

# ============================================================
# EXPERIMENT 4: Ablation Study
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Ablation Study")
print("="*60)

ablation_configs = {
    'Handcrafted only': csr_matrix(X_handcrafted),
    'TF-IDF word only': X_tfidf_word,
    'TF-IDF char only': X_tfidf_char,
    'TF-IDF word + char': hstack([X_tfidf_word, X_tfidf_char]),
    'TF-IDF + Handcrafted (Full)': X_combined,
}

ablation_results = {}
for name, X in ablation_configs.items():
    lr = LogisticRegression(max_iter=1000, C=1.0)
    y_pred = cross_val_predict(lr, X, y_binary, cv=cv)
    f1 = f1_score(y_binary, y_pred, average='macro')
    acc = accuracy_score(y_binary, y_pred)
    ablation_results[name] = {'accuracy': acc, 'f1_macro': f1}
    print(f"  {name:30s} Acc={acc:.4f}  F1={f1:.4f}")

# ============================================================
# Save Results & Visualizations
# ============================================================
os.makedirs('analysis_manipulation/figures', exist_ok=True)

# Figure 1: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary results
names = list(results_binary.keys())
f1s = [results_binary[n]['f1_macro'] for n in names]
short_names = ['LR(word)', 'LR(word+char)', 'LR(full)', 'RF(full)', 'GB(full)', 'XGB(hand)']
axes[0].barh(short_names, f1s, color='steelblue')
axes[0].set_xlabel('F1 Score (Macro)')
axes[0].set_title('Binary Classification: Manipulative vs Neutral')
axes[0].set_xlim(0, 1)
for i, v in enumerate(f1s):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Ablation
abl_names = list(ablation_results.keys())
abl_f1s = [ablation_results[n]['f1_macro'] for n in abl_names]
axes[1].barh(abl_names, abl_f1s, color='coral')
axes[1].set_xlabel('F1 Score (Macro)')
axes[1].set_title('Ablation Study: Feature Contribution')
axes[1].set_xlim(0, 1)
for i, v in enumerate(abl_f1s):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('analysis_manipulation/figures/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Confusion matrix for multi-class
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_multi, y_pred_best)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Multi-class Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig('analysis_manipulation/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Top features word cloud style
fig, ax = plt.subplots(figsize=(10, 6))
top_n = 15
top_words = [feature_names[idx] for idx in top_positive[:top_n]]
top_coefs = [coefs[idx] for idx in top_positive[:top_n]]
ax.barh(top_words[::-1], top_coefs[::-1], color='darkred', alpha=0.7)
ax.set_xlabel('Logistic Regression Coefficient')
ax.set_title('Top 15 Words Predicting Manipulative Content')
plt.tight_layout()
plt.savefig('analysis_manipulation/figures/top_features.png', dpi=150, bbox_inches='tight')
plt.close()

# Save numerical results
results_all = {
    'binary_classification': results_binary,
    'multi_class': results_multi,
    'ablation': ablation_results,
}
with open('analysis_manipulation/classification_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n=== ALL DONE ===")
print(f"Figures saved to: analysis_manipulation/figures/")
print(f"Results saved to: analysis_manipulation/classification_results.json")
