"""
Fast ML classification for manipulation detection in kidfluencer titles.
Uses SGDClassifier for speed on large sparse matrices.
"""
import pandas as pd
import numpy as np
import warnings, json, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Load
df = pd.read_csv('data/manipulation_detection/labeled_titles.csv')
df['is_manipulative'] = (df['manipulation_category'] != 'NEUTRAL').astype(int)
feature_cols = ['title_length', 'word_count', 'caps_ratio', 'exclamation_count',
                'question_count', 'all_caps_words', 'has_emoji', 'has_ellipsis',
                'has_asterisk', 'num_count']
df[feature_cols] = df[feature_cols].fillna(0).astype(float)
df['title'] = df['title'].fillna('')

print(f"Dataset: {len(df)} videos, {df['is_manipulative'].mean():.1%} manipulative")

# Features - use smaller TF-IDF for speed
tfidf_word = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english', min_df=5)
tfidf_char = TfidfVectorizer(max_features=2000, analyzer='char_wb', ngram_range=(3,5), min_df=5)
X_word = tfidf_word.fit_transform(df['title'])
X_char = tfidf_char.fit_transform(df['title'])
X_hand = csr_matrix(df[feature_cols].values)
X_full = hstack([X_word, X_char, X_hand])
print(f"Features: word={X_word.shape[1]}, char={X_char.shape[1]}, hand={X_hand.shape[1]}, total={X_full.shape[1]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============ BINARY CLASSIFICATION ============
print("\n=== BINARY CLASSIFICATION (Manipulative vs Neutral) ===")
y_bin = df['is_manipulative'].values

configs = [
    ('SGD-LR (word)', X_word),
    ('SGD-LR (word+char)', hstack([X_word, X_char])),
    ('SGD-LR (full)', X_full),
    ('SGD-LR (handcrafted only)', X_hand),
]

results = {}
for name, X in configs:
    model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
    yp = cross_val_predict(model, X, y_bin, cv=cv)
    acc = accuracy_score(y_bin, yp)
    f1 = f1_score(y_bin, yp, average='macro')
    results[name] = {'acc': acc, 'f1_macro': f1}
    print(f"  {name:30s} Acc={acc:.4f} F1={f1:.4f}")

# ============ MULTI-CLASS ============
print("\n=== MULTI-CLASS (6 categories) ===")
le = LabelEncoder()
y_multi = le.fit_transform(df['manipulation_category'])

model_mc = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, class_weight='balanced')
yp_mc = cross_val_predict(model_mc, X_full, y_multi, cv=cv)
print(classification_report(y_multi, yp_mc, target_names=le.classes_))
mc_acc = accuracy_score(y_multi, yp_mc)
mc_f1_macro = f1_score(y_multi, yp_mc, average='macro')
mc_f1_weighted = f1_score(y_multi, yp_mc, average='weighted')
print(f"Overall: Acc={mc_acc:.4f}, F1_macro={mc_f1_macro:.4f}, F1_weighted={mc_f1_weighted:.4f}")

# ============ FEATURE IMPORTANCE ============
print("\n=== TOP PREDICTIVE WORDS ===")
model_fi = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
model_fi.fit(X_word, y_bin)
feat_names = tfidf_word.get_feature_names_out()
coefs = model_fi.coef_[0]
top_pos = np.argsort(coefs)[-20:][::-1]
print("Top 20 words predicting MANIPULATIVE:")
for i in top_pos:
    print(f"  {feat_names[i]:25s} coef={coefs[i]:.3f}")

top_neg = np.argsort(coefs)[:10]
print("\nTop 10 words predicting NEUTRAL:")
for i in top_neg:
    print(f"  {feat_names[i]:25s} coef={coefs[i]:.3f}")

# ============ VIEW BOOST ANALYSIS ============
print("\n=== VIEW BOOST BY CATEGORY ===")
neutral_median = df[df['manipulation_category'] == 'NEUTRAL']['viewCount'].median()
print(f"NEUTRAL baseline: median views = {neutral_median:,.0f}")
for cat in ['STAGED_CONFLICT', 'FAKE_EMERGENCY', 'EMOTIONAL_BAIT', 'CHALLENGE_DARE', 'DECEPTION_NARRATIVE']:
    cat_vids = df[df['manipulation_category'] == cat]
    if len(cat_vids) > 10:
        med = cat_vids['viewCount'].median()
        boost = med / neutral_median - 1
        print(f"  {cat:25s} median={med:>12,.0f}  boost={boost:>+7.1%}  n={len(cat_vids)}")

# ============ CHANNEL-LEVEL ANALYSIS ============
print("\n=== CHANNEL-LEVEL MANIPULATION RATES ===")
ch = df.groupby('channel_short_name').agg(
    n_videos=('title', 'count'),
    manip_rate=('is_manipulative', 'mean'),
    mean_views=('viewCount', 'mean'),
    median_views=('viewCount', 'median'),
).sort_values('manip_rate', ascending=False)
print(ch.head(10).to_string())

# Correlation: manipulation rate vs views
from scipy.stats import spearmanr
rho, p = spearmanr(ch['manip_rate'], ch['mean_views'])
print(f"\nSpearman correlation (manip_rate vs mean_views): rho={rho:.3f}, p={p:.4f}")

# ============ SAVE RESULTS ============
os.makedirs('analysis_manipulation', exist_ok=True)
all_results = {
    'binary': results,
    'multiclass': {'acc': mc_acc, 'f1_macro': mc_f1_macro, 'f1_weighted': mc_f1_weighted},
    'view_boost': {cat: float(df[df['manipulation_category']==cat]['viewCount'].median() / neutral_median - 1) 
                   for cat in ['STAGED_CONFLICT','FAKE_EMERGENCY','EMOTIONAL_BAIT','CHALLENGE_DARE','DECEPTION_NARRATIVE']},
}
with open('analysis_manipulation/results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=float)

print("\n=== DONE ===")
