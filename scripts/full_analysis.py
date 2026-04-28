"""
Full Analysis Pipeline for Kidfluencer Exploitation Study
==========================================================
1. Compute sentence-transformers embeddings for all video titles
2. Build exploitation direction vector from anchor cases
3. Compute drift scores for all videos
4. Identify viral hits (view_count > μ + 2σ per channel)
5. Run Difference-in-Differences (DiD) causal inference

No GPU required - all-MiniLM-L6-v2 runs fast on CPU for ~100K titles.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: Load data and compute embeddings
# ============================================================

def load_all_videos():
    """Load all videos from raw JSON files."""
    print("Loading all video data...")
    all_videos = []
    
    for f in sorted(RAW_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f) as fp:
            data = json.load(fp)
        if data.get("error") or not data.get("videos"):
            continue
        for v in data["videos"]:
            v["channel_short_name"] = data["short_name"]
            v["channel_category"] = data["category"]
            v["channel_title"] = data.get("channel_title", "")
            all_videos.append(v)
    
    df = pd.DataFrame(all_videos)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.dropna(subset=['publishedAt'])
    df = df.sort_values(['channel_short_name', 'publishedAt']).reset_index(drop=True)
    print(f"  Loaded {len(df):,} videos from {df['channel_short_name'].nunique()} channels")
    return df


def compute_embeddings(df, batch_size=512):
    """Compute sentence-transformers embeddings for all video titles."""
    from sentence_transformers import SentenceTransformer
    
    embeddings_file = RESULTS_DIR / "title_embeddings.npy"
    
    if embeddings_file.exists():
        print(f"  Loading cached embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        if len(embeddings) == len(df):
            return embeddings
        print(f"  Cache size mismatch ({len(embeddings)} vs {len(df)}), recomputing...")
    
    print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    titles = df['title'].fillna("").tolist()
    print(f"  Computing embeddings for {len(titles):,} titles (batch_size={batch_size})...")
    
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    
    # Save embeddings
    np.save(embeddings_file, embeddings)
    print(f"  Embeddings saved to {embeddings_file} (shape: {embeddings.shape})")
    
    return embeddings


# ============================================================
# STEP 2: Build exploitation direction vector
# ============================================================

def build_direction_vector(model=None):
    """
    Build the exploitation direction vector using anchor titles.
    
    Exploitation anchors: Titles characteristic of exploitative content
    (based on known cases: 8 Passengers, DaddyOFive, extreme family clickbait)
    
    Healthy anchors: Titles characteristic of healthy, non-exploitative content
    (educational, cooking, travel, technology)
    """
    from sentence_transformers import SentenceTransformer
    
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Exploitation anchor titles (based on known extreme cases)
    exploitation_anchors = [
        # 8 Passengers / Ruby Franke style
        "Kids get punished for not doing chores",
        "No food until homework is done punishment",
        "Child crying after being grounded for a week",
        "Kids lose their room as punishment",
        "Making my kids sleep outside as discipline",
        "Child begs to go back to school",
        "Taking away all my kids toys forever",
        "Kids get no Christmas presents this year",
        
        # DaddyOFive style (pranks on kids)
        "Prank on my kids gone wrong they cried",
        "Telling my kids we're giving them away",
        "Fake emergency prank on children",
        "Making my daughter cry prank",
        "Destroying my son's favorite toy prank",
        "Telling kids they're adopted prank",
        "Screaming at my kids prank",
        
        # Exploitative clickbait with children
        "My 5 year old daughter's first boyfriend",
        "Kid gets plastic surgery at age 10",
        "Letting my toddler drive a car",
        "Baby does dangerous stunt gone wrong",
        "Child eats world's hottest pepper challenge",
        "Forcing my kids to do extreme challenge",
        "Kid stays awake for 24 hours challenge",
        "Making my child do embarrassing dare in public",
        
        # Emotional exploitation
        "My child's emotional breakdown on camera",
        "Kids react to parents divorce announcement",
        "Telling my kids their pet died on camera",
        "Child's worst day ever filmed everything",
        "Making my kid cry for views",
        "Exposing my child's secrets on YouTube",
    ]
    
    # Healthy anchor titles (non-exploitative content)
    healthy_anchors = [
        # Educational / wholesome
        "How to make homemade pasta from scratch",
        "Best budget travel tips for Europe 2024",
        "iPhone 15 Pro review after one month",
        "Learning to play piano in 30 days",
        "My morning routine for productivity",
        "How I organize my workspace",
        "Best books I read this year",
        "Trying street food in Bangkok Thailand",
        
        # Adult vlog / lifestyle
        "Moving to a new apartment tour",
        "Day in my life as a software engineer",
        "Grocery haul and meal prep Sunday",
        "Working out at the gym after a long break",
        "Road trip across the country vlog",
        "Cooking dinner for my friends",
        "Reviewing the best coffee shops in NYC",
        "My honest thoughts on minimalism",
        
        # Technology / science
        "Building a custom PC from scratch",
        "The science behind black holes explained",
        "How algorithms actually work",
        "Testing the fastest internet in the world",
        "Why this math problem is unsolvable",
        "The engineering behind skyscrapers",
        
        # Healthy family content (positive, non-exploitative)
        "Family cooking challenge everyone wins",
        "Teaching my kids to ride bikes",
        "Family vacation highlights best moments",
        "Kids learn about nature at the park",
        "Family game night fun board games",
        "Reading bedtime stories together",
    ]
    
    print("  Computing anchor embeddings...")
    exploit_embs = model.encode(exploitation_anchors, normalize_embeddings=True)
    healthy_embs = model.encode(healthy_anchors, normalize_embeddings=True)
    
    # Direction vector: mean(exploitation) - mean(healthy)
    exploit_centroid = exploit_embs.mean(axis=0)
    healthy_centroid = healthy_embs.mean(axis=0)
    
    direction = exploit_centroid - healthy_centroid
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    print(f"  Direction vector computed (dim={len(direction)})")
    print(f"  Exploitation centroid norm: {np.linalg.norm(exploit_centroid):.4f}")
    print(f"  Healthy centroid norm: {np.linalg.norm(healthy_centroid):.4f}")
    
    # Save direction vector
    np.save(RESULTS_DIR / "direction_vector.npy", direction)
    
    return direction


# ============================================================
# STEP 3: Compute drift scores
# ============================================================

def compute_drift_scores(embeddings, direction):
    """Compute drift score = dot product of each embedding with direction vector."""
    print("  Computing drift scores (cosine projection onto direction vector)...")
    # Since embeddings are already L2-normalized, dot product = cosine similarity
    drift_scores = embeddings @ direction
    print(f"  Drift scores: mean={drift_scores.mean():.4f}, std={drift_scores.std():.4f}")
    print(f"  Range: [{drift_scores.min():.4f}, {drift_scores.max():.4f}]")
    return drift_scores


# ============================================================
# STEP 4: Identify viral hits
# ============================================================

def identify_viral_hits(df):
    """
    Identify viral hits per channel: videos with view_count > μ + 2σ.
    Returns a boolean Series.
    """
    print("  Identifying viral hits (view_count > μ + 2σ per channel)...")
    
    # Compute per-channel mean and std
    channel_stats = df.groupby('channel_short_name')['viewCount'].agg(['mean', 'std']).reset_index()
    channel_stats.columns = ['channel_short_name', 'channel_mean_views', 'channel_std_views']
    
    df_merged = df.merge(channel_stats, on='channel_short_name')
    threshold = df_merged['channel_mean_views'] + 2 * df_merged['channel_std_views']
    is_viral = df['viewCount'] > threshold
    
    n_viral = is_viral.sum()
    print(f"  Viral hits: {n_viral:,} / {len(df):,} ({100*n_viral/len(df):.1f}%)")
    
    # Per-category breakdown
    family_viral = is_viral[df['channel_category'] == 'family'].sum()
    adult_viral = is_viral[df['channel_category'] == 'adult'].sum()
    family_total = (df['channel_category'] == 'family').sum()
    adult_total = (df['channel_category'] == 'adult').sum()
    print(f"    Family: {family_viral:,} / {family_total:,} ({100*family_viral/family_total:.1f}%)")
    print(f"    Adult:  {adult_viral:,} / {adult_total:,} ({100*adult_viral/adult_total:.1f}%)")
    
    return is_viral


# ============================================================
# STEP 5: Difference-in-Differences (DiD) Causal Inference
# ============================================================

def run_did_analysis(df):
    """
    Run Difference-in-Differences analysis.
    
    Treatment: Family channels
    Control: Adult channels
    Event: Viral hit (view_count > μ + 2σ)
    Outcome: Change in drift score after viral hit
    
    For each viral hit, we compare:
    - Pre-period: mean drift score of 10 videos before the viral hit
    - Post-period: mean drift score of 10 videos after the viral hit
    
    DiD estimand:
    ΔDrift_family_post_viral - ΔDrift_adult_post_viral
    """
    import statsmodels.api as sm
    
    print("\n  Running Difference-in-Differences (DiD) analysis...")
    
    WINDOW = 10  # videos before/after viral hit
    
    # For each channel, identify viral hits and compute pre/post drift
    did_observations = []
    
    for channel_name, group in df.groupby('channel_short_name'):
        group = group.sort_values('publishedAt').reset_index(drop=True)
        category = group['channel_category'].iloc[0]
        
        viral_indices = group.index[group['is_viral']].tolist()
        
        for viral_idx in viral_indices:
            # Need at least WINDOW videos before and after
            if viral_idx < WINDOW or viral_idx >= len(group) - WINDOW:
                continue
            
            pre_scores = group.iloc[viral_idx - WINDOW:viral_idx]['drift_score'].values
            post_scores = group.iloc[viral_idx + 1:viral_idx + 1 + WINDOW]['drift_score'].values
            
            if len(pre_scores) == WINDOW and len(post_scores) == WINDOW:
                drift_change = post_scores.mean() - pre_scores.mean()
                
                did_observations.append({
                    'channel': channel_name,
                    'category': category,
                    'is_family': 1 if category == 'family' else 0,
                    'viral_video_idx': viral_idx,
                    'viral_views': group.iloc[viral_idx]['viewCount'],
                    'pre_drift_mean': pre_scores.mean(),
                    'post_drift_mean': post_scores.mean(),
                    'drift_change': drift_change,
                })
    
    did_df = pd.DataFrame(did_observations)
    print(f"  DiD observations: {len(did_df):,}")
    print(f"    Family: {(did_df['is_family'] == 1).sum():,}")
    print(f"    Adult:  {(did_df['is_family'] == 0).sum():,}")
    
    if len(did_df) < 10:
        print("  [WARNING] Too few observations for reliable DiD analysis")
        return did_df, None
    
    # Simple DiD: Compare mean drift_change between family and adult
    family_change = did_df[did_df['is_family'] == 1]['drift_change']
    adult_change = did_df[did_df['is_family'] == 0]['drift_change']
    
    print(f"\n  --- DiD Results (Simple) ---")
    print(f"  Family mean drift change after viral hit: {family_change.mean():.6f} (±{family_change.std():.6f})")
    print(f"  Adult mean drift change after viral hit:  {adult_change.mean():.6f} (±{adult_change.std():.6f})")
    print(f"  DiD estimate (Family - Adult): {family_change.mean() - adult_change.mean():.6f}")
    
    # OLS regression: drift_change ~ is_family + controls
    print(f"\n  --- DiD Results (OLS Regression) ---")
    X = did_df[['is_family']].copy()
    X['log_viral_views'] = np.log1p(did_df['viral_views'])
    X = sm.add_constant(X)
    y = did_df['drift_change']
    
    model = sm.OLS(y, X).fit(cov_type='HC1')  # Heteroscedasticity-robust SEs
    print(model.summary2().tables[1].to_string())
    
    # Statistical significance
    coef = model.params['is_family']
    pval = model.pvalues['is_family']
    ci = model.conf_int().loc['is_family']
    
    print(f"\n  Key result:")
    print(f"  β(is_family) = {coef:.6f}")
    print(f"  p-value = {pval:.4f}")
    print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
    
    if pval < 0.05:
        print(f"  *** STATISTICALLY SIGNIFICANT at p < 0.05 ***")
        if coef > 0:
            print(f"  → Family channels show GREATER drift toward exploitation after viral hits")
        else:
            print(f"  → Family channels show LESS drift toward exploitation after viral hits")
    else:
        print(f"  Not statistically significant at p < 0.05")
    
    # Save DiD results
    did_df.to_csv(RESULTS_DIR / "did_observations.csv", index=False)
    
    # Save model summary
    with open(RESULTS_DIR / "did_regression_summary.txt", 'w') as f:
        f.write(str(model.summary2()))
    
    return did_df, model


# ============================================================
# STEP 6: Additional analyses
# ============================================================

def compute_channel_level_stats(df):
    """Compute channel-level drift statistics for visualization."""
    channel_stats = df.groupby(['channel_short_name', 'channel_category']).agg(
        n_videos=('id', 'count'),
        mean_drift=('drift_score', 'mean'),
        std_drift=('drift_score', 'std'),
        median_drift=('drift_score', 'median'),
        total_views=('viewCount', 'sum'),
        mean_views=('viewCount', 'mean'),
        n_viral=('is_viral', 'sum'),
        viral_rate=('is_viral', 'mean'),
    ).reset_index()
    
    channel_stats = channel_stats.sort_values('mean_drift', ascending=False)
    channel_stats.to_csv(RESULTS_DIR / "channel_drift_stats.csv", index=False)
    
    print(f"\n  --- Channel-Level Drift Scores ---")
    print(f"  Top 10 highest drift (most exploitation-like):")
    for _, row in channel_stats.head(10).iterrows():
        print(f"    {row['channel_short_name']:20s} ({row['channel_category']:6s}): "
              f"drift={row['mean_drift']:.4f}, {row['n_videos']} videos")
    
    print(f"\n  Bottom 10 lowest drift (least exploitation-like):")
    for _, row in channel_stats.tail(10).iterrows():
        print(f"    {row['channel_short_name']:20s} ({row['channel_category']:6s}): "
              f"drift={row['mean_drift']:.4f}, {row['n_videos']} videos")
    
    # Group comparison
    family_mean = channel_stats[channel_stats['channel_category'] == 'family']['mean_drift'].mean()
    adult_mean = channel_stats[channel_stats['channel_category'] == 'adult']['mean_drift'].mean()
    print(f"\n  Group means: Family={family_mean:.4f}, Adult={adult_mean:.4f}")
    
    return channel_stats


def temporal_drift_analysis(df):
    """Compute quarterly drift trends per channel category."""
    df['quarter'] = df['publishedAt'].dt.to_period('Q')
    
    quarterly = df.groupby(['quarter', 'channel_category']).agg(
        mean_drift=('drift_score', 'mean'),
        n_videos=('id', 'count'),
    ).reset_index()
    
    quarterly['quarter_str'] = quarterly['quarter'].astype(str)
    quarterly.to_csv(RESULTS_DIR / "quarterly_drift_trends.csv", index=False)
    
    print(f"\n  Quarterly drift trends saved to quarterly_drift_trends.csv")
    return quarterly


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("KIDFLUENCER EXPLOITATION STUDY - FULL ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df = load_all_videos()
    
    # Step 2: Compute embeddings
    print("\n[STEP 2] Computing embeddings...")
    embeddings = compute_embeddings(df)
    
    # Step 3: Build direction vector
    print("\n[STEP 3] Building exploitation direction vector...")
    direction = build_direction_vector()
    
    # Step 4: Compute drift scores
    print("\n[STEP 4] Computing drift scores...")
    drift_scores = compute_drift_scores(embeddings, direction)
    df['drift_score'] = drift_scores
    
    # Step 5: Identify viral hits
    print("\n[STEP 5] Identifying viral hits...")
    df['is_viral'] = identify_viral_hits(df)
    
    # Step 6: Channel-level statistics
    print("\n[STEP 6] Computing channel-level statistics...")
    channel_stats = compute_channel_level_stats(df)
    
    # Step 7: Temporal drift analysis
    print("\n[STEP 7] Temporal drift analysis...")
    quarterly = temporal_drift_analysis(df)
    
    # Step 8: DiD causal inference
    print("\n[STEP 8] Difference-in-Differences causal inference...")
    # Install statsmodels if needed
    try:
        import statsmodels
    except ImportError:
        print("  Installing statsmodels...")
        os.system("pip3 install statsmodels -q")
        import statsmodels
    
    did_df, did_model = run_did_analysis(df)
    
    # Save full dataset with drift scores
    print("\n[SAVING] Full dataset with drift scores...")
    save_cols = ['id', 'title', 'publishedAt', 'channel_short_name', 'channel_category',
                 'channel_title', 'viewCount', 'likeCount', 'commentCount',
                 'drift_score', 'is_viral']
    df[save_cols].to_csv(RESULTS_DIR / "full_analysis_results.csv", index=False)
    print(f"  Saved to {RESULTS_DIR / 'full_analysis_results.csv'}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  - title_embeddings.npy ({embeddings.shape})")
    print(f"  - direction_vector.npy")
    print(f"  - full_analysis_results.csv ({len(df):,} rows)")
    print(f"  - channel_drift_stats.csv")
    print(f"  - quarterly_drift_trends.csv")
    print(f"  - did_observations.csv")
    print(f"  - did_regression_summary.txt")


if __name__ == "__main__":
    main()
