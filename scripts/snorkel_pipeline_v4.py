#!/usr/bin/env python3
"""
Snorkel Pipeline v4: Combines LLM text classifications with GPT-4 Vision thumbnail analysis.

Approach: Weighted combination of two signal sources:
  1. LLM text-based classification (title only) - binary {0,1}
  2. GPT-4 Vision classification (title + thumbnail + description) - continuous [0,1]

Vision is given higher weight based on validation (κ=0.617 vs κ=0.309).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

DATA_DIR = Path("/home/ubuntu/KidInfluencer/data")
OUTPUT_DIR = Path("/home/ubuntu/KidInfluencer/analysis_discovery/snorkel_results_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    'performative_labor', 'emotional_bait', 'narrative_conflict',
    'challenge_format', 'commercial_content', 'privacy_violation'
]

# Weights based on validation (κ_vision=0.617, κ_llm=0.309)
W_VISION = 0.67
W_LLM = 0.33


def load_data():
    llm = pd.read_csv(DATA_DIR / "llm_classifications_v2.csv")
    vision = pd.read_csv(DATA_DIR / "vision_results" / "vision_classifications.csv")
    sample = pd.read_csv(DATA_DIR / "stratified_sample_v2.csv")

    merged = llm.merge(
        vision[['video_id'] + DIMENSIONS + ['overall_exploitative']],
        left_on='id', right_on='video_id', how='left',
        suffixes=('_llm', '_vision')
    )
    merged = merged.merge(
        sample[['id', 'publishedAt', 'channelId']],
        on='id', how='left'
    )
    merged['viewCount'] = pd.to_numeric(merged['viewCount'], errors='coerce')
    print(f"Loaded {len(merged)} videos, {merged['channel_short_name'].nunique()} channels")
    print(f"  Vision coverage: {merged['overall_exploitative'].notna().sum()}/{len(merged)}")
    return merged


def compute_scores(df):
    for dim in DIMENSIONS:
        llm_col = f"{dim}_llm"
        vis_col = f"{dim}_vision"
        if llm_col in df.columns and vis_col in df.columns:
            llm_v = df[llm_col].fillna(0).astype(float)
            vis_v = df[vis_col].fillna(0.5)
            df[f"{dim}_combined"] = W_LLM * llm_v + W_VISION * vis_v

    # Overall score
    llm_dims = [f"{d}_llm" for d in DIMENSIONS if f"{d}_llm" in df.columns]
    llm_overall = df[llm_dims].fillna(0).mean(axis=1)
    vis_overall = df['overall_exploitative'].fillna(0.5)
    df['exploitation_score_v4'] = W_LLM * llm_overall + W_VISION * vis_overall

    df['is_exploitative_v4'] = (df['exploitation_score_v4'] >= 0.5).astype(int)

    n_e = df['is_exploitative_v4'].sum()
    print(f"\nClassification: {n_e} exploitative ({100*n_e/len(df):.1f}%), "
          f"{len(df)-n_e} clean ({100*(len(df)-n_e)/len(df):.1f}%)")
    print(f"Score: mean={df['exploitation_score_v4'].mean():.3f}, "
          f"median={df['exploitation_score_v4'].median():.3f}, "
          f"std={df['exploitation_score_v4'].std():.3f}")
    return df


def engagement_analysis(df):
    valid = df.dropna(subset=['viewCount', 'exploitation_score_v4']).copy()
    valid = valid[valid['viewCount'] > 0]
    valid['log_views'] = np.log1p(valid['viewCount'])

    print(f"\n{'='*60}")
    print("OVERALL ENGAGEMENT PREMIUM (Within-Channel)")
    print(f"{'='*60}")

    # Channel-level premiums
    ch_rows = []
    for ch, g in valid.groupby('channel_short_name'):
        ex = g[g['is_exploitative_v4'] == 1]
        cl = g[g['is_exploitative_v4'] == 0]
        if len(ex) >= 3 and len(cl) >= 3:
            prem = ex['log_views'].mean() - cl['log_views'].mean()
            ch_rows.append({
                'channel': ch,
                'n_exploit': len(ex), 'n_clean': len(cl),
                'mean_views_exploit': ex['viewCount'].mean(),
                'mean_views_clean': cl['viewCount'].mean(),
                'log_premium': prem,
                'view_ratio': ex['viewCount'].mean() / cl['viewCount'].mean()
            })
    ch_df = pd.DataFrame(ch_rows)

    if len(ch_df) > 0:
        stat, p = stats.wilcoxon(ch_df['log_premium'])
        n_pos = (ch_df['log_premium'] > 0).sum()
        print(f"  Channels: {len(ch_df)}")
        print(f"  Mean log premium: {ch_df['log_premium'].mean():.4f}")
        print(f"  Median log premium: {ch_df['log_premium'].median():.4f}")
        print(f"  Median view ratio: {ch_df['view_ratio'].median():.2f}x")
        print(f"  Wilcoxon p = {p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")
        print(f"  Positive premium: {n_pos}/{len(ch_df)} ({100*n_pos/len(ch_df):.1f}%)")

    # Per-dimension
    print(f"\n{'='*60}")
    print("PER-DIMENSION ENGAGEMENT PREMIUM")
    print(f"{'='*60}")

    dim_rows = []
    for dim in DIMENSIONS:
        col = f"{dim}_combined"
        if col not in valid.columns:
            continue
        boosts = []
        for ch, g in valid.groupby('channel_short_name'):
            hi = g[g[col] >= 0.5]
            lo = g[g[col] < 0.5]
            if len(hi) >= 3 and len(lo) >= 3:
                boosts.append(hi['log_views'].mean() - lo['log_views'].mean())
        if len(boosts) >= 5:
            s, p = stats.wilcoxon(boosts)
            dim_rows.append({
                'dimension': dim,
                'n_channels': len(boosts),
                'mean_premium': np.mean(boosts),
                'median_premium': np.median(boosts),
                'p_value': p
            })
            print(f"  {dim}: n={len(boosts)}, mean={np.mean(boosts):.4f}, "
                  f"p={p:.6f} {'*' if p<0.05 else 'n.s.'}")

    # FDR correction
    if dim_rows:
        ps = [d['p_value'] for d in dim_rows]
        reject, p_corr, _, _ = multipletests(ps, method='fdr_bh')
        print(f"\n{'='*60}")
        print("FDR-CORRECTED RESULTS")
        print(f"{'='*60}")
        for i, d in enumerate(dim_rows):
            d['p_corrected'] = p_corr[i]
            d['significant_fdr'] = bool(reject[i])
            print(f"  {d['dimension']}: p_raw={d['p_value']:.6f}, "
                  f"p_fdr={p_corr[i]:.6f}, sig={reject[i]}")

    # Same-year robustness check
    print(f"\n{'='*60}")
    print("ROBUSTNESS: SAME-YEAR WITHIN-CHANNEL")
    print(f"{'='*60}")
    if 'publishedAt' in valid.columns:
        valid['year'] = pd.to_datetime(valid['publishedAt'], errors='coerce').dt.year
        year_boosts = []
        for (ch, yr), g in valid.groupby(['channel_short_name', 'year']):
            ex = g[g['is_exploitative_v4'] == 1]
            cl = g[g['is_exploitative_v4'] == 0]
            if len(ex) >= 2 and len(cl) >= 2:
                year_boosts.append(ex['log_views'].mean() - cl['log_views'].mean())
        if len(year_boosts) >= 5:
            s, p = stats.wilcoxon(year_boosts)
            n_pos = sum(1 for b in year_boosts if b > 0)
            print(f"  Channel-year pairs: {len(year_boosts)}")
            print(f"  Mean premium: {np.mean(year_boosts):.4f}")
            print(f"  Wilcoxon p = {p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")
            print(f"  Positive: {n_pos}/{len(year_boosts)}")

    # Spearman correlation
    rho, p_rho = stats.spearmanr(valid['exploitation_score_v4'], valid['log_views'])
    print(f"\nSpearman correlation (score vs log_views): ρ={rho:.4f}, p={p_rho:.4e}")

    return ch_df, dim_rows


def save_results(df, ch_df, dim_rows):
    df.to_csv(OUTPUT_DIR / "classified_videos_v4.csv", index=False)
    if len(ch_df) > 0:
        ch_df.to_csv(OUTPUT_DIR / "channel_premiums_v4.csv", index=False)
    if dim_rows:
        pd.DataFrame(dim_rows).to_csv(OUTPUT_DIR / "dimension_results_v4.csv", index=False)

    summary = {
        'total_videos': len(df),
        'exploitative': int(df['is_exploitative_v4'].sum()),
        'clean': int(len(df) - df['is_exploitative_v4'].sum()),
        'mean_score': float(df['exploitation_score_v4'].mean()),
        'channels': int(df['channel_short_name'].nunique()),
        'vision_coverage': int(df['overall_exploitative'].notna().sum()),
    }
    with open(OUTPUT_DIR / "summary_v4.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("PIPELINE v4: Vision + LLM Combined Classification")
    print("=" * 60)
    df = load_data()
    df = compute_scores(df)
    ch_df, dim_rows = engagement_analysis(df)
    save_results(df, ch_df, dim_rows)
    print("\nDone!")


if __name__ == "__main__":
    main()
