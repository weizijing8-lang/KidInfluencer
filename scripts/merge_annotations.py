#!/usr/bin/env python3
"""
Merge all LLM annotation CSV files into a single dataset.
"""
import pandas as pd
import os
import glob

EXTRACT_DIR = '/home/ubuntu/output_file_extracted'
OUTPUT_PATH = '/home/ubuntu/KidInfluencer/data/annotations_merged.csv'

# Find all CSV files
csv_files = glob.glob(os.path.join(EXTRACT_DIR, '*.csv'))
print(f"Found {len(csv_files)} annotation files")

# Read and merge
all_dfs = []
errors = 0
for f in sorted(csv_files):
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            all_dfs.append(df)
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(f)}: {e}")
        errors += 1

if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    # Standardize column names
    col_map = {}
    for col in merged.columns:
        col_map[col] = col.strip().lower().replace(' ', '_')
    merged = merged.rename(columns=col_map)
    
    print(f"\nMerged dataset:")
    print(f"  Rows: {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")
    
    # Check for expected columns
    expected = ['video_id', 'content_type', 'emotional_manipulation', 'commercial_signals', 
                'child_role', 'privacy_concern', 'clickbait_level']
    found = [c for c in expected if c in merged.columns]
    missing = [c for c in expected if c not in merged.columns]
    print(f"  Expected columns found: {found}")
    if missing:
        print(f"  Missing columns: {missing}")
    
    # Basic stats
    if 'content_type' in merged.columns:
        print(f"\nContent Type Distribution:")
        print(merged['content_type'].value_counts().head(10).to_string())
    
    if 'emotional_manipulation' in merged.columns:
        print(f"\nEmotional Manipulation Distribution:")
        print(merged['emotional_manipulation'].value_counts().to_string())
    
    if 'commercial_signals' in merged.columns:
        print(f"\nCommercial Signals Distribution:")
        print(merged['commercial_signals'].value_counts().to_string())
    
    if 'child_role' in merged.columns:
        print(f"\nChild Role Distribution:")
        print(merged['child_role'].value_counts().to_string())
    
    if 'privacy_concern' in merged.columns:
        print(f"\nPrivacy Concern Distribution:")
        print(merged['privacy_concern'].value_counts().to_string())
    
    if 'clickbait_level' in merged.columns:
        print(f"\nClickbait Level Distribution:")
        print(merged['clickbait_level'].value_counts().to_string())
    
    # Save
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged annotations → {OUTPUT_PATH}")
else:
    print("ERROR: No valid DataFrames to merge!")
