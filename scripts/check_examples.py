import pandas as pd

df = pd.read_csv('analysis_discovery/labor_classification_full.csv')

# 1. "stole" / "took my" titles
print("=" * 80)
print("1. VIDEOS WITH 'stole/took my/steal' IN TITLE")
print("=" * 80)
mask = df['title'].str.contains('stole|took my|steal|who took', case=False, na=False)
results = df[mask].sort_values('viewCount', ascending=False).head(30)
print(f'Found {mask.sum()} total videos\n')
for _, row in results.iterrows():
    views = int(row['viewCount']) if pd.notna(row['viewCount']) else 0
    print(f'  labor={row["labor_type"]:12s} | exploit={str(row["emotional_exploitation"]):5s} | format={row["content_format"]:12s} | views={views:>12,}')
    print(f'    channel={row["channel_short_name"]:20s} | {row["title"][:75]}')
    print()

# 2. Cocomelon / music_dance examples
print("=" * 80)
print("2. MUSIC/DANCE FORMAT EXAMPLES (top by views)")
print("=" * 80)
music = df[df['content_format'] == 'music_dance'].sort_values('viewCount', ascending=False).head(20)
for _, row in music.iterrows():
    views = int(row['viewCount']) if pd.notna(row['viewCount']) else 0
    print(f'  labor={row["labor_type"]:12s} | exploit={str(row["emotional_exploitation"]):5s} | views={views:>12,}')
    print(f'    channel={row["channel_short_name"]:20s} | {row["title"][:75]}')
    print()

# 3. Check what Cocomelon videos look like
print("=" * 80)
print("3. COCOMELON SAMPLE (first 15)")
print("=" * 80)
coco = df[df['channel_short_name'] == 'cocomelon'].sort_values('viewCount', ascending=False).head(15)
for _, row in coco.iterrows():
    views = int(row['viewCount']) if pd.notna(row['viewCount']) else 0
    print(f'  labor={row["labor_type"]:12s} | format={row["content_format"]:12s} | views={views:>12,} | {row["title"][:65]}')

# 4. Summary: how many music_dance are actually "kids dancing" vs animated
print("\n" + "=" * 80)
print("4. MUSIC/DANCE BY CHANNEL")
print("=" * 80)
music_all = df[df['content_format'] == 'music_dance']
ch_counts = music_all.groupby('channel_short_name').agg(
    n=('id', 'count'),
    perf_rate=('labor_type', lambda x: (x == 'performative').mean()),
    med_views=('viewCount', 'median')
).sort_values('n', ascending=False)
print(ch_counts.to_string())
