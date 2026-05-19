"""
Filter out adult-centric channels from the dataset.
Keep only channels where children are the primary subjects/creators.
"""
import pandas as pd

# Channels where the PRIMARY content creator/subject is an ADULT, not a child.
# These channels may occasionally feature children but are not "kidfluencer" channels.
adult_centric_channels = [
    # Adult YouTubers / creators
    'brentrivera',       # Adult comedy creator
    'crazygorilla',      # Adult comedy/sketch channel
    'blippi',            # Adult educational entertainer (character played by adult)
    'itsyeboi',          # Adult challenge/stunt creator
    'jordanmatter',      # Adult photographer (features kids in some videos)
    'piersonwodzynski',  # Adult content creator
    'rebeccazamolo',     # Adult challenge/game creator
    'ronaldomg',         # Teen/adult gaming creator
    'itsrucka',          # Adult comedy/music creator
    'jordynjones',       # Teen/adult dancer/singer (aged out)
    'gavinmagnus',       # Teen singer (aged out of "kid" category)
    
    # Family vlog channels where PARENTS are the main focus
    'itsjudyslife',      # Adult mom vlogger
    'jesssfam',          # Adult mom vlogger
    'thebramfam',        # Adult couple vlog, kids secondary
    'thedashleys',       # Adult couple vlog
    'samandnia',         # Adult couple vlog
    'meetthemillers',    # Adult family vlog, parents primary
    'family5vlogs',      # Adult parents primary focus
    'yawivlogs',         # Adult vlogger
    'bonniehoellein',    # Adult mom creator
    
    # Animated/educational channels (no real children)
    'babybus',           # Animated content, no real children
    'kidssongs',         # Music/animated, no real child creators
    
    # Adult content creators who do challenges
    'salishmatter',      # Adult (Jordan Matter's channel extension)
]

# Load the classified dataset
df = pd.read_csv('analysis_discovery/snorkel_proper/classified_videos_ws.csv')
print(f"Before filtering: {len(df)} videos, {df['channel_short_name'].nunique()} channels")

# Filter
df_filtered = df[~df['channel_short_name'].isin(adult_centric_channels)]
removed = df[df['channel_short_name'].isin(adult_centric_channels)]

print(f"Removed: {len(removed)} videos from {removed['channel_short_name'].nunique()} adult-centric channels")
print(f"After filtering: {len(df_filtered)} videos, {df_filtered['channel_short_name'].nunique()} channels")
print()

# Show what was removed
print("Removed channels:")
for ch in sorted(adult_centric_channels):
    n = len(removed[removed['channel_short_name'] == ch])
    if n > 0:
        print(f"  - {ch}: {n} videos removed")

# Show what remains (kid-centric channels)
print(f"\nRemaining kid-centric channels ({df_filtered['channel_short_name'].nunique()}):")
remaining = df_filtered.groupby('channel_short_name').size().sort_values(ascending=False)
for ch, n in remaining.items():
    print(f"  - {ch}: {n} videos")

# Save filtered dataset
df_filtered.to_csv('analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv', index=False)
print(f"\nFiltered dataset saved: analysis_discovery/snorkel_proper/classified_videos_ws_filtered.csv")

# Also save the channel classification for reference
channel_class = pd.DataFrame({
    'channel': sorted(df['channel_short_name'].unique()),
    'classification': ['adult_centric' if ch in adult_centric_channels else 'kid_centric' 
                       for ch in sorted(df['channel_short_name'].unique())]
})
channel_class.to_csv('analysis_discovery/snorkel_proper/channel_classification.csv', index=False)
print(f"Channel classification saved: analysis_discovery/snorkel_proper/channel_classification.csv")
