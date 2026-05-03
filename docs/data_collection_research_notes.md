# Data Collection Research Notes

## Available APIs (Manus Data API)

### 1. Youtube/get_channel_details
- Channel ID, title, description, custom URL, handle
- Country, joined date
- Stats: subscribers, total videos, total views
- Avatar, banner, badges, links, keywords
- Available regions

### 2. Youtube/get_channel_videos
- Filter types: videos_latest, streams_latest, shorts_latest, live_now
- Per video: title, videoId, publishedTimeText, lengthSeconds, views, isLiveNow, badges, thumbnails
- Supports pagination via cursor

### 3. Youtube/search
- Search by keyword
- Returns videos, channels, playlists
- Per video: title, videoId, channelTitle, publishedTimeText, lengthText, viewCountText, descriptionSnippet
- Supports pagination and language/country filters

## Channel Discovery Strategies

### Seed Sources:
1. **SocialBlade** top made-for-kids channels (top 100 by subscribers, filtered by made_for_kids=true)
   - Cocomelon (201M), Kids Diana Show, Vlad and Niki, Like Nastya, etc.
   - But many are animation/corporate, not family vlogs
2. **Feedspot** "100 Family YouTubers" list (feedspot.com/family_youtube_channels)
3. **Academic papers**: Kissgen et al. "Child Influencers on YouTube: From Collection to Overlapping Community Detection"
4. **Reddit** r/YTVloggerFamilies - community-curated lists
5. **YouTube search** with keywords: "family vlog", "day in our life family", "kids routine", "mom vlog"
6. **Snowball sampling**: Use "related channels" from seed channels

### Key Distinction: Kidfluencer vs Kids Content
- **Kidfluencer/Family vlog**: Real children featured as main content (our target)
- **Kids content (animation)**: Cocomelon, BabyBus, Peppa Pig (NOT our target)
- Need filtering criteria to separate these

## Known Exploitation Cases (Ground Truth Positive Set)

### Confirmed Legal/Media Cases:
1. **8 Passengers (Ruby Franke)** - 2.3M subs, arrested 2023, sentenced to prison Feb 2024 for child abuse
2. **DaddyOFive (Mike Martin)** - Lost custody of 2 children May 2017, channel deleted
3. **Piper Rockelle** - Mother sued by 11 former members of "Piper's Squad" for exploitation
4. **FamilyOFive** - Rebranded DaddyOFive, eventually removed
5. **Fantastic Adventures (Machelle Hobson)** - Arrested 2019 for child abuse, pepper-sprayed kids to force them to perform
6. **Jordan Cheyenne** - Caught coaching son to cry on camera, channel deleted 2021
7. **Austin McBroom (ACE Family)** - Multiple lawsuits, exploitation allegations
8. **Myka Stauffer** - Rehomed adopted son after monetizing adoption journey

## Data Fields Needed

### Channel-level:
- Channel ID, name, description, country, join date
- Subscriber count, total views, total videos
- Keywords/tags, links (cross-platform presence)
- Made-for-kids flag (if detectable)

### Video-level (per channel, sample or full):
- Video ID, title, description
- Duration (seconds), publish date
- View count, like count, comment count
- Comments enabled/disabled
- Tags, category
- Thumbnail URL (for potential visual analysis)

### Derived metrics:
- Upload frequency (videos per week/month)
- Average video duration
- Engagement rate (likes+comments / views)
- Commercial indicators in titles/descriptions (sponsor, ad, #ad, brand names)
- Emotional manipulation indicators in titles (crying, punishment, surprise, prank)
- Cross-platform count (Instagram, TikTok links in channel description)
- Comment disable rate

## Kissgen et al. (2023) Dataset
- "Child Influencers on YouTube: From Collection to Overlapping Community Detection"
- Network: 72,577 channels, 2,025,879 edges, 388 confirmed child influencers
- Used automatic scripts targeting child influencers on YouTube
- Stored in ArangoDB graph database
- Applied overlapping community detection algorithms
- Found: family channels and single child influencer channels form big communities, with a divide between the two
- Collection scripts, software, and dataset are freely available (open source WebOCD framework)
- This is a KEY resource — 388 confirmed child influencer channels could be our seed list
