# Kidfluencer Data Feasibility Analysis

## 1. YouTube API Test Results

### Successfully Fetched Channels:
| Channel | Subscribers | Videos | Total Views | Joined |
|---------|------------|--------|-------------|--------|
| Ryan's World | 40.3M | 3,684 | 63.3B | 2015-03 |
| The ACE Family | 18M | 713 | 4.6B | 2016-01 |
| Family Fun Pack | 10.5M | 3,449 | 15.9B | 2011-10 |
| The LaBrant Fam | 12.7M | 651 | 4.7B | 2012-08 |
| JesssFam | 2.46M | 2,566 | 1.7B | 2009-09 |

### Available Data Fields:
- **Channel level:** name, description, subscribers, total views, country, join date
- **Video level:** title, views, duration (seconds), publish time
- **Per batch:** 30 videos per API call (latest videos)
- **Missing:** comments text, likes/dislikes per video, video descriptions, tags

### Limitations:
- Some channels not found with @ handles (need channel IDs)
- No direct comment text retrieval from this API
- Need to paginate to get full video history

## 2. TikTok API Test Results
- Search API works but returns data in `item_list` (not `data`)
- Available fields: description, author info, follower count, play/like/comment/share counts, duration
- Kidfluencer content is findable

## 3. Notable Cases & Documentaries

### Netflix Documentary: "Bad Influence: The Dark Side of Kidfluencing" (April 2025)
- About Piper Rockelle and "The Squad"
- Mother Tiffany Smith orchestrated child influencer group
- Teens and parents reveal disturbing accounts of abuse and exploitation

### Hulu/Freeform: "Born to Be Viral: The Real Lives of Kidfluencers" (June 2025)
- 5-year longitudinal documentary following 3 families
- ABC News Studios production

### Notable Exploitation Cases:
1. **Ruby Franke / 8 Passengers** — Convicted of felony child abuse, sentenced to 4 consecutive 1-15 year terms
2. **DaddyOFive** — Channel removed for child abuse content
3. **Piper Rockelle / The Squad** — Netflix documentary about exploitation by mother/manager
4. **Wren Eleanor** — TikTok controversy about sexualization of toddler content
5. **The LaBrant Fam** — Known for clickbait thumbnails featuring children in distress

## 4. Key Channels for Dataset (Seed List)

### High-profile kidfluencer channels:
- Ryan's World (40.3M subs) — toy reviews, unboxing
- Like Nastya (~100M+ subs) — need correct URL
- Kids Diana Show — massive channel
- Family Fun Pack (10.5M subs) — family vlog
- The ACE Family (18M subs) — family vlog, controversial
- The LaBrant Fam (12.7M subs) — family vlog, clickbait
- JesssFam (2.46M subs) — teen mom vlog
- 8 Passengers (removed/archived) — convicted case

### Data Collection Strategy:
1. Start with ~50-100 family vlog / kidfluencer channels
2. For each channel, collect ALL video metadata (title, views, duration, publish date)
3. Use YouTube transcript API to get video captions/transcripts
4. Collect comment text for engagement analysis
5. Track longitudinal patterns (content evolution over time)

## 5. Feasibility Assessment

### What's EASY to get:
- Video titles, views, durations, publish dates ✓
- Channel metadata ✓
- Video-level engagement metrics ✓

### What's HARDER but possible:
- Video transcripts (via YouTube transcript API or whisper)
- Comment text (need separate API calls)
- Thumbnail images (available via URL)

### What's NOT available via API:
- Whether children appear in video (need CV model)
- Emotional state of children (need multimodal analysis)
- Whether content is "exploitative" (need annotation framework)

### Ethical Considerations:
- All data is publicly available
- No need to identify individual children by name (can use channel-level analysis)
- Focus on content patterns, not individual children
- IRB may be needed depending on university requirements
