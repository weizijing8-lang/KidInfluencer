# Literature Review: How Other Papers Handle View Count / Video Age / Algorithm Proxy

## 1. Huszar et al. 2022 (PNAS) - "Algorithmic Amplification of Politics on Twitter"

**Gold standard approach:** They had access to Twitter's internal A/B test.
- Treatment group: algorithmic timeline (personalized)
- Control group: reverse-chronological timeline (no algorithm)
- "Amplification ratio" = (impressions in treatment) / (impressions in control) - 1
- Normalized so that 0% = equal reach in both groups

**Key insight:** They don't use view count as a proxy for algorithmic behavior. They directly compare algorithmic vs non-algorithmic feeds. This is the ideal but requires platform cooperation.

**Relevance to us:** We can't do this. We only have observational data. We should explicitly acknowledge this gap.

## 2. Ribeiro et al. 2020 (FAT*) - "Auditing Radicalization Pathways on YouTube"

**Approach:** 330,925 videos from 349 channels.
- They analyze user migration patterns (comments across channels over time)
- For engagement, they use raw view counts but focus on RELATIVE comparisons within communities
- They normalize channel recommendation graphs (outgoing edges sum to 1)
- They do NOT claim views = algorithmic amplification directly
- Their main contribution is about recommendation pathways, not view count analysis

**Key insight:** They avoid the "views = algorithm" conflation by studying recommendations directly (using the YouTube recommendation sidebar).

## 3. Papadamou et al. 2020 (ICWSM) - "Disturbed YouTube for Kids"

**Approach:** 
- They classify videos as appropriate/disturbing using a deep learning classifier
- They compare view counts between appropriate and disturbing videos
- They report CDF of views: "suitable videos have more views than disturbing videos"
- They do NOT control for video age
- They do NOT claim this is algorithmic - they frame it as "reach" or "exposure"

**Key insight:** They use views as a measure of "reach" without claiming it's purely algorithmic. This is a weaker but defensible claim.

## 4. Common Approaches in YouTube Research

### Option A: Log(views) as dependent variable
- Most common in observational studies
- Simple, interpretable
- Problem: confounded by video age, channel size, upload time

### Option B: Views per day (views / days_since_publish)
- Controls for video age mechanically
- Used in some marketing/business research
- Problem: assumes linear view accumulation (not true - most views come in first 48 hours)

### Option C: Within-channel z-score
- Normalize each video's views by channel mean and SD
- Controls for channel-level differences
- Used in Eom et al. 2025 (Korean beauty channels) with fixed-effects panel data

### Option D: Log(views) with video age as covariate
- Include log(days_since_publish) as a control variable in regression
- Most flexible approach
- Problem: relationship between age and views is non-linear

### Option E: Relative performance (video views / channel median)
- Each video expressed as ratio to channel median
- Our current "within-channel" approach
- Advantage: intuitive, controls for channel size
- Disadvantage: doesn't control for temporal trends within channel

### Option F: Mixed-effects model with channel random effect + time fixed effect
- Most sophisticated approach
- Channel as random effect, year/quarter as fixed effect
- Used in panel data econometrics

## 5. Key Distinction: "Views" vs "Algorithmic Amplification"

**Critical point from Huszar et al.:** Views ≠ algorithmic amplification.

Views are a product of:
1. Algorithmic recommendation (suggested videos, home feed)
2. Search (user actively searching)
3. External traffic (social media shares, embeds)
4. Subscriber notifications
5. Browse features (trending, etc.)

Only #1 is "algorithmic amplification." We cannot separate these.

**Defensible framing:** "Videos with exploitative characteristics receive higher engagement (views), which may reflect algorithmic amplification, audience preferences, or both."

## 6. Recommendation for Our Paper

**Best approach for us:**
1. Use log(views) as DV with within-channel comparison (our current approach)
2. Add log(days_since_publish) as a covariate - BUT interpret carefully
3. Report BOTH with and without the age control as robustness
4. Frame as "engagement premium" not "algorithmic amplification"
5. Explicitly discuss the limitation that views ≠ algorithm in a dedicated paragraph
6. Consider views_per_day as a robustness check (knowing it's imperfect)

**The key argument:** Even if we can't prove it's the algorithm, the INCENTIVE STRUCTURE still exists. If exploitative videos get more views (regardless of mechanism), creators are incentivized to produce more of them. This is the policy-relevant finding regardless of whether it's algorithm-driven or audience-driven.
