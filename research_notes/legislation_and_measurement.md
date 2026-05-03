# Research Notes: Legislation & Measurement Frameworks for Kidfluencer Labor

## 1. France - Loi Studer (Law No. 2020-1266, October 19, 2020)

The world's first kidfluencer-specific legislation. Key provisions:

**Definition of when child's activity = work:**
- French law uses THREE cumulative factors: work performance, remuneration, and relationship of subordination
- The Studer Report acknowledged that "many situations do not meet the combination of these three conditions" because child filmed in daily life doesn't provide a service, some videos aren't monetized, and child doesn't receive instructions from a director
- Solution: The law LIKENS kidfluencers to children working in entertainment industries (film, TV, fashion models) regardless of whether formal employment relationship exists

**Two triggers for mandatory work authorization (Article 3-I):**
1. When cumulative DURATION or NUMBER of contents exceeds a threshold (to be set by decree)
2. When distribution provides INCOME (direct or indirect) above a threshold (to be set by decree)
- These are NOT cumulative - either trigger alone activates the law
- Key insight: "Parents may well choose not to monetize a video that goes viral, which does not rule out damage to the exposed minor"

**Administrative authority makes recommendations about:**
- Times, duration, hygiene and safety of conditions for making videos
- Risks, in particular psychological, associated with dissemination
- Legal requirements for normal school attendance
- Financial obligations

**Financial protections:**
- Revenue must be paid to Caisse des Dépôts et Consignations (public escrow)
- Managed until child reaches age of majority
- Penalty: €3,750 fine for giving child's funds directly to parents

**Right to be forgotten (Article 6):**
- Child can request deletion of content when they reach majority

**Platform obligations:**
- Platforms must detect unauthorized content featuring minors under 16
- Administrative authority can refer to judge for emergency measures

---

## 2. Illinois (Public Act 103-0556, effective July 1, 2024)

First US state to protect child influencers. Key provisions:

- Applies to "minors featured in vlogs"
- Requires vloggers to set aside portion of earnings in trust
- Based on percentage of content featuring the child
- Allows adults to take legal action against parents if featured as minors
- Extends existing child labor law framework

---

## 3. California (A.B. 1880, September 2024)

- Amends Family Code to include content creators in protections for minors in artistic employment
- Requires Coogan Trust Accounts
- Clarifies employer responsibilities

---

## 4. How to Operationalize "Commercialization" (from Literature)

| Indicator | Source | Measurable? |
|-----------|--------|-------------|
| Monetization status (YouTube Partner Program) | Channel metadata | Partially (inferred from ads) |
| Sponsored content disclosure (#ad, #sponsored) | Video description/title NLP | Yes |
| Product placement in video | CV/manual coding | Yes (LLM annotation) |
| Affiliate links in description | URL parsing | Yes |
| Merchandise links | Description parsing | Yes |
| Brand deal mentions | NLP on description | Yes |
| Multi-channel network (MCN) membership | Channel metadata | Partially |
| Cross-platform presence (merch stores, etc.) | External links | Yes |
| Revenue estimates (SocialBlade) | Third-party data | Yes |
| Number of channels operated | YouTube API | Yes |
| Production quality indicators | CV analysis | Yes |

**From Clark & Jno-Charles (2025):**
- Family production companies (e.g., Sunlight Entertainment LLC)
- Full-time creative staff (30+ for Ryan's World)
- Multiple YouTube channels (21 for Vlad & Niki)
- Brand deals, merchandise, licensing
- Annual earnings ($22-24M for Ryan's World)
- Cross-platform expansion

---

## 5. How to Operationalize "Child Labor Intensity" (from Literature)

| Proxy Metric | Source | How to Compute |
|-------------|--------|----------------|
| Upload frequency | Video timestamps | Videos per week/month |
| Video duration | Video metadata | Minutes per video |
| Total content minutes/month | Duration × frequency | Cumulative production |
| Estimated filming time | Duration × shooting ratio (5:1 to 400:1) | Clark 2025 method |
| Child as main subject | Thumbnail/title analysis | % videos with child protagonist |
| Weekend/holiday uploads | Timestamp analysis | % uploads on non-school days |
| Publishing consistency | Time series analysis | Gaps, streaks, regularity |
| Content escalation over time | Longitudinal analysis | Increasing frequency/duration |

**From Clark & Jno-Charles (2025):**
- "Filming time, estimated from final video time, is the best barometer of a child's work commitment"
- Shooting ratios: 5:1 to 400:1 (minutes shot vs. finished)
- Even 100 min content/month → ~1000 min shooting → 4.2 hrs/week minimum
- Does NOT include travel, preparation, waiting time
- Ryan's World: 5-6 days/week upload frequency
- At low frequencies → appears harmless play
- At high frequencies → "activity habituates and closer approximates labor"

---

## 6. Control Group Design Considerations

**What makes a good control group for this study?**

Option A: Adult entertainment/vlog channels (same genre, no children)
- Advantage: Controls for genre effects
- Disadvantage: Different audience demographics

Option B: Non-monetized kidfluencer channels (same content type, not commercial)
- Advantage: Isolates commercialization effect
- Disadvantage: Hard to find truly non-monetized channels at scale

Option C: Within-channel comparison (same channel, commercial vs non-commercial videos)
- Advantage: Controls for channel-level confounds
- Disadvantage: Channels may be uniformly commercial

**Best approach for our study:**
- Use WITHIN-SAMPLE variation: channels with high vs low commercialization signals
- Operationalize commercialization as a continuous variable (composite score)
- Compare high-commercial vs low-commercial channels on labor intensity metrics
- This avoids the need for external data collection while still testing the hypothesis

---

## 7. Key Pew Research (2019) Findings

- Videos featuring children under 13 averaged 3x more views (297,574 vs 97,081)
- Videos both aimed at children AND featuring a child: 4x more views (416,985 vs 96,416)
- Only 2% of videos featured children, but they dominated views
- Channels producing child content averaged 1.8M subscribers vs 1.2M for others
- 79% of videos featuring children were oriented toward general audience (not just kids)
- Key finding: Platform algorithmically rewards child-featuring content

---

## 8. Synthesis: Our Paper's Approach

**Core hypothesis:** Higher commercialization → more intensive child labor indicators

**Operationalization:**
- IV (Commercialization): Composite of sponsored_content + product_placement + affiliate_links + brand_mentions + cross_platform_count + channel_size
- DV (Child Labor Intensity): Composite of upload_frequency + video_duration + child_protagonist_rate + emotional_performance + weekend_uploads
- Controls: Channel age, country, content genre

**What makes this novel:**
1. First COMPUTATIONAL study to test commercialization → labor intensity link at scale
2. Uses NLP + CV + LLM annotation (multi-modal)
3. Grounded in legal frameworks (France, Illinois, UNCRC)
4. Within-sample design avoids external data collection issues
5. Provides empirical evidence for ongoing legislative debates

---

## 9. Control Group Design in Similar Studies

### Key Methodological Insights from Literature:

**Abidin (2017) - #familygoals:**
- Used ethnographically informed content analysis
- Compared TWO GROUPS of family influencers on social media
- Analyzed "anchor" content (creative) vs "filler" content (domestic routines)
- Key concept: "calibrated amateurism" — strategic presentation of raw/natural content
- Did NOT use non-family control group; compared within the kidfluencer ecosystem

**Choi (2023) - Brand Integration in Child-Targeted YouTube:**
- Content analysis of kid-friendly YouTube videos
- Coded for 3 types of implicit brand integration techniques
- Used systematic sampling of popular child-targeted videos
- Comparison was between videos WITH and WITHOUT brand integration

**Liddle & Sherrill (2023) - ACM SIGDOC Methodology Paper:**
- Sampled 300 YouTube videos across 100 channels
- Key insight: channel-level vs video-level analysis matters
- "Popular channels with thousands or millions of subscribers are able to amplify videos with greater rhetorical velocity"
- "A small channel can adopt excellent practices and still not achieve wide circulation"
- Solution: Statistical analysis to identify variance WITHIN individual channels
- Used YouTube Channel Crawler for consistent sampling
- Challenges: personalized search, geolocation differences, API limitations

**Pew Research (2019):**
- Compared videos featuring children vs NOT featuring children
- Used random sample from popular channels
- Key finding: child-featuring videos get 3x more views
- This IS a comparison design but at video level, not channel level

### Recommended Control Group Design for Our Study:

**Option 1: Within-Sample Comparison (PREFERRED)**
- Split our 115 kidfluencer channels into HIGH vs LOW commercialization groups
- Use median split or tertile/quartile grouping
- Compare labor intensity metrics across groups
- Advantage: Same population, same data collection method, no confounds from different genres

**Option 2: Matched Adult Creator Comparison**
- Collect adult entertainment/vlog channels matched on:
  - Subscriber count (within same order of magnitude)
  - Upload frequency
  - Country/language
  - Channel age
- Compare content strategies and labor indicators
- Advantage: Can show kidfluencer-specific effects
- Disadvantage: Genre differences confound results

**Option 3: Non-Commercial Kidfluencer Channels**
- Find channels featuring children that are NOT monetized
- Compare with commercial kidfluencer channels
- Advantage: Isolates commercialization effect perfectly
- Disadvantage: Hard to find truly non-monetized channels at scale; survivorship bias

### My Recommendation:
Use **Option 1 as primary analysis** (within-sample, continuous IV) with **Option 2 as robustness check** (if we can collect 30-50 matched adult channels). This gives us:
1. Main finding: Within kidfluencers, commercialization predicts more labor
2. Robustness: Kidfluencers show different patterns than adult creators
3. No need for perfect matching (continuous regression handles this)

---

## 10. Summary: Revised Paper Framework

**Title:** "From Play to Labor: How Commercialization Drives Content Escalation in Kidfluencer Channels"

**RQ1:** Does higher commercialization predict greater child labor intensity in kidfluencer channels?
**RQ2:** What specific content strategies mediate the relationship between commercialization and engagement?
**RQ3:** How do kidfluencer labor patterns compare to adult entertainment creators?

**Method:**
1. Operationalize commercialization (composite: sponsored content + brand mentions + affiliate links + cross-platform + production quality)
2. Operationalize child labor intensity (composite: upload frequency + video duration + child_protagonist rate + emotional performance + weekend uploads)
3. Run channel-level regression: Labor_Intensity ~ Commercialization + Controls
4. Run mediation analysis: Commercialization → Content_Strategy → Views
5. (Optional) Compare with matched adult creator sample

**Theoretical Grounding:**
- Clark & Jno-Charles (2025): Ethics of care framework for kidfluencer labor
- Abidin (2017): Calibrated amateurism and justifying digital labor
- France Loi Studer (2020): Legal definition of when child activity = work
- Illinois PA 103-0556 (2024): Revenue-based child protection
- UNCRC General Comment 25 (2021): Children's rights in digital environment
- Pew Research (2019): Platform rewards for child-featuring content
