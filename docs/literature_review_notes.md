# Literature Review Notes

## Paper 1: Papadamou et al. (2020) - "Disturbed YouTube for Kids" (ICWSM 2020)

**Key methodology:**
- Three phases: (1) collect & manually review toddler-oriented videos, (2) characterize inappropriate videos, (3) build classifier
- Extended definition from "toddler" to any child aged 1-5 years
- Collected both Elsagate-related and other child-related videos, plus random videos
- Manual labeling by researchers as ground truth
- Built classifier with 84.3% accuracy to detect inappropriate content targeting toddlers
- Used classifier to show children likely encounter disturbing videos through random browsing
- Key finding: YouTube's counter-measures are ineffective at timely detection

**Features used (need to check):** likely video metadata, comments, thumbnails, possibly audio/visual features

**Ground truth:** Manual annotation by researchers - binary (appropriate/inappropriate)

**What we can learn:**
- They had manual annotation as ground truth - we need something similar
- They focused on VIDEO CONTENT, not just metadata
- The classifier approach is well-established at ICWSM
- They showed the PATHWAY to harmful content (recommendation chains), not just detection


### Detailed Methodology (Papadamou et al. 2020):

**Data Collection:**
- 12,097 seed videos + 844K recommended videos (total ~856K)
- 4 crawling strategies: Elsagate-related (1,074 seed → 232K rec), Other child-related (2,342 → 152K), Random (8,391 → 473K), Popular (500 → 10K)
- For each video: title, description, thumbnail, tags, view/like/dislike stats

**Manual Annotation:**
- 5K videos manually reviewed by 3 annotators
- Each video inspected: video content, title, thumbnail, tags
- Labels: Suitable, Disturbing, Restricted, Irrelevant
- Final ground truth: 1,513 suitable, 929 disturbing, 419 restricted, 1,936 irrelevant
- 76 undergraduate students + 2 authors did annotation
- Key: they WATCHED the videos, not just read titles

**Key insight for us:** Their ground truth came from actually watching videos. Title-only analysis is fundamentally limited. But they only needed to manually label ~5K videos, then trained a classifier on those to scale to 856K.

---


### Annotation Details (Papadamou et al.):

**Inter-annotator agreement:** Fleiss κ = 0.60 (moderate), experienced annotators κ = 0.80 (substantial)

**Key finding about titles vs content:** "These results reveal that disturbing videos on YouTube refer to seemingly 'innocent' cartoon videos in their title, but in reality the content of the video is likely to be either restricted or disturbing. Note that we find these terms in suitable videos too. This demonstrates that it is quite hard to distinguish suitable from disturbing videos by only inspecting their titles."

**This directly validates our concern** — title-only analysis is fundamentally limited. The paper explicitly says titles are unreliable for detecting disturbing content.

**Category breakdown (Table 3):** Shows that Entertainment (EN) and Film & Animation (F&A) are the dominant categories for both suitable and disturbing videos. Tags overlap significantly between suitable and disturbing.

---


### Classifier Architecture (Papadamou et al.):

**Features used (4 branches):**
1. **Title** — embedding layer → LSTM (RNN), max 21 words, vocab 12,023
2. **Tags** — separate embedding → LSTM, max 78 tags, vocab 40,096
3. **Thumbnail** — Inception-v3 CNN (pretrained ImageNet), scaled to 299x299, outputs 2,048-dim vector
4. **Statistics + Style** — views, likes, dislikes, comments + style features (duration, # bad words, # child-related words, # emoticons, Jaccard similarity, etc.)

**Style Features (Table 4):**
- Video-related: category, duration
- Statistics: ratio of likes to dislikes
- Title/description: length, ratio desc/title, Jaccard sim, # emoticons, # bad words, # child-related words
- Tags: # tags, # bad words in tags, # child-related words, Jaccard sim of tags & title

**Model:** Fusing Network — 4 branches merged → 2-layer dense → 512 units → Dropout 0.5 → softmax (4 classes)
**Accuracy:** 84.3% (binary suitable/disturbing)
**Training:** Keras/TensorFlow, 5-fold stratified cross-validation

**KEY TAKEAWAY:** They used MULTIMODAL features (title + tags + thumbnail + stats). Title alone is not enough. The thumbnail (visual) branch was critical for detecting disturbing content that disguises itself with innocent titles.

---


## Paper 2: SEPS - Co-commenters as clues (Amure & Agarwal, 2025)
**Venue**: Social Network Analysis and Mining, 2025
**Data**: 97 channels, 702,160 videos, 12.5M commenters, 123.9M comments (Indo-Pacific region)

### Key Methodology:
- **Co-commenter network**: Build bipartite graph of channels ↔ commenters
- If two channels share many commenters, they're likely similar (or coordinated)
- **SEPS model**: Semi-supervised GNN, spreads partial labels through network
- Only needs a FEW known anomalous channels as seeds → propagates to find more
- **Features used**:
  - Network features: co-commenter overlap, graph centrality
  - Engagement features: views, likes, comments ratios
- **Ground truth**: "Suspended" channels by YouTube = known anomalies
- **Key insight**: You don't need full labels. A few known bad actors + network structure = find the rest

### What we can borrow:
- Semi-supervised approach → use known controversy cases as seeds
- Network propagation idea → our collaboration network
- **BUT**: We don't have commenter data at scale (only 22K comments)

---

## Paper 3: Behavior Change as Signal for Social Media Manipulation (2026)
**Venue**: arXiv 2026
- Track accounts over time, detect CHANGES in behavior
- Temporal anomaly detection rather than static classification
- **Borrow**: Frequency acceleration as temporal signal

## Paper 4: Predicting Misinformation Spreaders (Verdolotti, 2025)
- Regression model to predict future contribution to misinformation
- **Ranking approach** rather than binary classification
- **Borrow**: Risk scoring rather than binary "exploitative or not"

---

## SYNTHESIS: Methodological Gaps & Opportunities

### Ground Truth Problem (THE critical issue):
| Paper | Ground Truth | Scale |
|-------|-------------|-------|
| Papadamou 2020 | Manual annotation (76 students watched videos) | 5K labeled → 856K classified |
| SEPS 2025 | YouTube suspensions | 97 channels |
| Our study | ~5-10 known controversy cases | 25 channels |

### What features are available to us:
- ✅ Upload frequency (videos/week)
- ✅ Frequency acceleration (trend over time)
- ✅ Cross-platform presence (YouTube + TikTok)
- ✅ Sponsorship rate + child brand targeting
- ✅ Collaboration network centrality
- ✅ MCN affiliation
- ✅ Engagement ratios (views/sub, likes/views, comments/views)
- ✅ Video duration statistics
- ❌ Co-commenter network (limited)
- ❌ Visual/thumbnail features (not collected)
- ❌ Video content analysis (not feasible at scale)

### Key Takeaways for Our Paper Design:
1. **Title/description NLP is NOT enough** — Papadamou explicitly showed this
2. **Metadata + engagement features can get ~80% accuracy** — encouraging
3. **Semi-supervised with few seeds works** — SEPS showed this
4. **Temporal signals matter** — behavior change papers
5. **Ranking > binary classification** — more nuanced, more useful
6. **Need to SCALE UP** — 25 channels is too few; need hundreds

### Recommended Paper Design:
**"Detecting At-Risk Kidfluencer Channels: A Multi-Signal Behavioral Approach"**

1. Scale up to 200+ family channels (use YouTube API search)
2. Compute behavioral features (frequency, engagement, commercial, network)
3. Unsupervised anomaly detection to find outlier patterns
4. Validate against known controversy cases
5. Predict risk scores for unlabeled channels
6. Cross-platform validation (YouTube → TikTok consistency)

This avoids the ground truth problem by framing as anomaly detection, not classification.
