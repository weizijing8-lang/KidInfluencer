# Literature Review: Methods for Detecting Harmful Content/Accounts on Social Media

## 1. Papadamou et al. (2020) - "Disturbed YouTube for Kids" (ICWSM 2020)

**RQ:** Can we detect inappropriate videos targeting toddlers on YouTube?

**Data Collection (KEY - snowball approach):**
- Used 4 seed approaches to find videos:
  1. Keywords from r/ElsaGate subreddit → 64 keywords → extract n-grams from titles → YouTube search → 893 seed videos
  2. 33 channels mentioned on r/ElsaGate → collect all their videos → 181 additional seeds
  3. Keywords from r/fullcartoonsonYouTube → 83 keywords → 2,342 seed videos
  4. REST API for random YouTube video IDs → 8,391 random videos
  5. Most popular videos in USA, UK, Russia, India, Canada → 500 seed videos
- For each seed video: follow YouTube recommendations for up to 3 hops → 12K seed + 844K recommended = ~856K total
- **Total dataset: 12,097 unique seed videos + recommended videos**

**Ground Truth / Annotation:**
- 4 categories: suitable, disturbing, restricted (MPAA NC-17/R), irrelevant
- Manually labeled by trained annotators
- Collapsed to binary: appropriate vs inappropriate
- Made dataset publicly available (4,797 videos with labels)

**Features (for classifier):**
- 1) Title and description text
- 2) Thumbnail image
- 3) Tags
- 4) Video metadata (views, likes, dislikes, comments, duration)
- 5) Channel metadata
- **Used BOTH metadata + content features**

**Classifier:**
- Deep learning binary classifier
- 84.3% accuracy
- Outperformed several baselines

**Key Insight for Us:**
- They had a CLEAR ground truth (human-annotated "disturbing" vs "appropriate")
- They used snowball sampling from seed videos, not hand-picked channels
- They analyzed VIDEOS not CHANNELS
- Their contribution was the classifier + the characterization of the problem at scale

---

## 2. SEPS (2025) - "Partial-Label Anomalous YouTube Channel Detection"

**Method:**
- Semi-supervised approach with partial labels
- Used channel-level behavioral features
- Anomaly detection framework
- Key innovation: works with incomplete/noisy labels

---

## 3. Key Methods from Search Results

### Cross-Platform Analysis:
- "Green vs. Greed" (2025): 19,737 Instagram posts + 27,913 TikTok videos + 36,688 YouTube videos by 236+490+X influencers
- Cross-platform user identity linkage is a mature field (Zhou et al. 2015, 242 citations)

### Sponsored Content Detection:
- "Your posts betray you" (2022): Deep learning to detect sponsored influencer posts
- Features: linguistic cues, posting patterns, engagement metrics

### Anomaly Detection on Social Networks:
- Network-centric approach for YouTube (2025): coordinated commenter analysis using PCA + unsupervised methods
- "Towards detecting anomalous user behavior" (USENIX 2014): unsupervised, PCA on behavioral features

### Child Safety Risk Assessment:
- Ta (2024): "Safety risk assessment framework for children's online safety"
- Livingstone & Stoilova (2021): "The 4Cs" framework - Content, Contact, Conduct, Contract risks (236 citations)

---

## 4. Available APIs for Scaling Up

We have access to:
- **YouTube Data API v3** (our own key, quota limited)
- **Manus YouTube Channel Details API** (no quota limit, gets channel info + latest videos)
- **Manus YouTube Channel Videos API** (no quota limit, gets full video list with metadata)
- **Manus TikTok APIs** (user info, popular posts, search)

This means we can potentially scale to hundreds or thousands of channels without quota issues.

---

## 5. Gaps and Opportunities

### What's been done:
- Video-level classification of "disturbing" content (Papadamou)
- Channel-level anomaly detection (SEPS)
- Cross-platform influencer analysis (various)
- Sponsored content detection (various)

### What's NOT been done:
- **Channel-level risk scoring for child exploitation** using only metadata (no video content)
- **Cross-platform labor burden quantification** for kidfluencers
- **Industrial structure mapping** (MCN networks) behind kidfluencers
- **Longitudinal analysis** of how channels evolve over time
- **Systematic comparison** of kidfluencer channels vs adult channels on structural metrics

### Our potential unique contribution:
- A **metadata-only risk scoring framework** that doesn't require watching videos
- Scalable to any channel using public API data
- Multi-dimensional (labor + commercial + network + temporal)
- Cross-platform (YouTube + TikTok)

---

## 6. Adeliyi et al. 2024 - "Detecting and Characterizing Inorganic User Engagement on YouTube"
- **Source**: ICWSM 2024 Workshop
- **Dataset**: 3,542 Indo-Pacific YouTube channels
- **Method**: Time series analysis of user engagement statistics + supervised/unsupervised ML
- **Key approach**: 
  - Analyze comments, views, subscribers, videos of channels to detect inorganic activity
  - Highlight specific period where activity is detected
  - Explain behavior within channels that exhibit these behaviors
  - Predict which channels are likely to exhibit such behaviors in the future
- **Techniques**: Rolling window correlation analysis, anomaly detection, rule-based classification, clustering
- **Data collection**: YouTube API (Google Developers 2021) + Social Blade (2023)
- **Key insight**: Time series analysis of engagement metrics can reveal patterns invisible in aggregate statistics
- **Relevance to us**: We could use similar time series anomaly detection on upload frequency, view patterns, etc. to identify "abnormal" family channels


---

## 7. Jo & Wojcieszak (2025) - "MetaHarm: Harmful YouTube Video Dataset" (ICWSM 2025)

**Key approach:**
- 60,906 potentially harmful YouTube videos identified using 3 approaches: keyword-based, channel-based, external dataset integration
- 19,422 videos annotated by domain experts, GPT-4-Turbo (14 image frames + 1 thumbnail + text metadata), and crowdworkers (AMT)
- **6 harm categories**: Information, Hate and harassment, Addictive, Clickbait, Sexual, Physical harm
- Both binary (harmful vs harmless) AND multi-label categorization
- **Multimodal**: combines image, audio, and text
- Dataset at https://zenodo.org/records/14647452

**Key insight for us:**
- They use a TAXONOMY of harm types, not a single score
- They combine multiple annotation sources (experts + LLM + crowd) for validation
- 60K videos is a realistic scale for this kind of study
- The 6 harm categories could inspire our exploitation dimensions


### MetaHarm Taxonomy (Table 1):
| Harm Category | Subcategories |
|---|---|
| Information harms | Fake news, Conspiracy theories, Unverified medical treatments, Unproven scientific myths |
| Hate and harassment | Insults/obscenities, Identity attacks, Hate speech (gender/race/religion/etc) |
| Addictive harms | Online gameplay, Drug/smoking/alcohol, Gambling-play videos |
| Clickbait harms | Clickbaitive titles, Get-rich-quick schemes, Gossip promotion |
| Sexual harms | Erotic scenes/images, Depictions of sexual acts/nudity, Sexual abuse |
| Physical harms | Self-injury/suicide, Eating disorder promotion, Dangerous challenges/pranks |

### MetaHarm Data Collection:
- 3-step process: (1) keyword-based, (2) channel-based, (3) external dataset integration
- Used 169 keywords from platform community guidelines + past work
- Used YouTube Data API v3 for metadata + YouTubeTranscriptApi for transcripts
- Recency + relevance filters in 7:3 ratio
- Collected Oct 2023 - Jan 2024


## 8. Dutta et al. (2021) - "Detecting and Analyzing Collusive Entities on YouTube" (ACM TIST)

This paper addresses the detection of three types of collusive YouTube entities: videos seeking artificial likes, channels seeking artificial subscriptions, and videos seeking artificial comments. The key methodological contribution is the use of **one-class classifiers** (SVM-based) trained only on curated collusive entities plus novel features, achieving TPR of 0.911 for videos and 0.910 for channels. For temporal collusion (comments), they propose **CollATe**, an end-to-end neural architecture combining metadata features, anomaly features from comment time-series, and comment text similarity scores.

The one-class classifier approach is directly relevant because our kidfluencer problem also has a "small positive set" issue — we have a few known exploitation cases (8 Passengers, DaddyOFive) but cannot easily label the full population. One-class SVM learns from only the positive (anomalous) class, which maps well to our scenario where we know some channels are exploitative but cannot confidently label others as "safe."

The feature engineering combines metadata features (view count, like count, subscriber count, video count, channel age) with temporal features (posting patterns, engagement spikes) and textual features (comment similarity). This multi-signal approach validates our structural metrics strategy.

**Ground truth strategy:** They curated collusive entities from blackmarket services — essentially using known fraud sources as positive labels. For our work, the analog would be using legally adjudicated cases + media investigations as positive labels.


## 9. Shajari & Agarwal (2025) - "Safeguarding YouTube Discussions: A Framework for Detecting Anomalous Commenter and Engagement Behaviors" (Social Network Analysis and Mining)

This paper introduces an **unsupervised framework** for detecting anomalous YouTube channels by analyzing two components: commenter behaviors and engagement patterns. The dataset covers 71 channels, 642,952 videos, 12.4M commenters, and 123.9M comments. The approach is notable for requiring **no labeled training data**.

**Methodology:**

The framework operates in two stages. First, it constructs a **co-commenter network** linking commenters who posted on the same videos, with edge weights based on the number of co-commented videos. From these networks, 20 features are extracted. Second, it applies **kernel density estimation (KDE)** and **Gaussian mixture model (GMM)** to assign anomaly scores without supervision. KDE estimates probability density distributions of network features, while GMM models commenter behavior as a mixture of Gaussians for probabilistic anomaly assessment.

A **composite scoring system** integrates engagement and commenter behavior scores at both feature level (cosine similarity + PCA) and output level (three methods: harmonic mean, weighted average with interaction term, agreement-weighted maximum). The final score is normalized between 0 and 1, where higher values indicate greater anomalous activity.

**Validation:** They compare detected anomalous channels with **actually suspended YouTube channels**, confirming real-world applicability.

**Key insights for our work:**

The unsupervised approach is highly relevant because we also lack comprehensive ground truth labels. The composite scoring methodology — combining multiple behavioral dimensions into a single normalized score — maps directly to our multi-dimensional exploitation risk scoring concept. Their use of KDE + GMM for identifying distributional anomalies could be applied to our structural metrics (upload frequency, commercial density, cross-platform burden) to identify channels that deviate from "normal" family channel behavior.

The validation strategy of comparing with suspended channels is particularly instructive. For our work, we could validate against: (a) channels with known legal/media investigations, (b) channels flagged by child advocacy organizations, (c) channels that have been demonetized or age-restricted.


## 10. Shajari & Agarwal (2025b) - "Developing a Network-Centric Approach for Anomalous Behavior Detection on YouTube" (Social Network Analysis and Mining)

This companion paper focuses on detecting **commenter mobs** — coordinated groups that artificially inflate engagement — using a network-centric approach. The dataset covers 47 YouTube channels (20 misinformation + 27 control), 26,901 videos, 1.4M commenters, and 2.5M comments.

**Methodology:** The approach constructs co-commenter networks per channel, then applies three representation methods in parallel: (1) **PCA** on extracted network features for dimensionality reduction, (2) **Graph2Vec** to encode each co-commenter network into fixed-length feature vectors capturing structural patterns, and (3) **UMAP** for non-linear dimensionality reduction to uncover hidden relationships. These representations are then fed into **K-means** and **hierarchical clustering** to identify anomalous channel groups.

**Key methodological challenges they identify (directly relevant to us):**

1. **Lack of ground truth** — they explicitly acknowledge this as a core challenge, making it difficult to verify detection accuracy. This mirrors our problem exactly.
2. **Limited comparative studies** — few benchmarks exist for this specific research question.
3. **Data vanishing** — YouTube suspends channels, permanently removing data needed for validation.

**Key insights for our work:**

Their approach is **content and language agnostic** — it works purely on structural/behavioral features, not on video content. This validates our decision to abandon NLP-based exploitation scoring in favor of structural metrics. The Graph2Vec + UMAP + clustering pipeline could be adapted for our channel-level risk profiling, where instead of co-commenter networks, we use channel behavioral feature vectors (upload frequency, commercial density, cross-platform presence, collaboration patterns).

Their use of a **control group** (27 randomly selected channels) to establish baseline "normal" behavior is methodologically important. For our work, the adult creator channels already serve this function.


## 11. Amure & Agarwal (2025) - "Co-commenters as Clues: A Partial-Label Approach to Detecting Anomalous Channels on YouTube" (SEPS)

This paper proposes **SEPS (Semi-supervised Embedding-based Propagation Scoring)**, a method that detects anomalous YouTube channels using only a small set of labeled anomalies. The approach combines co-commenter networks with engagement features and uses graph neural networks to propagate partial labels.

**Methodology:** SEPS uses **Deep Graph Infomax (DGI)** with GraphSAGE layers to learn node embeddings from the co-commenter network, combined with a lightweight classifier head for partial supervision. The loss function combines an unsupervised contrastive objective (DGI) with a supervised classification loss weighted by hyperparameter lambda. After training, each node (channel) receives an anomaly score between 0 and 1 via sigmoid activation.

**Key innovation:** The method remains effective when **only a handful of known anomalies are present**, maintaining high cluster purity under sparse label conditions. This is exactly our scenario — we have 2-3 known exploitation cases (8 Passengers, DaddyOFive, Piper Rockelle lawsuit) and need to score the remaining channels.

**Direct applicability to our work:**

This paper provides a concrete technical blueprint for our risk prediction framework. Instead of co-commenter networks, we could construct a **collaboration network** among family channels (from our Paper 2 data) and use SEPS-style label propagation. Our "labeled anomalies" would be channels with known legal/media investigations. The engagement features would be replaced by our structural exploitation metrics (upload frequency, commercial density, cross-platform burden, child appearance ratio).

The key insight is that **graph-based semi-supervised learning** can propagate risk labels through network connections — if a channel frequently collaborates with known exploitative channels, it receives a higher risk score even without direct evidence. This is methodologically stronger than simple feature-based clustering because it leverages network structure.


## 12. Algorithmic Auditing Methodology (Multiple Papers)

### Panigutti et al. (2025) - "How to Investigate Algorithmic-Driven Risks in Online Platforms" (FAccT 2025)
This paper develops a **taxonomy of study designs** for algorithm auditing. It analyzes existing auditing literature and categorizes approaches into structured plans for investigating algorithmic systems. This is directly relevant to framing our work as a "computational audit" of the kidfluencer ecosystem.

### Haroon et al. (2023) - "Auditing YouTube's Recommendation System" (PNAS, 90 citations)
Uses **sock puppet audit methodology** with 100,000 pre-programmed agents acting as YouTube users. Key methodological contribution: creating controlled experimental conditions to test algorithmic behavior. While our work doesn't audit recommendations per se, the sock puppet methodology could inform how we test whether YouTube's algorithm rewards exploitative content (our finding that beta(exploit_title) = 0.459).

### Chandio et al. (2024) - "How Audit Methods Impact Our Understanding of YouTube's Recommendation Systems" (ICWSM 2024)
Compares different audit methodologies and shows that methodological choices significantly affect conclusions. Important meta-methodological insight: we need to be transparent about how our channel selection, metric definitions, and analytical choices affect our findings.

### Sandvig et al. (2014) - "Auditing Algorithms: Research Methods for Detecting Discrimination on Internet Platforms"
The foundational paper on algorithmic auditing methodology. Proposes five audit designs: (1) code audit, (2) noninvasive user audit, (3) scraping audit, (4) sock puppet audit, (5) crowdsourced audit. Our approach is closest to a **scraping audit** — using publicly available API data to characterize platform-level patterns.

**Framing insight for our papers:** Rather than calling our work "risk prediction" (which implies a classifier), we should frame it as a **computational audit** of the kidfluencer ecosystem. This framing:
1. Doesn't require a large labeled dataset
2. Is well-established in the FAccT/ICWSM literature
3. Focuses on characterizing structural patterns rather than predicting individual outcomes
4. Aligns with platform accountability discourse


## 13. Legislative Context (2024-2026) — For Paper Motivation

The legislative landscape for kidfluencer protection is rapidly evolving, providing strong motivation for computational tools that can identify at-risk channels at scale:

| State | Year | Key Provision |
|---|---|---|
| Illinois | 2024 (Jul) | First state to expand child labor laws to cover kidfluencers; if 30%+ of content features a child, 15% of earnings must go to trust |
| California | 2024 (Sep) | AB 1880 expands Coogan Law to minors employed as content creators on YouTube and other platforms |
| Minnesota | 2024 | HF3488 passed House to protect exploitation of young children for profit |
| Tennessee | 2026 (Apr) | Requires family influencers to pay children a portion of earnings |
| 16 states total | 2025 | Have introduced content creator legislation requiring trust accounts for minors |

This legislative momentum creates a clear policy audience for our research. The key gap: **legislators need tools to identify which channels require oversight**, but current laws rely on self-reporting (parents declaring what percentage of content features children). A computational audit framework that can automatically flag high-risk channels would directly support enforcement.


## 14. Positive-Unlabeled (PU) Learning and One-Class Classification for Anomaly Detection

### ConPU: Contrastive Positive Unlabeled Learning (IEEE Big Data 2022)
Proposes a **contrastive loss function for PU learning** in fraud detection. The key insight: when only a few labeled malicious samples and many unlabeled samples exist, ConPU approximates the cluster center of normal sessions in representation space by using distributions of unlabeled and malicious sessions, then predicts labels by analyzing proximity to cluster centers. This is directly applicable to our scenario where we have a few known exploitative channels and many unlabeled ones.

### One-Class Neural Networks (Chalapathy et al., 2018, 676 citations)
Proposes OC-NN for unsupervised anomaly detection, combining deep feature extraction with one-class classification. The model learns a representation that separates normal from anomalous data using only normal examples. This could be inverted for our use case — train on "normal" adult creator channels and flag kidfluencer channels that deviate.

### Isolation Forest (Liu et al., 2008/extended versions)
Isolation Forest detects anomalies by randomly partitioning data and measuring how quickly instances are isolated. Anomalies require fewer partitions. This is a strong baseline method for our structural metrics — channels with unusual combinations of high upload frequency + high commercial density + young channel age would be quickly isolated.

**Key methodological insight:** The PU learning literature provides a rigorous framework for our problem. Rather than trying to build a binary classifier (exploitative vs. non-exploitative), we can frame the problem as:
- **Positive set:** Known exploitation cases (8 Passengers, DaddyOFive, Piper Rockelle)
- **Unlabeled set:** All other kidfluencer channels
- **Goal:** Estimate P(exploitative | features) for each unlabeled channel

This avoids the need for comprehensive ground truth labels while still producing meaningful risk scores.

