# Risk Prediction Framework & Ground Truth Strategy

This document outlines the specific computational methodologies for predicting exploitation risk in kidfluencer channels, addressing the critical challenge of limited ground truth data. This framework directly informs the methodological design for our upcoming papers.

## 1. The Core Challenge: Lack of Ground Truth

In traditional supervised machine learning, detecting harmful content requires a large, balanced dataset of labeled examples (e.g., 10,000 "safe" channels and 10,000 "exploitative" channels). 

For kidfluencers, this is impossible because:
1. **Exploitation is structural, not always visual:** A child might appear happy on camera but be subjected to grueling 60-hour work weeks off-camera.
2. **Subjectivity:** What one annotator considers "normal family vlogging," another might consider "privacy violation."
3. **Data Scarcity:** We only have a handful of confirmed, legally adjudicated cases of severe exploitation (e.g., 8 Passengers, DaddyOFive).

Therefore, our framework must operate in a **Positive-Unlabeled (PU)** or **Semi-Supervised** paradigm.

## 2. Defining the "Positive Set" (Ground Truth Strategy)

Instead of trying to label thousands of channels, we will curate a highly reliable, albeit small, **Positive Set** of known exploitative channels. 

This set will be constructed using rigorous external criteria:
1. **Legal Adjudication:** Channels where parents were convicted of child abuse or labor violations (e.g., Ruby Franke of *8 Passengers*).
2. **Media/Journalistic Investigations:** Channels subject to major exposés by reputable news organizations (e.g., *DaddyOFive*, *FamilyOFive*).
3. **Platform Action:** Channels that were permanently demonetized or terminated by YouTube specifically for child safety violations (excluding general copyright strikes).
4. **Civil Lawsuits:** Cases where former child stars sued their parents/managers for financial exploitation or abuse (e.g., the Piper Rockelle squad lawsuits).

**The Unlabeled Set:** All other kidfluencer channels in our dataset. We *do not* assume they are safe; we simply treat their status as unknown.

## 3. The Three-Tiered Risk Prediction Framework

We propose a three-tiered computational approach, moving from simple heuristics to advanced semi-supervised learning.

### Tier 1: Structural Heuristic Scoring (Rule-Based)

Before applying complex ML, we will develop a transparent, interpretable risk score based on the structural metrics developed in Paper 1.

*   **Labor Burden Score:** Normalized combination of upload frequency, average video duration, and cross-platform presence (e.g., maintaining active YouTube, TikTok, and Instagram accounts simultaneously).
*   **Commercialization Score:** Normalized combination of sponsored video ratio, merchandise link density in descriptions, and presence of dedicated business contact emails.
*   **Privacy Intrusion Score:** Ratio of videos filmed in private spaces (bedrooms, bathrooms) vs. public spaces, and frequency of highly personal topics (medical issues, discipline, emotional breakdowns) in titles/tags.

**Method:** Calculate Z-scores for each metric relative to the adult creator baseline. Channels exceeding a certain threshold (e.g., +2 standard deviations) across multiple dimensions are flagged as high risk.

### Tier 2: Unsupervised Anomaly Detection (One-Class Classification)

This approach learns what "normal" labor looks like on YouTube and flags kidfluencers who deviate significantly.

*   **Training Data:** The dataset of adult creator channels (the control group).
*   **Features:** The structural metrics defined above.
*   **Algorithms:**
    *   **Isolation Forest:** Highly effective at isolating anomalies in multi-dimensional space. Kidfluencer channels with extreme combinations of high output and high commercialization will be isolated quickly.
    *   **One-Class SVM (OC-SVM):** Learns a boundary encompassing the "normal" adult labor patterns. Kidfluencers falling outside this boundary are flagged.

**Output:** An anomaly score for each kidfluencer channel. Higher scores indicate structural labor patterns that deviate significantly from typical adult creators.

### Tier 3: Graph-Based Semi-Supervised Learning (Label Propagation)

This is the most advanced tier, adapting the SEPS (Semi-Supervised Embedding-based Propagation Scoring) methodology [1]. It leverages network structure to propagate risk from our small Positive Set to the Unlabeled Set.

1.  **Network Construction:** Build a collaboration/similarity graph of kidfluencer channels. Edges can be defined by:
    *   Direct collaborations (mentioning each other in titles/descriptions).
    *   Shared management/MCN (Multi-Channel Network) affiliation.
    *   High cosine similarity in their structural feature vectors.
2.  **Label Initialization:** Assign a risk score of 1.0 to the Positive Set channels. All Unlabeled channels start at 0.5 (unknown).
3.  **Propagation:** Use Graph Neural Networks (e.g., Deep Graph Infomax with a classification head) to propagate the labels through the network.
4.  **Rationale:** If an unlabeled channel shares management, frequently collaborates with, or exhibits identical structural patterns to a known abusive channel (like *8 Passengers*), its predicted risk score will increase.

## 4. Validation Strategy

Because we lack a massive ground truth dataset, we must validate our predicted risk scores creatively:

1.  **Top-K Qualitative Review:** Expert annotators will conduct a deep-dive qualitative review of the top 1% (highest risk) and bottom 1% (lowest risk) channels predicted by the models to ensure face validity.
2.  **Correlation with Platform Moderation:** We will check if high-risk channels are more likely to have their comments disabled by YouTube (a known platform mitigation for child safety risks).
3.  **Predictive Validity:** Longitudinally track the channels over 6-12 months to see if high-risk channels are more likely to be demonetized, abandoned, or subject to controversy in the future.

## References

[1] R. Amure and N. Agarwal, "Co-commenters as Clues: A Partial-Label Approach to Detecting Anomalous Channels on YouTube," *Social Network Analysis and Mining*, 2025.
