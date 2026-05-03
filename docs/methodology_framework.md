# Computational Audit Framework for Kidfluencer Exploitation Risk

This document synthesizes current computational methodologies for detecting anomalous and harmful content on social media, adapting them into a structured framework for auditing the kidfluencer ecosystem. This framework directly supports the methodological design for our two upcoming papers.

## 1. The Challenge of "Ground Truth" in Child Exploitation

A central challenge in computational research on child exploitation in family vlogging is the lack of comprehensive ground truth labels. Unlike explicit policy violations (e.g., CSAM or hate speech) which platforms actively moderate, exploitation in family vlogging is often subtle, structural, and legal under current (but evolving) frameworks [1]. 

Previous research on disturbing YouTube content for children (e.g., Papadamou et al., 2020) relied on manual annotation of thousands of videos [2]. However, this approach is insufficient for our research because:
1. We are evaluating **channel-level structural exploitation** (labor burden, commercialization), not just video-level disturbing content.
2. The true status of a child's well-being cannot be definitively ascertained from video content alone.
3. We only have a very small set of confirmed positive cases (e.g., legally adjudicated cases like 8 Passengers or DaddyOFive).

## 2. Methodological Paradigms for Small Positive Sets

To address the "small positive set" problem, we adapt methodologies from fraud detection, anomaly detection, and algorithmic auditing.

### 2.1 Positive-Unlabeled (PU) Learning
When only a few labeled malicious samples and many unlabeled samples exist, the problem can be framed as Positive-Unlabeled (PU) learning. Recent advances, such as Contrastive PU Learning (ConPU), approximate the cluster center of normal sessions in representation space and predict labels by analyzing proximity to these centers [3]. 

**Application to Kidfluencers:**
*   **Positive set:** Known exploitation cases (e.g., 8 Passengers, DaddyOFive, Piper Rockelle).
*   **Unlabeled set:** All other kidfluencer channels.
*   **Goal:** Estimate $P(\text{exploitative} | \text{features})$ for each unlabeled channel based on its structural similarity to the positive set.

### 2.2 One-Class Classification and Unsupervised Anomaly Detection
One-class classification (OCC) models, such as One-Class Neural Networks (OC-NN) [4] and Isolation Forests [5], learn a representation that separates normal from anomalous data using only normal examples. Shajari & Agarwal (2025) successfully applied unsupervised techniques (Kernel Density Estimation and Gaussian Mixture Models) to detect anomalous YouTube commenter behaviors without labeled training data [6].

**Application to Kidfluencers:**
*   We can use our adult creator channels as the "normal" baseline.
*   We train OCC models or apply KDE/GMM to our structural metrics (upload frequency, commercial density, cross-platform burden).
*   Channels that deviate significantly from adult labor norms are flagged as anomalous (high risk).

### 2.3 Graph-Based Semi-Supervised Learning (Label Propagation)
Amure & Agarwal (2025) proposed SEPS (Semi-Supervised Embedding-based Propagation Scoring), which detects anomalous YouTube channels using only a small set of labeled anomalies by propagating labels through a co-commenter network [7]. 

**Application to Kidfluencers:**
*   We can construct a **collaboration network** among family channels (using features, tags, or actual collaborations).
*   We apply SEPS-style label propagation, where the few known exploitative channels act as the "labeled anomalies."
*   Risk scores propagate through the network—channels structurally similar or connected to known bad actors receive higher risk scores.

## 3. Framing: The Computational Audit

Rather than framing our work as building a "risk prediction classifier" (which implies a definitive ground truth we lack), we adopt the framing of a **Computational Audit**.

Algorithmic auditing is a well-established methodology in the FAccT and ICWSM communities for investigating systemic risks on platforms [8, 9]. Our approach aligns with the **scraping audit** design [10], using publicly available API data to characterize platform-level patterns and test whether platform algorithms (e.g., monetization, recommendations) incentivize structural exploitation.

This framing is particularly powerful given the rapidly evolving legislative context. In 2024, Illinois and California passed laws protecting the financial interests of kidfluencers [11, 12], and 16 other states have introduced similar legislation. A computational audit provides the empirical foundation needed to support these policy interventions.

## 4. Proposed Methodology for Our Papers

Based on this synthesis, we propose the following methodological framework for our two papers.

### Paper 1: The Kidfluencer Labor Market (Structural Metrics)

**Goal:** Quantify the structural differences between child and adult labor on YouTube.

**Methodology:**
1.  **Data Collection:** Snowball sampling of kidfluencer and adult creator channels using the YouTube API.
2.  **Feature Engineering:** Extract structural metrics:
    *   *Labor Burden:* Upload frequency, video duration, cross-platform presence.
    *   *Commercialization:* Sponsored video ratio, merch link density.
    *   *Performance:* Views, engagement rates.
3.  **Statistical Analysis:** Conduct rigorous comparative statistics (e.g., Mann-Whitney U tests) to demonstrate that kidfluencers operate under higher structural demands than adults.
4.  **Incentive Audit:** Use regression models to test if the platform incentivizes exploitative structures (e.g., does higher upload frequency correlate with disproportionately higher views for kids vs. adults?).

### Paper 2: Risk Profiling and the "Small Positive Set" Problem

**Goal:** Develop a scalable framework to identify at-risk kidfluencer channels.

**Methodology:**
1.  **Feature Representation:** Represent each channel as a vector of its structural metrics (from Paper 1).
2.  **Baseline Anomaly Detection:** Apply Isolation Forests and One-Class SVMs, training on the adult "normal" baseline to identify kidfluencer channels with anomalous labor structures.
3.  **PU Learning / Label Propagation:** 
    *   Define a small "Positive Set" of known, documented exploitation cases.
    *   Apply a PU learning or graph-based propagation approach (inspired by SEPS [7] and ConPU [3]) to assign a continuous "Exploitation Risk Score" (0 to 1) to all unlabeled kidfluencer channels.
4.  **Validation:** Validate the risk scores by qualitative review of the top 1% highest-risk channels and correlation with external indicators (e.g., comments disabled by YouTube, demonetization events).

## References

[1] R. Kramer, "The Exploitation of Children in Family Vlogging," *Columbia Undergraduate Law Review*, 2023.
[2] A. Papadamou et al., "Disturbed YouTube for Kids: Characterizing and Detecting Inappropriate Videos Targeting Young Children," *ICWSM*, 2020.
[3] Anonymous, "Fraud Detection via Contrastive Positive Unlabeled Learning," *IEEE Big Data*, 2022.
[4] R. Chalapathy, A. K. Menon, and S. Chawla, "Anomaly detection using one-class neural networks," *arXiv preprint arXiv:1802.06360*, 2018.
[5] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," *ICDM*, 2008.
[6] S. Shajari and N. Agarwal, "Safeguarding YouTube Discussions: A Framework for Detecting Anomalous Commenter and Engagement Behaviors," *Social Network Analysis and Mining*, 2025.
[7] R. Amure and N. Agarwal, "Co-commenters as Clues: A Partial-Label Approach to Detecting Anomalous Channels on YouTube," *Social Network Analysis and Mining*, 2025.
[8] C. Panigutti et al., "How to Investigate Algorithmic-Driven Risks in Online Platforms," *FAccT*, 2025.
[9] M. Haroon et al., "Auditing YouTube's Recommendation System," *PNAS*, 2023.
[10] C. Sandvig et al., "Auditing Algorithms: Research Methods for Detecting Discrimination on Internet Platforms," *ICA*, 2014.
[11] Illinois Child Labor Law (SB 1782), Effective July 1, 2024.
[12] California AB 1880 (Expanded Coogan Law), Enacted September 2026.
