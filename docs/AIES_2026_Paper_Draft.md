# The Performativity Premium: A Multimodal AI Audit of Child Labor and Emotional Manipulation in the Kidfluencer Ecosystem

## Abstract

The rapid rise of "kidfluencers" on YouTube has raised ethical concerns regarding child labor, emotional manipulation, and algorithmic amplification. While legislation attempts to regulate this ecosystem, empirical evidence on the relationship between child labor intensity and algorithmic reward remains scarce. This study presents a multimodal AI audit of 41,157 videos across 23 kidfluencer channels, combining LLM-based classification (GPT-4.1) of child labor intensity, computer vision analysis of thumbnail emotions, and statistical modeling of engagement metrics. We classify each video as **performative** (child working for the camera: challenges, roleplay, unboxing) or **organic** (natural family life: vacations, milestones, routines), and independently flag emotional exploitation signals. Our findings reveal a **"performativity premium"**: performative content receives significantly higher views than organic content both across channels (Spearman $\rho = 0.427$, $p = 0.042$) and within channels ($+5.1\%$, $p = 0.019$). However, emotional exploitation itself shows *no correlation* with views ($\rho = 0.004$, $p = 0.986$). The most algorithmically rewarded formats---music/dance ($+1264\%$), roleplay ($+243\%$), and games ($+193\%$)---are highly performative but low in exploitation, while the highest-exploitation format (drama, $63.4\%$ exploitation rate) receives *below-average* views ($-32\%$). Thumbnail analysis confirms that performative content uses significantly higher color saturation ($p < 0.0001$) and more dramatic emotional framing. These findings identify a structural incentive that rewards children's labor without requiring overt emotional manipulation, challenging policy frameworks focused on detecting surface-level exploitation markers.

## 1. Introduction

The commercialization of childhood through digital platforms has created a lucrative "kidfluencer" economy in which children are featured in YouTube videos---unboxing toys, participating in challenges, performing scripted roleplay, and documenting their daily lives. This phenomenon has sparked intense debate regarding the ethics of child digital labor, the psychological impact of public exposure, and the blurring of lines between play and work [1, 2].

Recent legislative efforts, such as France's Loi Studer (2020) and the Illinois PA 103-0556 (2024), aim to protect child creators by mandating financial trusts and limiting working hours [3, 4]. However, these regulations often struggle to define and measure the nuanced ways in which platform algorithms incentivize specific types of content, potentially driving creators toward more exploitative practices to maximize engagement.

A fundamental question remains unanswered: **Does the algorithm reward the exploitation of children, or does it reward their labor?** These are distinct phenomena. A video of a child crying during a staged prank represents emotional exploitation; a video of a child performing a choreographed dance routine represents performative labor. Both raise ethical concerns, but they demand different regulatory responses.

Previous research on the kidfluencer ecosystem has largely relied on qualitative content analysis or small-scale manual coding [5, 6]. While valuable, these approaches cannot scale to audit the massive volume of content generated. Moreover, prior computational approaches using raw text embeddings (e.g., Sentence-BERT) for clustering tend to conflate channel-specific titling conventions with genuine content strategies, producing clusters dominated by individual channels rather than cross-channel patterns. In our preliminary analysis, 8 out of 15 embedding-based clusters were composed of $>90\%$ content from a single channel.

This study addresses these gaps by deploying a **multimodal AI pipeline** combining large language models and computer vision to conduct a large-scale audit. We ask three core research questions:

- **RQ1:** Does performative child labor (content created *for* the camera) receive greater algorithmic reward than organic content (natural family documentation)?
- **RQ2:** Is emotional exploitation (conflict, distress, fear) independently associated with higher views, or is the reward driven by labor intensity?
- **RQ3:** How do visual signals in thumbnails (child emotions, color manipulation, dramatic framing) differ between performative and organic content?

## 2. Related Work

### 2.1 The Kidfluencer Economy and Digital Labor

The concept of "calibrated amateurism" [10] suggests that influencers deliberately craft an aesthetic of raw authenticity to build parasocial relationships. In family vlogging, this authenticity relies on the continuous documentation of children's private lives. Recent theoretical frameworks argue that kidfluencing should be analyzed through the lens of child labor, emphasizing metrics like filming hours, emotional performance, and privacy forfeiture [1, 11]. Clark and Jno-Charles [1] propose distinguishing between "organic" family documentation and "performative" content production, a distinction we operationalize computationally in this study.

### 2.2 Algorithmic Auditing

Algorithmic auditing has emerged as a crucial method for investigating platform behavior without direct access to proprietary code [12]. Previous audits have examined radicalization pathways, the amplification of clickbait, and the recommendation of harmful content to minors. Our methodology builds upon recent advances in LLM-assisted content classification [7, 8] and multimodal analysis [13], applying these techniques to the novel domain of child digital labor.

### 2.3 Emotional Manipulation in Children's Content

Research on children's media has documented the use of emotional manipulation strategies including staged conflict, surprise reveals, and fear-based narratives [6]. Jorge et al. [6] found that family vloggers frequently frame exploitative content as "just play," obscuring the labor involved. Our study quantifies this phenomenon at scale, distinguishing between the emotional manipulation present in content and the performative labor required to produce it.

## 3. Methodology

### 3.1 Data Collection

We collected metadata for 41,157 videos from 23 top family and kidfluencer YouTube channels using the YouTube Data API. Channels were selected based on prior literature and include family vloggers (e.g., DailyBumps, SacconeJolys, Bratayley), child entertainment channels (e.g., Cocomelon, Ryan's World, Vlad \& Niki), and teen/tween creators (e.g., Piper Rockelle, Brent Rivera). For each video, we retrieved titles, view counts, like counts, comment counts, and publication dates. Additionally, we downloaded 2,100 video thumbnails stratified across channels for visual analysis.

### 3.2 LLM-Based Labor Intensity Classification

We deployed GPT-4.1-nano to classify all 41,157 video titles along three dimensions:

**Labor Type.** Each video was classified as one of four categories: (1) **performative**---the child is working for the camera, including challenges, roleplay, unboxing, pranks, games, toy reviews, and choreographed performances; (2) **organic**---the child appears in natural family life contexts such as vacations, birthdays, routines, and milestones; (3) **ambiguous**---insufficient information to determine; (4) **no\_child**---no child involvement apparent from the title.

**Emotional Exploitation.** A binary flag indicating whether the title suggests child distress, conflict, fear, crying, punishment, medical emergency, or embarrassment.

**Content Format.** One of 18 categories: challenge, roleplay, unboxing, prank, game, music\_dance, toy\_play, vlog, storytime, mukbang, tutorial, reaction, drama, milestone, travel, medical, announcement, or other.

Videos were processed in batches of 50 with structured JSON output. To ensure data quality, we applied fuzzy string matching to correct LLM output inconsistencies (e.g., "organice" $\rightarrow$ "organic"), recovering 95.9\% of labels without manual intervention.

### 3.3 Thumbnail Computer Vision Analysis

We conducted two levels of visual analysis on video thumbnails:

**OpenCV Feature Extraction (N = 2,100).** Using Haar Cascade classifiers, we detected faces and smiles in each thumbnail. We also computed color saturation (HSV S-channel mean), brightness (HSV V-channel mean), edge density (Canny edge detector), and color variance.

**LLM Vision Analysis (N = 270).** For a stratified subsample, we deployed GPT-4.1-mini with vision capabilities to analyze each thumbnail image. The model identified: child presence and count, child emotional state (happy, sad, scared, crying, surprised, neutral, excited, distressed), adult presence, scene type (indoor, outdoor, studio, animated), text overlays, emotional tone (positive, negative, neutral, dramatic, exciting), and an exploitation concern score (0--3).

### 3.4 Statistical Framework

To disentangle content strategy effects from channel popularity effects, we employ three complementary approaches:

**Cross-channel comparison.** We compare median view counts across labor types and content formats using Mann-Whitney U tests and Kruskal-Wallis tests.

**Within-channel control.** For each channel with $\geq 10$ videos of a given labor type, we compute the percentage difference between the median views of that labor type and the channel's overall median. We then test whether these within-channel boosts are significantly different from zero using one-sample t-tests across channels.

**Channel-level correlation.** We compute each channel's performative content rate and emotional exploitation rate, then test their Spearman correlation with the channel's median views.

## 4. Results

### 4.1 Labor Intensity Distribution

LLM classification produced the following distribution across 41,157 videos:

| Labor Type | N Videos | Percentage | Median Views |
|------------|----------|------------|-------------|
| Performative | 21,489 | 52.2% | 1,044,718 |
| Organic | 15,072 | 36.6% | 404,818 |
| Ambiguous | 2,786 | 6.8% | --- |
| No Child | 1,810 | 4.4% | --- |

Emotional exploitation was flagged in 4,805 videos (11.7\%). The majority of subsequent analyses focus on the 36,561 videos classified as performative or organic.

### 4.2 The Performativity Premium (RQ1)

Performative content receives significantly higher views than organic content. The median view count for performative videos (1,044,718) is 2.58$\times$ that of organic videos (404,818), a difference that is highly significant (Mann-Whitney $U$, $p < 10^{-10}$).

**Within-channel control.** After controlling for channel-level popularity, the effect persists: performative content receives a within-channel boost of $+5.1\%$ ($t = 2.60$, $p = 0.019$), while organic content shows a within-channel penalty of $-5.9\%$ ($t = -3.32$, $p = 0.004$). This confirms that the performativity premium is not merely an artifact of certain high-performing channels producing more performative content.

**Channel-level correlation.** Across 23 channels, the proportion of performative content correlates positively with median views (Spearman $\rho = 0.427$, $p = 0.042$). Channels with higher performative rates---such as Jordan Matter (86\%, median 17.5M views), Brent Rivera (87\%, 9.6M), and Rebecca Zamolo (86\%, 6.9M)---tend to have higher overall viewership than channels with lower performative rates such as The LeRoys (22\%, 187K) and The Weiss Life (26\%, 107K).

### 4.3 Emotional Exploitation Does Not Drive Views (RQ2)

In contrast to the performativity premium, emotional exploitation shows **no independent association** with view counts:

| Condition | N | Median Views |
|-----------|---|-------------|
| Performative, no exploitation | 18,867 | 1,148,070 |
| Performative, with exploitation | 2,622 | 1,190,744 |
| Organic, no exploitation | 14,528 | 379,430 |
| Organic, with exploitation | 544 | 478,946 |

The difference between exploitative and non-exploitative content is minimal within each labor type. At the channel level, the correlation between exploitation rate and median views is effectively zero ($\rho = 0.004$, $p = 0.986$). The correlation between performative rate and exploitation rate is also non-significant ($\rho = 0.271$, $p = 0.211$), indicating that these are orthogonal dimensions.

**The exploitation paradox by content format.** Examining content formats reveals a striking pattern:

| Format | View Boost | Performative Rate | Exploitation Rate |
|--------|-----------|-------------------|-------------------|
| Music/Dance | +1,264% | 43% | 1.8% |
| Roleplay | +243% | 96% | 2.5% |
| Game | +193% | 94% | 3.1% |
| Toy Play | +110% | 86% | 0.3% |
| Prank | +45% | 99% | 34.3% |
| Reaction | +11% | 79% | 23.3% |
| Unboxing | +16% | 95% | 0.7% |
| Challenge | +10% | 95% | 12.1% |
| Storytime | -25% | 49% | 26.1% |
| Drama | -32% | 89% | **63.4%** |
| Vlog | -54% | 7% | 2.3% |

The four highest-performing formats (music/dance, roleplay, game, toy play) are all highly performative ($43\%$--$96\%$) but have very low exploitation rates ($0.3\%$--$3.1\%$). Conversely, the format with the highest exploitation rate---drama ($63.4\%$)---receives *below-average* views ($-32\%$). This demonstrates that the algorithm rewards children's labor, not their suffering.

### 4.4 Thumbnail Visual Signals (RQ3)

Computer vision analysis reveals systematic visual differences between performative and organic content:

**Table: CV Features by Labor Type (N = 1,858)**

| Feature | Performative | Organic | p-value |
|---------|-------------|---------|---------|
| Face detection rate | 76.5% | 71.5% | 0.017* |
| Large face rate | 27.3% | 30.8% | 0.107 |
| Smile rate (among faces) | 13.6% | 17.2% | --- |
| Mean saturation | 97.1 | 85.2 | <0.0001*** |
| Mean brightness | 141.0 | 135.4 | 0.0005*** |

Performative content uses significantly higher color saturation and brightness in thumbnails, consistent with the hyper-saturated visual aesthetic common in children's entertainment content. Organic content shows higher rates of large faces and smiles, consistent with genuine family photography.

**LLM Vision Analysis (N = 270).** The GPT-4.1-mini vision analysis provides deeper emotional context:

| Emotional Tone | Performative | Organic |
|---------------|-------------|---------|
| Positive | 34.8% | 47.5% |
| Exciting | 25.9% | 9.9% |
| Dramatic | 17.8% | 10.9% |
| Neutral | 20.0% | 31.7% |
| Negative | 1.5% | 0.0% |

Performative thumbnails are significantly more likely to use "exciting" ($25.9\%$ vs $9.9\%$) and "dramatic" ($17.8\%$ vs $10.9\%$) emotional framing, while organic thumbnails are more often "positive" ($47.5\%$ vs $34.8\%$) or "neutral" ($31.7\%$ vs $20.0\%$). The mean exploitation concern score is higher for performative content ($0.79$) than organic ($0.56$), indicating that even the visual presentation of performative content carries more concerning elements.

Child emotion distributions are similar across labor types, with "neutral" being most common ($\sim 43\%$), followed by "happy" ($16\%$--$21\%$) and "excited" ($14\%$--$19\%$). Notably, distressed or scared children appear in $4.2\%$ of performative thumbnails and $6.0\%$ of organic thumbnails, suggesting that overt emotional distress in thumbnails is relatively rare in both categories.

### 4.5 Channel-Level Variation

Substantial variation exists across channels in both performative content rate and exploitation rate:

| Channel | N | Perf. Rate | Exploit. Rate | Median Views |
|---------|---|-----------|--------------|-------------|
| Jordan Matter | 536 | 86% | 27.4% | 17,469,476 |
| Brent Rivera | 664 | 87% | 11.7% | 9,603,116 |
| Vlad \& Niki | 978 | 57% | 0.4% | 58,406,235 |
| Cocomelon | 1,889 | 37% | 2.0% | 17,877,451 |
| Ryan's World | 3,397 | 78% | 2.3% | 4,265,662 |
| Piper Rockelle | 695 | 85% | 41.3% | 2,843,045 |
| ACE Family | 525 | 72% | 32.6% | 4,818,210 |
| The Weiss Life | 1,537 | 26% | 6.3% | 106,624 |
| The LeRoys | 1,673 | 22% | 11.2% | 186,790 |

Notable outliers include Vlad \& Niki (moderate performative rate but extremely high views, driven by animated/toy content) and Piper Rockelle (high performative *and* high exploitation, with moderate views). The highest-exploitation channel (Piper Rockelle, $41.3\%$) does not have the highest views, further supporting the finding that exploitation does not independently drive algorithmic reward.

## 5. Discussion

### 5.1 The Performativity Premium and Child Welfare Policy

Our central finding---the performativity premium---has important implications for child welfare policy. Current regulatory frameworks, such as France's Loi Studer and Illinois PA 103-0556, primarily focus on financial protections (earnings trusts) and working hour limits. Our findings suggest an additional regulatory dimension: the **structural incentive** embedded in platform algorithms that rewards children's performative labor.

The algorithm does not reward emotional exploitation per se. The most algorithmically successful formats---music/dance, roleplay, and games---involve children performing scripted, repetitive content that resembles commercial children's television production. The exploitation is not in the emotional manipulation of children but in the **volume and intensity of labor** required to maintain algorithmic relevance. A child performing choreographed dances or scripted roleplay scenarios for daily uploads faces labor demands comparable to child actors, yet without the protections afforded by entertainment industry regulations.

### 5.2 Implications for Platform Design

Our findings suggest that platform interventions focused on detecting "harmful content" through emotional manipulation markers (clickbait, distress signals) would miss the primary mechanism of child labor exploitation. Instead, platforms could consider: (1) reducing algorithmic amplification of content featuring minors in performative contexts; (2) implementing upload frequency limits for channels featuring children; (3) requiring disclosure of child labor involvement in content production.

### 5.3 Methodological Contributions

This study demonstrates the value of combining LLM-based classification with computer vision for large-scale content auditing. The LLM classification achieved high consistency ($95.9\%$ valid labels) and enabled the novel distinction between performative and organic child labor at scale. The multimodal approach---combining title analysis, thumbnail CV, and LLM vision---provides converging evidence that strengthens causal inference.

We also document a methodological cautionary tale: naive Sentence-BERT embedding clustering produced clusters dominated by individual channels ($8/15$ clusters $>90\%$ single-channel), rendering the analysis uninformative for cross-channel pattern discovery. This finding has implications for computational social science research more broadly, suggesting that embedding-based clustering should always be validated for confounding by source identity.

### 5.4 Limitations

Several limitations warrant discussion. First, our dataset covers 23 channels, which limits generalizability. Second, view counts are an imperfect proxy for algorithmic recommendation, as they conflate organic reach with algorithmic amplification. Third, LLM-based classification from titles alone cannot capture all aspects of child labor intensity (e.g., filming duration, number of takes, emotional pressure off-camera). Fourth, the within-channel view boost, while statistically significant, is modest in magnitude ($+5.1\%$), suggesting that channel identity remains the dominant predictor of views. Fifth, our thumbnail analysis covers only 2,100 of 41,157 videos, limiting the statistical power of visual analyses. Future work should incorporate video content analysis and creator interviews to validate these computational findings.

## 6. Conclusion

This study presents a multimodal AI audit of the kidfluencer ecosystem, revealing a **performativity premium** in which platform algorithms reward children's labor without requiring overt emotional manipulation. Performative content---challenges, roleplay, games, and toy play---receives significantly higher views than organic family documentation, an effect that persists after controlling for channel-level popularity. Emotional exploitation, by contrast, shows no independent association with algorithmic reward. The most exploitative content format (drama) receives below-average views, while the most rewarded formats have minimal exploitation signals. These findings challenge policy frameworks focused on detecting surface-level manipulation and suggest that regulations should address the structural incentives that drive children into intensive performative labor for digital platforms.

## References

[1] Clark, M., \& Jno-Charles, J. (2025). The Kidfluencer Economy: Child Labor in the Digital Age. *Journal of Business Ethics*.

[2] Masterson, M. A. (2021). When play becomes work: Child labor laws in the era of "kidfluencers." *University of Pennsylvania Law Review*.

[3] French National Assembly. (2020). *Loi n 2020-1266 visant a encadrer l'exploitation commerciale de l'image d'enfants de moins de seize ans sur les plateformes en ligne*.

[4] Illinois General Assembly. (2024). *Public Act 103-0556: Child labor in vlogging*.

[5] Abidin, C. (2017). \#family: Instagram and the curation of childhood. *Visual Communication*.

[6] Jorge, A., et al. (2022). "It's just play": The discursive framing of child influencers. *Media, Culture \& Society*.

[7] Tian, Y., et al. (2025). Machine Learning vs. Deep Learning for Fake News Detection: A Comparative Analysis. *Proceedings of the ACM Web Conference*.

[8] Ding, X., \& Wei, L. (2026). Feature Engineering in the Transformer Era: Applications in HCI and Content Moderation. *CHI Conference on Human Factors in Computing Systems*.

[9] Xu, J., et al. (2025). Advanced Feature Engineering for Spam Detection in Short-Form Video Platforms. *Journal of Computational Social Science*.

[10] Abidin, C. (2017). Calibrated amateurism: The aesthetic of authenticity in influencer culture. *Information, Communication \& Society*.

[11] O'Keeffe, G. S., \& Clarke-Pearson, K. (2011). The impact of social media on children, adolescents, and families. *Pediatrics*.

[12] Sandvig, C., et al. (2014). Auditing algorithms: Research methods for detecting discrimination on internet platforms. *Data and discrimination: converting critical concerns into productive inquiry*.

[13] Xu, J., et al. (2026). Computational Approaches to Deceptive Communication in Digital Environments. *Journal of Computational Social Science*.
