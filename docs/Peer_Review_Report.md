# Peer Review Report: "Auditing Engagement Incentives in the Kidfluencer Ecosystem: A Multimodal Weak Supervision Approach"

**Conference:** AIES 2026 (Artificial Intelligence, Ethics, and Society)
**Recommendation:** Accept (with minor revisions)
**Confidence:** High

## Summary of the Paper
This paper presents a large-scale observational audit of the "kidfluencer" ecosystem on YouTube. The authors aim to understand how platform engagement metrics (views) associate with various dimensions of child exploitation. To overcome the bottleneck of manual annotation, the authors employ a multimodal weak supervision pipeline, combining Large Language Models (LLMs) for text classification and Vision-Language Models (VLMs) for analyzing thumbnails and descriptions. Analyzing 5,051 videos across 79 channels, the study finds a significant "engagement premium" for videos featuring performative labor, emotional bait, and privacy violations. Interestingly, explicit commercial content is associated with a negative engagement premium. The authors argue that platforms systematically reward the commodification of children's identities and emotions over traditional advertising.

## Overall Evaluation
This is an excellent, timely, and methodologically innovative paper. The topic of child digital labor is highly relevant to the AIES community, and the authors tackle a significant empirical gap: how to scale the measurement of subjective ethical concepts like "exploitation" in massive digital ecosystems. The use of multimodal weak supervision (combining LLMs and VLMs) is a clever and effective solution. The findings, particularly the contrast between the "performativity premium" and the "commercial penalty," offer nuanced and valuable insights for policymakers. The paper is well-written, methodologically sound, and its limitations are acknowledged transparently.

## Strengths
1. **Methodological Innovation:** The application of weak supervision using modern VLMs (GPT-4 Vision) to audit subjective ethical dimensions is highly novel. The authors successfully demonstrate how to operationalize complex sociological frameworks (e.g., the UNCRC and "transactional childhood") into computable signals.
2. **Nuanced Findings:** The discovery that commercial content (product placement) suffers a penalty while performative labor and emotional bait receive a massive premium is a crucial contribution. It challenges the assumption that kidfluencer exploitation is purely about traditional advertising, highlighting the shift toward the commodification of identity and drama.
3. **Statistical Rigor:** The authors employ appropriate statistical controls, including mixed-effects regression to account for channel-level baseline popularity and within-channel, same-year robustness checks to control for video age. This strengthens the claim that the engagement premium is a structural feature of the content type rather than a confounding artifact.
4. **Clear and Responsible Framing:** The authors correctly identify their metric as an "engagement premium" rather than claiming direct access to YouTube's recommendation algorithm. This observational framing is intellectually honest and appropriate for an external audit.

## Weaknesses and Areas for Improvement
1. **Validation Details Could Be Expanded:** While the authors report strong discriminative ability (AUC-ROC = 0.795) and significant rank correlation on a validation set of 50 videos, the details of this validation process are somewhat sparse. 
    * *Suggestion:* Briefly explain who annotated the 50 videos, what guidelines were provided, and whether there was inter-annotator agreement among human coders before comparing them to the pipeline. A confusion matrix or a brief discussion of false positives/negatives in the validation set would add depth.
2. **Subjectivity of Dimensions:** The paper acknowledges that dimensions like "emotional bait" and "narrative conflict" are highly subjective. However, it would be beneficial to provide 1-2 qualitative examples of videos where the pipeline succeeded or struggled to differentiate between genuine family documentation (vlogs) and manufactured "staged drama."
3. **Generalizability of the 79 Channels:** The authors mention selecting channels based on "prior literature and popular influencer lists." 
    * *Suggestion:* Provide a slightly more detailed demographic breakdown of these 79 channels (e.g., geographic distribution, average subscriber count) to help readers understand the scope of the ecosystem being audited. Are these predominantly US-based, English-speaking channels?

## Specific Questions for the Authors
1. How sensitive is the pipeline to the specific weights chosen for the VLM (0.67) and LLM (0.33)? Was this weighting optimized on the validation set, and if so, how might that affect the generalizability of the AUC-ROC score?
2. Did the authors observe any differences in the engagement premium based on the perceived age or gender of the children involved? While perhaps beyond the scope of this paper, it would be an interesting point for the discussion section.
3. The negative premium for commercial content is fascinating. Could this be partially due to YouTube's own policies (e.g., algorithmic downranking of videos marked as "made for kids" that contain overly commercial content to comply with COPPA)? A brief discussion of this possibility would strengthen Section 5.1.

## Conclusion
This paper makes a substantial contribution to the study of platform ethics and algorithmic auditing. By quantifying the "performativity premium," the authors provide critical empirical evidence that can inform future regulations beyond simple financial trusts. I strongly recommend this paper for acceptance at AIES 2026.
