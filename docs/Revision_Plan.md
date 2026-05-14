# Revision Plan: Addressing Reviewer Feedback

This document outlines specific, actionable steps to address the three primary concerns raised by the reviewer. By implementing these changes, the paper will be significantly strengthened against methodological critiques.

## Attack Point 1: Validation Details
**Reviewer Concern:** "50 个视频，谁标的？标注指南是什么？有没有 inter-annotator agreement？" (Lack of detail on the human validation process).

**Actionable Steps:**
1. **Clarify Annotator Profile:** Acknowledge that the validation was performed by a single trained annotator. This is a common limitation in exploratory computational social science, but it must be stated transparently.
2. **Describe the Annotation Process:** Briefly mention that the annotator evaluated the videos based on the six definitions outlined in Section 3.2, examining the title, thumbnail, and video content as needed.
3. **Address the Lack of Inter-Rater Reliability (IRR):** Move the discussion of subjectivity into the limitations section, explicitly stating that future work should involve multiple annotators to calculate Cohen's $\kappa$ or Krippendorff's $\alpha$ between humans.

**Proposed Text Addition (Section 3.3):**
> "Validation was conducted on a subset of 50 videos manually coded by a single trained annotator using the definitions outlined in Section 3.2. While the use of a single annotator precludes the calculation of inter-rater reliability, it provides a necessary preliminary ground truth for pipeline evaluation."

**Proposed Text Addition (Section 5.3 Limitations):**
> "First, our validation relied on a single trained annotator. While our pipeline achieved strong discriminative ability against this baseline, future research must employ multiple independent annotators to establish rigorous inter-rater reliability, particularly given the subjective nature of dimensions like 'emotional bait.'"

---

## Attack Point 2: Channel Selection Criteria and Representativeness
**Reviewer Concern:** "这 79 个频道是怎么选的？代表性如何？" (Lack of detail on the 79 channels).

**Actionable Steps:**
1. **Provide Demographic/Size Data:** Add a sentence or two summarizing the size distribution of the 79 channels. Based on our data, the channels range from small (<50K median views) to very large (>5M median views), with a median of ~445K views per video.
2. **Clarify the Scope:** Explicitly state that these are predominantly English-language "family vlog" and "kidfluencer" channels.

**Proposed Text Addition (Section 3.1):**
> "Channels were selected based on prior literature and popular influencer lists, focusing primarily on English-language content. The selected 79 channels represent a diverse cross-section of the ecosystem, ranging from smaller creators to massive operations. Within our stratified sample, the median channel receives approximately 445,000 views per video, with the largest channels averaging over 5 million median views."

---

## Attack Point 3: VLM/LLM Weight Justification
**Reviewer Concern:** "0.67 和 0.33 是怎么来的？是在 validation set 上调的吗？" (Potential data leakage or overfitting in the weight selection).

**Actionable Steps:**
1. **Explain the Rationale:** The weights (0.67 for VLM, 0.33 for LLM) were not arbitrarily chosen or overfitted to the 50-video validation set to maximize AUC. Instead, they reflect a structural prior: the VLM analyzes three modalities (title, thumbnail, description) while the LLM analyzes only one (title). Therefore, the VLM is given roughly twice the weight.
2. **Report Sensitivity Analysis:** Briefly mention that the engagement premium finding (the positive correlation with views) is highly robust across *all* possible weighting schemes. Even if we rely 100% on the VLM or 100% on the LLM, the correlation remains highly significant ($p < 10^{-6}$).

**Proposed Text Addition (Section 3.3):**
> "The VLM and LLM signals were aggregated using a weighted average. We assigned a weight of 0.67 to the VLM and 0.33 to the LLM, reflecting the structural prior that the VLM analyzes a richer multimodal context (title, thumbnail, and description) compared to the LLM's text-only analysis (title). Sensitivity analyses confirmed that the highly significant correlation between the resulting exploitation score and view counts ($p < 10^{-6}$) is robust across all possible weighting schemes, including relying solely on the VLM or the LLM."

---

## Summary of Changes to be Made to `AIES_2026_Paper_Final.md`
1. Update Section 3.1 to include channel size metrics.
2. Update Section 3.3 to clarify the single-annotator validation process and the heuristic justification for the 0.67/0.33 weights.
3. Update Section 5.3 to explicitly mention the need for multiple annotators in future work.
