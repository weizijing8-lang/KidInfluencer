# Peer Review Report: AIES 2026

**Paper Title:** Auditing Engagement Incentives in the Kidfluencer Ecosystem: A Multimodal Weak Supervision Approach  
**Track:** Main Technical Track  
**Recommendation:** Accept (with minor revisions)  
**Confidence:** High

## 1. Summary

This paper presents a large-scale observational audit of the "kidfluencer" ecosystem on YouTube, investigating the relationship between child exploitation dimensions and algorithmic engagement (view counts). To overcome the lack of ground-truth data, the authors develop a multimodal weak supervision pipeline combining LLM text classification (GPT-4.1-mini) and Vision-Language Model analysis (GPT-4.1-mini Vision) across 5,051 videos from 79 channels. They operationalize exploitation into six dimensions based on the UNCRC framework. The core finding is a massive "engagement premium" for performative labor, privacy violations, and emotional bait, alongside a significant "commercial penalty" for overt product placement. The authors argue that this incentive structure systematically rewards the commodification of child labor and identity, challenging current policy frameworks that focus solely on financial compensation.

## 2. Overall Evaluation

This is a strong, timely, and methodologically innovative paper that aligns perfectly with the AIES mandate to analyze the societal and ethical impacts of AI systems. The authors tackle a difficult, highly subjective problem—child digital labor—by combining theoretical frameworks from media studies with scalable computational auditing techniques.

The shift from claiming direct algorithmic causation to measuring an "engagement premium" is a sophisticated and intellectually honest way to handle the black-box nature of YouTube's recommendation system. The empirical findings are compelling, particularly the contrasting effects of performative labor (positive premium) versus commercial content (negative premium), which provides a nuanced view of how the platform ecosystem has evolved.

The methodology is robust, utilizing mixed-effects regression to control for baseline channel popularity and within-channel pairwise comparisons with FDR correction. The visual presentation of the data (particularly the TikZ pipeline diagram and the well-formatted results charts) is excellent and publication-ready.

## 3. Strengths

- **Novel Methodology:** The application of multimodal weak supervision (combining text and vision signals) to operationalize subjective ethical concepts is highly innovative. It provides a blueprint for scaling content moderation research beyond massive manual annotation.
- **Nuanced Findings:** The discovery of the "commercial penalty" ($-32.5\%$) alongside the "performativity premium" is a significant contribution. It moves the discourse beyond simple "more exploitation = more views" to a sophisticated understanding of how audience preferences and platform policies interact to reward specific forms of labor.
- **Robust Statistical Analysis:** The use of mixed-effects modeling to isolate within-channel variance, coupled with the same-year robustness check, provides strong evidence that the engagement premium is a structural feature of the content, not just an artifact of popular channels or video age.
- **Strong Policy Relevance:** The discussion section effectively connects the empirical findings to current legislative efforts (e.g., Illinois PA 103-0556), demonstrating that financial trusts are insufficient if the platform systematically incentivizes intensive labor.

## 4. Weaknesses and Areas for Improvement

While the paper is strong, there are a few areas that should be addressed before final publication:

- **Validation Protocol Limitations:** The pipeline validation relies on a single human annotator ($n=50$). Given the highly subjective nature of dimensions like "emotional bait" and "narrative conflict," the lack of inter-rater reliability (IRR) metrics is a notable weakness. While the authors acknowledge this in the limitations, they should provide more detail on the annotator's training and the specific codebook used to mitigate subjective bias.
- **Weighting Heuristic:** The aggregation weights (0.67 for VLM, 0.33 for LLM) are presented as a "structural prior." While the sensitivity analysis confirms the robustness of the final correlation, the paper would benefit from a brief explanation of why these specific values were chosen (e.g., were they tuned on a small holdout set, or purely theoretical?).
- **Disconnect with Earlier Findings:** The paper argues for an engagement premium associated with emotional bait and narrative conflict. However, it is unclear if these dimensions are merely correlated with performative labor. The mixed-effects model results (Table 2) show that narrative conflict loses significance in the joint model. The authors should clarify whether emotional exploitation independently drives views, or if the "performativity premium" is the primary driver.
- **Channel Selection Criteria:** The paper states channels were selected based on "prior literature and popular influencer lists." A more systematic description of the inclusion/exclusion criteria would strengthen the reproducibility of the study.

## 5. Specific Questions for the Authors

1. Could you elaborate on the human annotation process? What steps were taken to ensure the single annotator remained objective, particularly for borderline cases of "staged drama" versus organic vlogging?
2. In the joint mixed-effects model (Table 2), narrative conflict and challenge formats lose significance. Does this suggest that performative labor is the underlying latent variable driving the engagement premium?
3. How might the recent enforcement of the FTC's COPPA settlement on YouTube specifically explain the negative premium for commercial content? Could you expand on this mechanism in the discussion?

## 6. Conclusion

This paper makes a substantial methodological and empirical contribution to the study of algorithmic incentives and child safety. By operationalizing complex ethical frameworks into scalable computational metrics, the authors provide critical evidence for the ongoing policy debate surrounding child digital labor. I strongly recommend acceptance, provided the authors address the minor concerns regarding the validation protocol and clarify the relationships between the specific exploitation dimensions.
