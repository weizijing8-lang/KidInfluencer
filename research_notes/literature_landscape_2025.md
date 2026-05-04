# Literature Landscape Analysis - May 2025

## What Already Exists (Metadata-based Detection)

### 1. Samba (Le et al., CIKM 2022)
- Task: Detecting inappropriate videos FOR children (content safety)
- Method: Metadata + subtitles fusion model
- Dataset: 70K videos
- Key finding: Metadata-only achieves ~88%, adding subtitles gets to 95%
- Features: thumbnails, title, comments, video metadata
- DIFFERENT from our work: They detect unsafe content for kids to WATCH, not exploitation OF kids

### 2. Safeguarding Children at Scale (ACM 2026)
- Task: Detecting inappropriate ADS shown on child-oriented YouTube content
- Method: Multimodal LLM (GPT-4V etc.) with ablation studies
- Key finding: Metadata-only yields low F1; full video analysis too expensive; their DAVSP frame sampling reduces cost 21.4x
- DIFFERENT from our work: They detect bad ads, not child labor

### 3. Detecting Child Objectification (Schirmer et al., WOAH/ACL 2025)
- Task: Detecting objectifying COMMENTS on child videos (TikTok)
- Method: NLP classification (RoBERTa, GPT-4, LLaMA, Mistral)
- Dataset: 562,508 comments from 9,090 videos across 482 TikTok accounts
- Key finding: RoBERTa best for detecting appearance/objectification language; 10.35% of comments contain appearance-related language
- DIFFERENT from our work: They analyze COMMENTS, not the video/channel behavior itself

### 4. KidsNanny (Panchal, 2026)
- Task: Two-stage multimodal content moderation for child safety
- Method: ViT + object detection + OCR + contextual reasoning
- DIFFERENT: Again about content safety FOR children, not exploitation OF children

### 5. LLM-Powered Nuanced Video Attribute Annotation (arxiv 2025)
- Task: Using LLMs for content annotation at scale
- Method: LLM as annotator for video attributes
- RELEVANT: Validates our LLM-as-annotator approach

### 6. How social media platforms manipulate kidinfluencers (Albuquerque et al., ACM 2023)
- Task: Analyzing deceptive design patterns targeting kidfluencers
- Method: Qualitative analysis of platform UX
- DIFFERENT: UX/design focus, not computational detection

## THE GAP WE FILL

Nobody has done:
1. **Detecting child LABOR indicators** (not content safety, not objectification in comments)
2. **From channel-level behavioral metadata** (upload patterns, duration distributions, title strategies)
3. **Using LLM + ML pipeline** to classify exploitation risk
4. **Comparing kidfluencer vs adult channels** on labor dimensions

The closest work (Schirmer 2025) looks at objectification in comments.
The closest system work (Samba 2022) looks at content safety for viewers.
NOBODY looks at whether the child IN the video is being exploited as a laborer.

## Positioning Our Paper

Our work sits at the intersection of:
- Algorithmic auditing (Sandvig et al. 2014; Bandy 2021)
- LLM-as-annotator reliability (Gilardi et al. 2023; Törnberg 2024)
- Child digital rights (UNCRC GC25, 2021)
- Platform labor studies (Duffy 2017; Cunningham & Craig 2019)

But with a UNIQUE detection target: child labor indicators from metadata.

## Key 2024-2025 References to Include
- Schirmer et al. 2025 (Child Objectification, WOAH/ACL)
- ACM 2026 (Safeguarding Children at Scale)
- KidsNanny 2026 (Multimodal moderation)
- Siewert 2024 (Child labor in kidfluencing, legal perspective)
- Ma et al. 2024 (Labeling in the Dark, DIS)
- Clark & Jno-Charles 2025 (Child labor ethics, J Business Ethics)
- Illinois PA 103-0556 (2024)
- France Loi Studer (2020, amended 2024)
