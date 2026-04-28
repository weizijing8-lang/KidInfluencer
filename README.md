# The Algorithmic Exploitation Ratchet

**Quantifying How Engagement Rewards Drive Content Escalation in Family Vlog Channels**

## Overview

This project provides the first quantitative, causal evidence that algorithmic engagement rewards act as a "ratchet mechanism," driving family vlog channels toward increasingly exploitative content involving children — an effect absent in adult-only creator channels.

## Full Dataset (April 2026)

Collected via YouTube Data API v3:

| Group | Channels | Videos | Description |
|-------|----------|--------|-------------|
| Family (Treatment) | 25 | 41,159 | Family vlog channels featuring children |
| Adult (Control) | 41 | 57,457 | Adult-only vloggers (no children) |
| **Total** | **66** | **98,616** | Full metadata including view/like/comment counts |

### Top Channels by Views
| Channel | Type | Videos | Total Views |
|---------|------|--------|-------------|
| Cocomelon | Family | 1,983 | 220B |
| Vlad and Niki | Family | 1,014 | 120B |
| MrBeast | Adult (control) | 970 | 119B |
| Ryan's World | Family | 3,685 | 63B |
| PewDiePie | Adult (control) | 4,617 | 29B |

## Pilot Study Results

Our pilot study on 6 YouTube channels (4 family vlogs + 2 adult-only controls) demonstrates that an embedding-based **Exploitation Drift Score** can effectively distinguish content types:

| Channel | Type | Mean Drift Score |
|---------|------|------------------|
| The ACE Family | Family (controversial) | **0.114** |
| Casey Neistat | Control (adult vlog) | 0.061 |
| Bratayley | Family | 0.016 |
| Family Fun Pack | Family | 0.007 |
| Ryan's World | Family (kid-focused) | -0.004 |
| Mark Wiens | Control (food vlog) | **-0.081** |

## Methodology

1. **Diachronic Embedding:** Map video titles to 384-dim vectors via `sentence-transformers`.
2. **Exploitation Direction Vector:** Define an unsupervised "exploitation direction" using known extreme cases (8 Passengers, DaddyOFive) as anchors.
3. **Drift Score:** Project each video's embedding onto the exploitation direction.
4. **Causal Inference (DiD/ITS):** Test whether "viral hits" causally drive subsequent content toward the exploitation direction.

## Repository Structure

```
├── README.md
├── docs/                          # Research plans and literature review
│   ├── kidfluencer_execution_plan.md    # Full execution plan with timeline
│   ├── kidfluencer_research_report.md   # Literature review and gap analysis
│   ├── kidfluencer_deep_research.md     # Deep research notes
│   ├── kidfluencer_data_plan.md         # Data collection strategy
│   └── kidfluencer_data_feasibility.md  # API feasibility assessment
├── scripts/                       # Data collection and analysis scripts
│   ├── channel_list.py                  # Full channel list (96 channels)
│   ├── collect_youtube_api.py           # YouTube Data API v3 collection
│   ├── collect_all.py                   # Legacy yt-dlp collection (pilot)
│   ├── analyze_drift.py                 # Embedding + drift score computation
│   ├── validate_data.py                 # Data validation & summary stats
│   └── visualize.py                     # Visualization of results
├── data/                          # Full study data (98,616 videos)
│   ├── raw/                             # Per-channel JSON (66 channels)
│   ├── all_videos.csv                   # Unified dataset
│   └── channel_statistics.csv           # Per-channel summary
├── pilot/                         # Pilot study data and results
│   ├── data/                            # Raw JSON data from YouTube
│   ├── results/                         # Drift scores (CSV) and summary
│   └── figures/                         # Visualization outputs
```

## Target Venue

**ICWSM 2027** (International Conference on Web and Social Media)
- Round 1 Deadline: May 15, 2026
- Round 2 Deadline: September 15, 2026

## Requirements

```
pip install sentence-transformers pandas numpy matplotlib seaborn requests
```

## License

MIT
