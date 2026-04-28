# Data Feasibility & Collection Plan: Kidfluencer Exploitation Research

## 1. Feasibility Assessment: Highly Feasible

Based on initial API testing, constructing a comprehensive dataset of kidfluencer content is entirely feasible without manual scraping. The YouTube API provides robust access to channel metadata, historical video performance, and engagement metrics [1].

The Netflix documentary you mentioned is likely **"Bad Influence: The Dark Side of Kidfluencing"** (released April 2025) [2], which focuses on Piper Rockelle and "The Squad," or the Hulu docuseries **"Born to Be Viral: The Real Lives of Kidfluencers"** (June 2025) [3]. These documentaries, along with high-profile criminal cases like Ruby Franke (8 Passengers) [4], provide excellent qualitative grounding for our quantitative approach.

## 2. Data Collection Architecture

To execute the causal inference study (Idea 1), we need longitudinal data. We can build an automated pipeline using the official YouTube Data API v3 (or the `data_api` client in the sandbox).

### Phase A: Seed Channel Identification
We will compile a seed list of 50-100 major family vlog and kidfluencer channels. Our initial tests successfully retrieved data for massive channels like *Ryan's World* (40.3M subs) and *The ACE Family* (18M subs). 

The seed list will be stratified into:
- **Top-tier commercial kidfluencers** (e.g., Ryan's World, Like Nastya)
- **Family Vlogs** (e.g., The ACE Family, JesssFam)
- **Controversial/Terminated channels** (e.g., 8 Passengers, DaddyOFive - data may need to be sourced from web archives or third-party tracking sites like SocialBlade)

### Phase B: Longitudinal Video Metadata Collection
For each channel, we will paginate through their entire upload history to collect:
- `video_id`, `published_date`, `duration`
- `view_count`, `like_count`, `comment_count`
- `title`, `description`, `tags`

### Phase C: NLP Feature Extraction (The "Exploitation Signal")
This is where your NLP expertise comes in. We will use the video titles and descriptions to extract features that proxy "exploitation intensity." 
- **Sentiment/Toxicity:** Are titles becoming more extreme/dramatic over time?
- **Clickbait Classification:** Using zero-shot LLM prompts to classify if the title relies on the child's distress (e.g., "PUNISHING OUR KIDS," "EMERGENCY ROOM TRIP").
- **Topic Modeling:** Tracking the shift from mundane topics (baking, toys) to high-stakes drama.

## 3. The Causal Inference Framework (Idea 1)

With the longitudinal dataset, we can model the algorithm-driven exploitation loop:

1. **Treatment ($T$):** Algorithmic reward (a spike in views/engagement on a specific video).
2. **Outcome ($Y$):** The "exploitation intensity" (NLP score) of subsequent videos uploaded by the channel.
3. **Confounders ($X$):** Channel age, baseline subscriber count, seasonal trends.

By applying Granger Causality or Difference-in-Differences (DiD) on this time-series data, we can quantitatively prove whether algorithmic rewards cause parents to escalate the extremity of content involving their children.

## 4. Next Steps & Timeline

1. **Week 1: Data Pipeline Construction**
   - Finalize the seed list of 100 channels.
   - Write the Python script to pull the full historical catalog for these channels via API.
2. **Week 2: NLP Feature Engineering**
   - Run LLM-based classification on the titles/descriptions to score the "drama/exploitation" level.
3. **Week 3: Causal Modeling**
   - Apply the causal inference models to the time-series data.
4. **Week 4: Paper Drafting**
   - Draft the paper targeting WWW or ICWSM.

This project perfectly bridges your NLP background with a high-impact social issue, completely sidesteps any Meta affiliation issues, and requires no manual data labeling.

## References
[1] YouTube API Test Results. Internal sandbox execution logs.
[2] Netflix. "Bad Influence: The Dark Side of Kidfluencing". 2025.
[3] Hulu. "Born to Be Viral: The Real Lives of Kidfluencers". 2025.
[4] Various News Outlets. Coverage of Ruby Franke / 8 Passengers conviction. 2024.
