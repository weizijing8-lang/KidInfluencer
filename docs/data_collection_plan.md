# Data Collection Plan: Kidfluencer Ecosystem & Risk Identification

This document outlines the systematic data collection strategy for our two interconnected papers on the YouTube kidfluencer ecosystem. The goal is to build a comprehensive dataset that supports both descriptive analysis (Paper 1) and risk prediction modeling (Paper 2).

## 1. Dataset Construction Strategy

Our target population is **kidfluencer/family vlog channels**—channels where real children are the primary focus of the content. We must explicitly exclude animated kids' content (e.g., Cocomelon, Peppa Pig) and gaming channels that do not feature real children on camera.

### 1.1 Seed Channel Identification
To build a robust and representative dataset, we will use a multi-source snowball sampling approach:

*   **Academic Seed Set:** Leverage the dataset from Kissgen et al. (2023), which identified 388 confirmed child influencer channels through overlapping community detection.
*   **Industry Lists:** Scrape top lists from SocialBlade (filtering for `made_for_kids=true` and manually verifying) and Feedspot's "Top 100 Family YouTubers".
*   **Ground Truth "Positive" Set:** Explicitly include known exploitation cases (e.g., 8 Passengers, DaddyOFive, Fantastic Adventures) to serve as the positive class for our risk prediction model in Paper 2.

### 1.2 Snowball Sampling via API
Using the initial seed channels, we will expand our dataset by querying the YouTube Data API for "Related Channels" or by analyzing channels frequently recommended alongside our seed set. We aim for a final dataset of **500–1,000 verified kidfluencer channels**.

---

## 2. Data Collection Pipeline (via YouTube Data API v3)

We will utilize the `Youtube/get_channel_details` and `Youtube/get_channel_videos` API endpoints to gather data at both the channel and video levels.

### 2.1 Channel-Level Data
For each verified channel, we will extract:
*   **Basic Metadata:** `channelId`, `title`, `description`, `country`, `joinedDate`
*   **Performance Statistics:** `subscribers`, `total_videos`, `total_views`
*   **Network/Commercial Indicators:** `links` (to detect cross-platform presence like Instagram/TikTok), `keywords`/`tags`

### 2.2 Video-Level Data
To understand the "labor intensity" and content strategy, we will sample the **latest 50-100 videos** per channel using the `videos_latest` filter.
*   **Basic Metadata:** `videoId`, `title`, `publishedTimeText`, `lengthSeconds`
*   **Performance Metrics:** `views` (and `likes`/`comments` if available via extended API calls)
*   **Content Indicators:** `descriptionSnippet` (to check for sponsor links or affiliate codes), `isLiveNow`

---

## 3. Derived Metrics & Feature Engineering

The raw API data will be transformed into structured features to test our hypotheses regarding labor intensity (Paper 1) and risk profiles (Paper 2).

### 3.1 Structural & Labor Intensity Metrics (Paper 1)
*   **Upload Frequency:** Average number of videos published per week/month over the sampled period.
*   **Content Volume:** Total video duration (in minutes) produced per month.
*   **Commercial Density:** Binary or count metric indicating the presence of promotional terms (e.g., "ad", "sponsor", affiliate links) in video descriptions.
*   **Cross-Platform Synchronization:** Boolean flag indicating if the channel actively links to other social media profiles.

### 3.2 Risk & Emotional Manipulation Indicators (Paper 2)
*   **Clickbait/Emotional Titles:** Keyword matching on video titles for high-arousal or negative emotional words (e.g., "crying", "punishment", "hospital", "secret").
*   **Comment Disablement Rate:** Percentage of recent videos where comments are disabled (often a platform-enforced safety measure for high-risk child content).
*   **Time-to-Monetization:** Gap between the channel's `joinedDate` and the first highly commercialized video (requires historical sampling).

---

## 4. Ethical Considerations & Data Management

*   **Privacy:** All data collected is publicly available via the YouTube API. We will not collect or store Personally Identifiable Information (PII) of the children beyond what is publicly broadcasted by the channel owners.
*   **Storage:** Raw JSON responses from the API will be stored securely in local storage, with structured CSVs generated for analysis.
*   **Rate Limiting:** Data collection scripts will include appropriate delays and cursor pagination handling to respect YouTube API quota limits.
