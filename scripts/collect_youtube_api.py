"""
YouTube Data API v3 - Full Video Metadata Collection Script
============================================================
Collects video metadata (title, view_count, like_count, comment_count, publish_date)
for all videos from specified YouTube channels.

Strategy:
1. Resolve channel handle → channel ID → uploads playlist ID
2. Paginate through playlistItems to get all video IDs
3. Batch video IDs (50 per request) to get statistics + snippet
4. Save per-channel JSON files with full metadata

Quota budget:
- channels.list: 1 unit per request
- playlistItems.list: 1 unit per request (50 items max)
- videos.list: 1 unit per request (50 IDs max)
- Free tier: 10,000 units/day

For ~100 channels averaging 1000 videos each:
- Channel resolution: 100 units
- PlaylistItems: 100 * (1000/50) = 2000 units
- Video details: 100 * (1000/50) = 2000 units
- Total: ~4100 units (well within daily quota)

Usage:
    python3 collect_youtube_api.py [--channels all|family|adult|pilot]
                                   [--output-dir /path/to/output]
                                   [--resume]
"""

import requests
import json
import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# YouTube Data API v3 key
API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://www.googleapis.com/youtube/v3"

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests (be polite)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Import channel list
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel_list import CHANNELS


class YouTubeCollector:
    def __init__(self, api_key, output_dir, verbose=True):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.quota_used = 0
        self.session = requests.Session()
    
    def _request(self, endpoint, params):
        """Make a YouTube API request with retry logic."""
        params["key"] = self.api_key
        url = f"{BASE_URL}/{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                response = self.session.get(url, params=params, timeout=30)
                self.quota_used += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    error_data = response.json()
                    error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "")
                    if error_reason == "quotaExceeded":
                        print(f"\n[ERROR] YouTube API quota exceeded! Used: {self.quota_used} units")
                        print("Please wait until tomorrow (Pacific Time midnight) for quota reset.")
                        sys.exit(1)
                    else:
                        print(f"\n[WARN] 403 error: {error_reason}. Retrying...")
                        time.sleep(RETRY_DELAY * (attempt + 1))
                elif response.status_code == 404:
                    return None
                else:
                    print(f"\n[WARN] HTTP {response.status_code}. Retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except requests.exceptions.RequestException as e:
                print(f"\n[WARN] Request error: {e}. Retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        print(f"\n[ERROR] Failed after {MAX_RETRIES} retries for {endpoint}")
        return None
    
    def resolve_channel(self, handle_or_id):
        """Resolve a channel handle to channel ID and uploads playlist ID."""
        # Try as handle first
        if handle_or_id.startswith("@"):
            params = {"part": "contentDetails,snippet", "forHandle": handle_or_id}
        else:
            params = {"part": "contentDetails,snippet", "id": handle_or_id}
        
        data = self._request("channels", params)
        if not data or not data.get("items"):
            return None, None, None
        
        item = data["items"][0]
        channel_id = item["id"]
        channel_title = item["snippet"]["title"]
        uploads_playlist = item["contentDetails"]["relatedPlaylists"]["uploads"]
        return channel_id, channel_title, uploads_playlist
    
    def get_all_video_ids(self, playlist_id):
        """Get all video IDs from an uploads playlist (paginated)."""
        video_ids = []
        page_token = None
        
        while True:
            params = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token
            
            data = self._request("playlistItems", params)
            if not data or not data.get("items"):
                break
            
            for item in data["items"]:
                vid_id = item["contentDetails"]["videoId"]
                video_ids.append(vid_id)
            
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        
        return video_ids
    
    def get_video_details(self, video_ids):
        """Get video details (snippet + statistics) for a batch of video IDs (max 50)."""
        if not video_ids:
            return []
        
        params = {
            "part": "snippet,statistics",
            "id": ",".join(video_ids),
            "maxResults": 50,
        }
        
        data = self._request("videos", params)
        if not data or not data.get("items"):
            return []
        
        videos = []
        for item in data["items"]:
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            
            videos.append({
                "id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", "")[:500],  # Truncate long descriptions
                "publishedAt": snippet.get("publishedAt", ""),
                "channelId": snippet.get("channelId", ""),
                "channelTitle": snippet.get("channelTitle", ""),
                "tags": snippet.get("tags", [])[:20],  # Limit tags
                "categoryId": snippet.get("categoryId", ""),
                "viewCount": int(stats.get("viewCount", 0)),
                "likeCount": int(stats.get("likeCount", 0)),
                "commentCount": int(stats.get("commentCount", 0)),
                "favoriteCount": int(stats.get("favoriteCount", 0)),
            })
        
        return videos
    
    def collect_channel(self, short_name, handle_or_id, category):
        """Collect all video data for a single channel."""
        output_file = self.output_dir / f"{short_name}.json"
        
        # Check if already collected (resume support)
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
            if existing.get("videos") and len(existing["videos"]) > 0:
                # Verify it has view counts
                if existing["videos"][0].get("viewCount", 0) > 0 or len(existing["videos"]) > 5:
                    if self.verbose:
                        print(f"  [SKIP] {short_name}: already collected ({len(existing['videos'])} videos)")
                    return existing
        
        if self.verbose:
            print(f"  [COLLECT] {short_name} ({handle_or_id})...", end="", flush=True)
        
        # Step 1: Resolve channel
        channel_id, channel_title, uploads_playlist = self.resolve_channel(handle_or_id)
        if not channel_id:
            print(f" FAILED (channel not found)")
            # Save empty result
            result = {
                "short_name": short_name,
                "handle": handle_or_id,
                "category": category,
                "channel_id": None,
                "channel_title": None,
                "error": "Channel not found",
                "collected_at": datetime.utcnow().isoformat(),
                "videos": []
            }
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            return result
        
        # Step 2: Get all video IDs
        video_ids = self.get_all_video_ids(uploads_playlist)
        if self.verbose:
            print(f" {len(video_ids)} videos found...", end="", flush=True)
        
        # Step 3: Batch get video details (50 at a time)
        all_videos = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            details = self.get_video_details(batch)
            all_videos.extend(details)
        
        if self.verbose:
            print(f" {len(all_videos)} collected. Quota used: {self.quota_used}")
        
        # Step 4: Save result
        result = {
            "short_name": short_name,
            "handle": handle_or_id,
            "category": category,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "uploads_playlist": uploads_playlist,
            "total_videos": len(all_videos),
            "collected_at": datetime.utcnow().isoformat(),
            "videos": all_videos
        }
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def collect_all(self, channel_filter="all", resume=True):
        """Collect data for all channels matching the filter."""
        if channel_filter == "all":
            channels = CHANNELS
        elif channel_filter == "family":
            channels = [c for c in CHANNELS if c[2] == "family"]
        elif channel_filter == "adult":
            channels = [c for c in CHANNELS if c[2] == "adult"]
        elif channel_filter == "pilot":
            pilot_names = ["acefamily", "ryansworld", "familyfunpack", "bratayley", "caseyneistat", "markwiens"]
            channels = [c for c in CHANNELS if c[0] in pilot_names]
        else:
            channels = CHANNELS
        
        print(f"=" * 60)
        print(f"YouTube Data Collection - {len(channels)} channels")
        print(f"Output: {self.output_dir}")
        print(f"Filter: {channel_filter}")
        print(f"=" * 60)
        
        results_summary = []
        
        for i, (short_name, handle, category) in enumerate(channels, 1):
            print(f"\n[{i}/{len(channels)}] {short_name} ({category})")
            result = self.collect_channel(short_name, handle, category)
            results_summary.append({
                "short_name": short_name,
                "category": category,
                "channel_title": result.get("channel_title", "N/A"),
                "total_videos": result.get("total_videos", 0),
                "error": result.get("error", None),
            })
        
        # Save summary
        summary_file = self.output_dir / "_collection_summary.json"
        summary = {
            "collected_at": datetime.utcnow().isoformat(),
            "filter": channel_filter,
            "total_channels": len(channels),
            "total_quota_used": self.quota_used,
            "channels": results_summary,
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE")
        print(f"{'=' * 60}")
        total_videos = sum(r["total_videos"] for r in results_summary)
        successful = sum(1 for r in results_summary if not r.get("error"))
        failed = sum(1 for r in results_summary if r.get("error"))
        print(f"Channels processed: {len(channels)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"Total videos collected: {total_videos}")
        print(f"Total API quota used: {self.quota_used} units")
        print(f"Summary saved to: {summary_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube video data via API v3")
    parser.add_argument("--channels", default="all", choices=["all", "family", "adult", "pilot"],
                       help="Which channels to collect (default: all)")
    parser.add_argument("--output-dir", default="/home/ubuntu/KidInfluencer/data/raw",
                       help="Output directory for JSON files")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Skip channels already collected (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Re-collect all channels even if data exists")
    args = parser.parse_args()
    
    collector = YouTubeCollector(
        api_key=API_KEY,
        output_dir=args.output_dir,
        verbose=True,
    )
    
    collector.collect_all(channel_filter=args.channels, resume=not args.no_resume)


if __name__ == "__main__":
    main()
