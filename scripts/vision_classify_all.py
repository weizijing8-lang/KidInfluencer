"""
GPT-4.1-mini Vision Classifier for Kid Influencer Exploitation Detection
=========================================================================
Analyzes YouTube video thumbnails + titles + descriptions to score exploitation
dimensions using multimodal LLM.

Features:
- Concurrent API calls (configurable workers)
- Checkpoint/resume support (saves progress every N videos)
- Rate limiting with exponential backoff
- Low-detail image mode for cost efficiency
- CSV + JSON output

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."
    
    # Run on all videos (downloads thumbnails automatically)
    python3 scripts/vision_classify_all.py
    
    # Run on a subset (e.g., only Snorkel-ambiguous videos)
    python3 scripts/vision_classify_all.py --score-range 0.3 0.7
    
    # Resume from checkpoint
    python3 scripts/vision_classify_all.py --resume

Estimated cost: ~$2-3 for 2,306 videos (low detail mode)
Estimated time: ~30-40 minutes with 10 concurrent workers
"""

import argparse
import base64
import json
import os
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbnails"
DESC_DIR = DATA_DIR / "descriptions"
OUTPUT_DIR = DATA_DIR / "vision_results"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# API settings
MAX_WORKERS = 10          # Concurrent API calls
RATE_LIMIT_RPM = 500      # Requests per minute limit
CHECKPOINT_EVERY = 50     # Save progress every N videos
MAX_RETRIES = 3           # Max retries per video
RETRY_BACKOFF = 2.0       # Exponential backoff base

# Image settings
IMAGE_DETAIL = "low"      # "low" (~1,100 tokens) or "high" (~5,600 tokens)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "vision_classify.log" if OUTPUT_DIR.exists() else "/tmp/vision_classify.log")
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# PROMPT
# ============================================================

ANALYSIS_PROMPT = """You are analyzing a YouTube video thumbnail and title to assess potential child exploitation signals. The video is from a family/kid YouTube channel.

VIDEO TITLE: "{title}"
VIDEO DESCRIPTION (first 300 chars): "{description}"

Analyze the thumbnail image and title together. For each dimension below, provide a score from 0.0 to 1.0:

1. **performative_labor** (0.0-1.0): Is the child performing scripted/staged content created specifically for the camera? (e.g., acting in a skit, doing a manufactured challenge, wearing costumes for a role). Score HIGH if the activity would NOT happen without the camera. Score LOW if it's a natural activity (vlog, travel, daily routine, organic play).

2. **emotional_bait** (0.0-1.0): Does the thumbnail/title use exaggerated emotions to attract clicks? Look for: children with mouths wide open in shock, crying faces, fear expressions, ALL CAPS emotional words, clickbait phrasing like "you won't believe", "scary", "shocking". Also consider if the thumbnail is designed to provoke curiosity or emotional reaction.

3. **narrative_conflict** (0.0-1.0): Does the content manufacture drama or conflict? (e.g., pranks, "caught doing X", family arguments, competitions with clear winners/losers, "cops come to house", betrayal narratives)

4. **challenge_format** (0.0-1.0): Is this a challenge/dare/competition format? (e.g., "24 hours", "last to leave", "buying everything in your color", timed challenges)

5. **commercial_content** (0.0-1.0): Is there visible product placement, unboxing, brand logos, or sponsored content signals in the thumbnail? Are there shopping links mentioned?

6. **privacy_violation** (0.0-1.0): Does the content expose the child's private moments, body, medical situations, embarrassing moments, or intimate relationships that a child might not want publicly shared?

7. **overall_exploitative** (0.0-1.0): Considering ALL dimensions together, is this video likely exploitative of the child? A video is exploitative if the child is being used primarily as a content-generation tool rather than being the genuine subject of family documentation.

Return ONLY valid JSON (no markdown, no explanation outside JSON):
{{"performative_labor": float, "emotional_bait": float, "narrative_conflict": float, "challenge_format": float, "commercial_content": float, "privacy_violation": float, "overall_exploitative": float, "reasoning": "one sentence"}}"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_thumbnail(video_id: str, thumb_dir: Path) -> Optional[Path]:
    """Download YouTube thumbnail if not already cached."""
    thumb_path = thumb_dir / f"{video_id}.jpg"
    if thumb_path.exists() and thumb_path.stat().st_size > 1000:
        return thumb_path
    
    for quality in ['maxresdefault', 'hqdefault', 'mqdefault']:
        url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 2000:
                thumb_path.write_bytes(resp.content)
                return thumb_path
        except requests.RequestException:
            continue
    return None


def load_description(video_id: str, channel: str, desc_dir: Path) -> str:
    """Load video description from cached JSON files."""
    desc_file = desc_dir / f"{channel}_desc.json"
    if desc_file.exists():
        try:
            with open(desc_file) as f:
                descs = json.load(f)
            return descs.get(video_id, {}).get('description', '')[:300]
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def analyze_video(client: OpenAI, video_id: str, title: str, description: str, 
                  thumb_path: Path, detail: str = "low") -> dict:
    """Call GPT-4.1-mini Vision API to analyze a single video."""
    
    # Encode thumbnail
    img_b64 = base64.b64encode(thumb_path.read_bytes()).decode()
    
    prompt = ANALYSIS_PROMPT.format(title=title, description=description)
    
    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {
                    'url': f'data:image/jpeg;base64,{img_b64}',
                    'detail': detail
                }}
            ]
        }],
        max_tokens=300,
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    
    # Parse JSON (handle markdown code blocks)
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    
    result = json.loads(content)
    result['video_id'] = video_id
    result['title'] = title
    
    return result


def process_video_with_retry(client: OpenAI, video_id: str, title: str, 
                             channel: str, thumb_dir: Path, desc_dir: Path,
                             detail: str = "low", max_retries: int = 3) -> dict:
    """Process a single video with retry logic."""
    
    # Download thumbnail
    thumb_path = download_thumbnail(video_id, thumb_dir)
    if thumb_path is None:
        return {'video_id': video_id, 'title': title, 'error': 'thumbnail_download_failed'}
    
    # Load description
    description = load_description(video_id, channel, desc_dir)
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            result = analyze_video(client, video_id, title, description, thumb_path, detail)
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error for {video_id} (attempt {attempt+1}): {e}")
        except Exception as e:
            wait_time = RETRY_BACKOFF ** attempt
            logger.warning(f"  API error for {video_id} (attempt {attempt+1}): {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
    
    return {'video_id': video_id, 'title': title, 'error': f'failed_after_{max_retries}_retries'}


def load_checkpoint(checkpoint_file: Path) -> set:
    """Load set of already-processed video IDs."""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            data = json.load(f)
        return set(data.get('completed', []))
    return set()


def save_checkpoint(checkpoint_file: Path, completed: set):
    """Save checkpoint with completed video IDs."""
    with open(checkpoint_file, 'w') as f:
        json.dump({'completed': list(completed), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-4 Vision exploitation classifier")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--score-range', nargs=2, type=float, default=None,
                        help='Only process videos with Snorkel score in this range (e.g., 0.3 0.7)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of concurrent workers')
    parser.add_argument('--detail', choices=['low', 'high'], default=IMAGE_DETAIL, help='Image detail level')
    parser.add_argument('--limit', type=int, default=None, help='Max videos to process (for testing)')
    parser.add_argument('--input-csv', type=str, default=None, help='Custom input CSV path')
    args = parser.parse_args()
    
    # Setup directories
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load video data
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        # Try to load the stratified sample with Snorkel scores
        sample_path = DATA_DIR / "stratified_sample_v2.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
        else:
            # Fallback to full dataset
            df = pd.read_csv(DATA_DIR / "full_expanded_dataset.csv")
    
    logger.info(f"Loaded {len(df)} videos")
    
    # Ensure required columns exist
    if 'id' in df.columns and 'video_id' not in df.columns:
        df = df.rename(columns={'id': 'video_id'})
    if 'channelId' in df.columns and 'channel_short_name' not in df.columns:
        # Try to map channelId to short name
        pass
    
    # Filter by score range if specified
    if args.score_range and 'exploitation_score' in df.columns:
        low, high = args.score_range
        df = df[(df['exploitation_score'] >= low) & (df['exploitation_score'] <= high)]
        logger.info(f"Filtered to {len(df)} videos in score range [{low}, {high}]")
    
    # Resume from checkpoint
    completed = set()
    if args.resume:
        completed = load_checkpoint(CHECKPOINT_FILE)
        logger.info(f"Resuming: {len(completed)} videos already completed")
    
    # Filter out already completed
    df = df[~df['video_id'].isin(completed)]
    
    # Apply limit
    if args.limit:
        df = df.head(args.limit)
    
    logger.info(f"Processing {len(df)} videos with {args.workers} workers (detail={args.detail})")
    
    # Process videos
    results = []
    errors = []
    start_time = time.time()
    
    # Rate limiting: track requests per minute
    request_times = []
    
    def rate_limited_process(row):
        """Process with rate limiting."""
        nonlocal request_times
        
        # Simple rate limiting
        now = time.time()
        request_times = [t for t in request_times if now - t < 60]
        if len(request_times) >= RATE_LIMIT_RPM:
            sleep_time = 60 - (now - request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        request_times.append(time.time())
        
        channel = row.get('channel_short_name', row.get('channelId', ''))
        return process_video_with_retry(
            client, row['video_id'], row['title'], channel,
            THUMB_DIR, DESC_DIR, args.detail, MAX_RETRIES
        )
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, row in df.iterrows():
            future = executor.submit(rate_limited_process, row.to_dict())
            futures[future] = row['video_id']
        
        for i, future in enumerate(as_completed(futures)):
            vid = futures[future]
            try:
                result = future.result()
                if 'error' in result:
                    errors.append(result)
                    logger.warning(f"  [{i+1}/{len(df)}] ❌ {vid}: {result['error']}")
                else:
                    results.append(result)
                    completed.add(vid)
                    logger.info(f"  [{i+1}/{len(df)}] ✅ {result['title'][:40]}  overall={result['overall_exploitative']:.2f}")
            except Exception as e:
                errors.append({'video_id': vid, 'error': str(e)})
                logger.error(f"  [{i+1}/{len(df)}] ❌ {vid}: {e}")
            
            # Checkpoint
            if (i + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(CHECKPOINT_FILE, completed)
                logger.info(f"  💾 Checkpoint saved ({len(completed)} completed)")
    
    # Final save
    elapsed = time.time() - start_time
    
    # Save results
    output_json = OUTPUT_DIR / "vision_classifications.json"
    output_csv = OUTPUT_DIR / "vision_classifications.csv"
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
    
    # Save errors
    if errors:
        errors_path = OUTPUT_DIR / "vision_errors.json"
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
    
    # Final checkpoint
    save_checkpoint(CHECKPOINT_FILE, completed)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE!")
    logger.info(f"  Processed: {len(results)} videos")
    logger.info(f"  Errors: {len(errors)} videos")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s per video)")
    logger.info(f"  Output: {output_csv}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
