"""
Run GPT-4.1-mini vision on all 23 annotated video thumbnails.
Analyze each thumbnail + title for ALL exploitation dimensions.
"""
import base64, json, os, time
from openai import OpenAI
import pandas as pd

client = OpenAI()

# Load user annotations for titles
user = pd.read_csv('/home/ubuntu/upload/pasted_content.txt', sep='\t')

# Load descriptions
with open('/home/ubuntu/KidInfluencer/data/descriptions/annotated_23_descriptions.json') as f:
    descriptions = json.load(f)

THUMB_DIR = "/home/ubuntu/KidInfluencer/thumbnails_test"

PROMPT = """You are analyzing a YouTube video thumbnail and title to assess potential child exploitation signals. The video is from a family/kid YouTube channel.

VIDEO TITLE: "{title}"
VIDEO DESCRIPTION (first 200 chars): "{description}"

Analyze the thumbnail image and title together. For each dimension below, provide a score from 0.0 to 1.0:

1. **performative_labor**: Is the child performing scripted/staged content created specifically for the camera? (e.g., acting in a skit, doing a manufactured challenge, wearing costumes for a role). Score HIGH if the activity would NOT happen without the camera. Score LOW if it's a natural activity (vlog, travel, daily routine).

2. **emotional_bait**: Does the thumbnail/title use exaggerated emotions to attract clicks? Look for: children with mouths wide open in shock, crying faces, fear expressions, ALL CAPS emotional words, clickbait phrasing like "you won't believe", "scary", "shocking".

3. **narrative_conflict**: Does the content manufacture drama or conflict? (e.g., pranks, "caught doing X", family arguments, competitions with winners/losers, "cops come to house")

4. **challenge_format**: Is this a challenge/dare/competition format? (e.g., "24 hours", "last to leave", "buying everything in your color")

5. **commercial_content**: Is there visible product placement, unboxing, or sponsored content signals in the thumbnail?

6. **privacy_violation**: Does the content expose the child's private moments, body, medical situations, or intimate relationships that a child might not want publicly shared?

7. **overall_exploitative**: Considering ALL dimensions together, is this video likely exploitative of the child? A video is exploitative if the child is being used primarily as a content-generation tool rather than being the genuine subject of family documentation.

Return ONLY valid JSON with these exact fields (all floats 0.0-1.0):
{{
  "performative_labor": float,
  "emotional_bait": float,
  "narrative_conflict": float,
  "challenge_format": float,
  "commercial_content": float,
  "privacy_violation": float,
  "overall_exploitative": float,
  "reasoning": "brief explanation"
}}"""

results = []

for idx, row in user.iterrows():
    vid = row['video_id']
    title = row['title']
    desc = descriptions.get(vid, {}).get('description', '')[:200]
    
    thumb_path = os.path.join(THUMB_DIR, f"{vid}.jpg")
    if not os.path.exists(thumb_path):
        print(f"  ❌ {vid} - no thumbnail")
        continue
    
    # Encode image
    with open(thumb_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    prompt_filled = PROMPT.format(title=title, description=desc)
    
    try:
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt_filled},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}}
                ]
            }],
            max_tokens=400,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        result = json.loads(content)
        result['video_id'] = vid
        result['title'] = title
        results.append(result)
        
        print(f"  ✅ [{idx+1}/23] {title[:45]}  →  overall={result['overall_exploitative']:.2f}")
        
    except Exception as e:
        print(f"  ❌ [{idx+1}/23] {vid}: {str(e)[:80]}")
        results.append({
            'video_id': vid, 'title': title,
            'performative_labor': -1, 'emotional_bait': -1,
            'narrative_conflict': -1, 'challenge_format': -1,
            'commercial_content': -1, 'privacy_violation': -1,
            'overall_exploitative': -1, 'reasoning': f'ERROR: {str(e)[:100]}'
        })
    
    time.sleep(0.5)  # Rate limiting

# Save results
output_path = '/home/ubuntu/KidInfluencer/data/vision_analysis_23.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Done! Saved {len(results)} results to {output_path}")
print(f"{'='*70}")

# Quick summary
for r in results:
    if r.get('overall_exploitative', -1) >= 0:
        print(f"  {r['title'][:40]:<42} overall={r['overall_exploitative']:.2f}  perf={r['performative_labor']:.2f}  emo={r['emotional_bait']:.2f}")
