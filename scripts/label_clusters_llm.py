"""
Use GPT-4 to label each of the 15 clusters with:
- A human-readable content category name
- A manipulation risk level (low/medium/high)
- A brief description of the content pattern
- Key exploitation concerns
"""
import json, os
from openai import OpenAI

client = OpenAI()

# Load cluster info
with open('analysis_discovery/cluster_info.json') as f:
    clusters = json.load(f)

# Build prompt for each cluster
def label_cluster(cluster_info):
    prompt = f"""You are an expert in child online safety and content analysis. 
Analyze this cluster of YouTube videos from kidfluencer (child influencer) channels.

CLUSTER DATA:
- Number of videos: {cluster_info['n_videos']}
- Percentage of dataset: {cluster_info['pct']:.1%}
- Median views: {cluster_info['median_views']:,.0f}
- View boost vs overall median: {cluster_info['view_boost']*100:+.0f}%
- Top words: {', '.join(cluster_info['top_words'])}
- Top channels: {', '.join(cluster_info['top_channels'])}
- Representative titles:
{chr(10).join('  - ' + t for t in cluster_info['representative_titles'])}

Based on this data, provide a JSON response with these fields:
1. "category_name": A concise 2-5 word label for this content category (e.g., "Toy Unboxing & Review", "Challenge & Dare Content", "Family Drama & Conflict")
2. "manipulation_risk": One of "low", "medium", "high" — how likely this content pattern involves emotional manipulation, staged conflict, or exploitation of children
3. "description": A 1-2 sentence description of what this cluster represents
4. "exploitation_concerns": A list of 1-3 specific concerns about child exploitation or manipulation in this content type (or empty list if minimal concerns)
5. "commercialization_level": One of "low", "medium", "high" — how commercially driven this content appears

Return ONLY valid JSON, no markdown formatting."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# Process all clusters
print("Labeling clusters with GPT-4...")
labeled_clusters = []

for i, cluster in enumerate(clusters):
    print(f"  Cluster {cluster['cluster']} ({i+1}/15)...")
    try:
        response_text = label_cluster(cluster)
        # Parse JSON response
        # Strip any markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
            response_text = response_text.rsplit('```', 1)[0]
        
        label_data = json.loads(response_text)
        label_data['cluster_id'] = cluster['cluster']
        label_data['n_videos'] = cluster['n_videos']
        label_data['view_boost_pct'] = round(cluster['view_boost'] * 100, 1)
        label_data['median_views'] = cluster['median_views']
        labeled_clusters.append(label_data)
        print(f"    → {label_data['category_name']} (risk: {label_data['manipulation_risk']}, boost: {label_data['view_boost_pct']:+.0f}%)")
    except Exception as e:
        print(f"    ERROR: {e}")
        # Fallback
        labeled_clusters.append({
            'cluster_id': cluster['cluster'],
            'category_name': f"Cluster {cluster['cluster']}",
            'manipulation_risk': 'unknown',
            'description': 'Error in labeling',
            'exploitation_concerns': [],
            'commercialization_level': 'unknown',
            'n_videos': cluster['n_videos'],
            'view_boost_pct': round(cluster['view_boost'] * 100, 1),
            'median_views': cluster['median_views']
        })

# Save results
output_path = 'analysis_discovery/cluster_labels_llm.json'
with open(output_path, 'w') as f:
    json.dump(labeled_clusters, f, indent=2)

print(f"\nSaved to {output_path}")

# Print summary table
print(f"\n{'='*80}")
print(f"{'ID':>3} {'Category':<30} {'Risk':<8} {'Comm':<8} {'Boost':>8} {'N':>6}")
print(f"{'='*80}")
for lc in sorted(labeled_clusters, key=lambda x: x['view_boost_pct'], reverse=True):
    print(f"{lc['cluster_id']:>3} {lc['category_name']:<30} {lc['manipulation_risk']:<8} {lc['commercialization_level']:<8} {lc['view_boost_pct']:>+7.0f}% {lc['n_videos']:>6}")
