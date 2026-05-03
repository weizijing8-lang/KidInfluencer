"""
Title NLP Feature Extraction for Kidfluencer Study
- Sentiment analysis (VADER)
- Readability scores
- Linguistic patterns (caps ratio, punctuation, word count, etc.)
- Clickbait indicators
- Child-related keyword presence
"""

import pandas as pd
import numpy as np
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

def extract_nlp_features(title):
    """Extract NLP features from a single video title."""
    if not isinstance(title, str) or len(title.strip()) == 0:
        return {
            'title_length': 0, 'word_count': 0, 'caps_ratio': 0, 'caps_word_count': 0,
            'exclamation_count': 0, 'question_count': 0, 'emoji_count': 0,
            'has_ellipsis': False, 'has_numbers': False, 'num_count': 0,
            'sentiment_pos': 0, 'sentiment_neg': 0, 'sentiment_neu': 0, 'sentiment_compound': 0,
            'flesch_reading_ease': 0, 'has_challenge': False, 'has_prank': False,
            'has_surprise': False, 'has_emotional_word': False, 'has_family_word': False,
            'has_child_word': False, 'has_first_person': False, 'has_clickbait_phrase': False,
            'special_char_ratio': 0, 'avg_word_length': 0,
        }
    
    result = {}
    
    # --- Basic text stats ---
    result['title_length'] = len(title)
    words = title.split()
    result['word_count'] = len(words)
    result['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    # --- Capitalization ---
    alpha_chars = [c for c in title if c.isalpha()]
    result['caps_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
    result['caps_word_count'] = sum(1 for w in words if w.isupper() and len(w) > 1)
    
    # --- Punctuation ---
    result['exclamation_count'] = title.count('!')
    result['question_count'] = title.count('?')
    result['has_ellipsis'] = '...' in title or '…' in title
    result['special_char_ratio'] = sum(1 for c in title if not c.isalnum() and not c.isspace()) / len(title) if title else 0
    
    # --- Numbers ---
    numbers = re.findall(r'\d+', title)
    result['has_numbers'] = len(numbers) > 0
    result['num_count'] = len(numbers)
    
    # --- Emoji detection ---
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    result['emoji_count'] = len(emoji_pattern.findall(title))
    
    # --- Sentiment (VADER) ---
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(title)
    result['sentiment_pos'] = scores['pos']
    result['sentiment_neg'] = scores['neg']
    result['sentiment_neu'] = scores['neu']
    result['sentiment_compound'] = scores['compound']
    
    # --- Readability ---
    try:
        result['flesch_reading_ease'] = textstat.flesch_reading_ease(title)
    except:
        result['flesch_reading_ease'] = 0
    
    # --- Content keywords ---
    title_lower = title.lower()
    
    # Challenge/prank/surprise
    result['has_challenge'] = any(w in title_lower for w in ['challenge', 'dare', '24 hour', '24hr', 'hours'])
    result['has_prank'] = any(w in title_lower for w in ['prank', 'pranked', 'pranking'])
    result['has_surprise'] = any(w in title_lower for w in ['surprise', 'surprised', 'shocking', 'shocked', 'unexpected'])
    
    # Emotional words
    emotional_words = ['cry', 'crying', 'cried', 'tears', 'scream', 'screaming', 'angry', 'mad', 
                       'sad', 'heartbreak', 'emotional', 'devastated', 'scared', 'terrified',
                       'panic', 'emergency', 'hospital', 'hurt', 'pain']
    result['has_emotional_word'] = any(w in title_lower for w in emotional_words)
    
    # Family words
    family_words = ['family', 'mom', 'dad', 'parent', 'brother', 'sister', 'baby', 'pregnant',
                    'husband', 'wife', 'son', 'daughter']
    result['has_family_word'] = any(w in title_lower for w in family_words)
    
    # Child-specific words
    child_words = ['kid', 'kids', 'child', 'children', 'toddler', 'teen', 'teenager',
                   'school', 'homework', 'birthday', 'toy', 'toys', 'play', 'playground']
    result['has_child_word'] = any(w in title_lower for w in child_words)
    
    # First person
    result['has_first_person'] = any(w in title_lower.split() for w in ['i', 'my', 'we', 'our', 'me', 'us'])
    
    # Clickbait phrases
    clickbait_phrases = ["you won't believe", "gone wrong", "not clickbait", "must watch",
                         "what happened", "the truth", "never again", "life changing",
                         "biggest mistake", "can't believe", "finally revealed", "last day"]
    result['has_clickbait_phrase'] = any(p in title_lower for p in clickbait_phrases)
    
    return result


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("TITLE NLP FEATURE EXTRACTION", flush=True)
    print("=" * 60, flush=True)
    
    df = pd.read_csv('/home/ubuntu/KidInfluencer/data/combined_videos.csv')
    print(f"Processing {len(df)} video titles...", flush=True)
    
    # Extract features
    features_list = []
    for i, row in df.iterrows():
        features = extract_nlp_features(row['title'])
        features['video_id'] = row['video_id']
        features_list.append(features)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(df)}...", flush=True)
    
    nlp_df = pd.DataFrame(features_list)
    
    # Stats
    print(f"\n--- RESULTS ---", flush=True)
    print(f"Avg title length: {nlp_df['title_length'].mean():.1f} chars", flush=True)
    print(f"Avg word count: {nlp_df['word_count'].mean():.1f}", flush=True)
    print(f"Avg caps ratio: {nlp_df['caps_ratio'].mean():.3f}", flush=True)
    print(f"Titles with ALL CAPS words: {(nlp_df['caps_word_count']>0).sum()} ({(nlp_df['caps_word_count']>0).mean()*100:.1f}%)", flush=True)
    print(f"Titles with exclamation: {(nlp_df['exclamation_count']>0).sum()} ({(nlp_df['exclamation_count']>0).mean()*100:.1f}%)", flush=True)
    print(f"Titles with question: {(nlp_df['question_count']>0).sum()} ({(nlp_df['question_count']>0).mean()*100:.1f}%)", flush=True)
    print(f"Titles with emoji: {(nlp_df['emoji_count']>0).sum()} ({(nlp_df['emoji_count']>0).mean()*100:.1f}%)", flush=True)
    print(f"Avg sentiment compound: {nlp_df['sentiment_compound'].mean():.3f}", flush=True)
    print(f"Avg Flesch reading ease: {nlp_df['flesch_reading_ease'].mean():.1f}", flush=True)
    print(f"\nContent keywords:", flush=True)
    print(f"  Challenge: {nlp_df['has_challenge'].sum()} ({nlp_df['has_challenge'].mean()*100:.1f}%)", flush=True)
    print(f"  Prank: {nlp_df['has_prank'].sum()} ({nlp_df['has_prank'].mean()*100:.1f}%)", flush=True)
    print(f"  Surprise: {nlp_df['has_surprise'].sum()} ({nlp_df['has_surprise'].mean()*100:.1f}%)", flush=True)
    print(f"  Emotional: {nlp_df['has_emotional_word'].sum()} ({nlp_df['has_emotional_word'].mean()*100:.1f}%)", flush=True)
    print(f"  Family: {nlp_df['has_family_word'].sum()} ({nlp_df['has_family_word'].mean()*100:.1f}%)", flush=True)
    print(f"  Child: {nlp_df['has_child_word'].sum()} ({nlp_df['has_child_word'].mean()*100:.1f}%)", flush=True)
    print(f"  First person: {nlp_df['has_first_person'].sum()} ({nlp_df['has_first_person'].mean()*100:.1f}%)", flush=True)
    print(f"  Clickbait phrase: {nlp_df['has_clickbait_phrase'].sum()} ({nlp_df['has_clickbait_phrase'].mean()*100:.1f}%)", flush=True)
    
    # Save
    output_dir = '/home/ubuntu/KidInfluencer/analysis_v3/nlp'
    os.makedirs(output_dir, exist_ok=True)
    nlp_df.to_csv(f'{output_dir}/title_nlp_features.csv', index=False)
    print(f"\nSaved to {output_dir}/title_nlp_features.csv", flush=True)
    print("DONE.", flush=True)
