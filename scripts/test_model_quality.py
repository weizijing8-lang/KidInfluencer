#!/usr/bin/env python3
"""Test classification quality across models with tricky examples."""
import json
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are an expert in children's media and child exploitation research.
You will classify YouTube video titles from kidfluencer/family vlog channels along 5 dimensions.

For each title, output a JSON object with these fields:

- "performative": 1 if the child is clearly performing/working FOR the video (challenges, roleplay, scripted skits, dance routines, unboxing, reviews, pranks, games designed for content). 0 if organic/natural (birthday, vacation, daily life that would happen without a camera). -1 if ambiguous or clearly no child involved.

- "emotional_bait": 1 if the title uses exaggerated emotional language or manufactured drama to attract clicks. This includes: ALL CAPS shouting, excessive punctuation (!!!), fake emergencies, exaggerated reactions ("SHE CRIED", "I CAN'T BELIEVE", "EMOTIONAL"), surprise reveals ("SURPRISING..."), manufactured urgency, or sensationalized everyday events. Basically any title that amplifies emotions beyond what the content likely warrants. 0 if the title is calm, descriptive, or matter-of-fact.

- "narrative_conflict": 1 if the title implies a story with interpersonal conflict, mystery, or dramatic tension (theft: "WHO STOLE...", confrontation: "CONFRONTING...", betrayal, punishment, secrets revealed, someone getting caught, villain/hero dynamics, "gone wrong"). 0 if no narrative tension or conflict.

- "challenge_format": 1 if the title indicates a challenge, competition, or game format (e.g., "24 HOURS...", "LAST TO LEAVE...", "...VS...", "WHO CAN...", "$10,000 CHALLENGE", "TRY NOT TO LAUGH", dares, races, contests, any structured competitive activity designed for content). 0 if not a challenge/competition format.

- "commercial_content": 1 if the title references specific brands, products, stores, or commercial activities (e.g., "UNBOXING NEW iPHONE", "TESTING SLIME FROM AMAZON", "TARGET SHOPPING SPREE", "TRYING [brand] PRODUCTS", toy names, app names, store names, haul videos, sponsored content indicators). 0 if no brand/product/commercial reference.

IMPORTANT RULES:
- Only output a JSON array of objects, one per title, in the same order as input
- "emotional_bait": YES for exaggerated clickbait style (ALL CAPS + !!! + emotional words). A simple "Valentine's Day Surprise" = 0, but "SURPRISING MY GIRLS FOR VALENTINE'S DAY!!!" = 1 because of the exaggerated formatting.
- "narrative_conflict": Focus on INTERPERSONAL conflict or mystery, not just any activity.
- "challenge_format": Mark 1 for ANY structured game/challenge/competition format, even if it seems harmless (e.g., "WHO KNOWS ME BETTER" = 1).
- "commercial_content": Mark 1 if ANY brand name, product name, store name, or commercial activity is mentioned. Generic words like "toy" or "game" without a specific brand = 0.
- "performative" should be 1 for ANY content clearly produced for YouTube (challenges, games, skits, tutorials, reviews, pranks).
"""

# Tricky test cases with expected answers
TEST_TITLES = [
    "Cody turns 7",                                          # organic, no bait, no conflict, no challenge, no commercial
    "SNEAKING INTO MY OLD HIGH SCHOOL!!! **SURPRISING STUDENTS**",  # performative, emotional_bait=1, no challenge, no commercial
    "RV Trip, Capitol Reef - travel vlog",                   # organic, no bait, no conflict, no challenge, no commercial
    "LAST TO LEAVE THE POOL WINS $10,000!!!",               # performative, emotional_bait=1, challenge=1, no commercial
    "UNBOXING NEW iPHONE 15 PRO MAX!!!",                    # performative, emotional_bait=1, no challenge, commercial=1
    "WHO STOLE MY iPAD?!?!",                                # performative, emotional_bait=1, narrative_conflict=1, no challenge, no commercial
    "Aunt Linda's First Outing with ME! Griffiths Reunion 2022",  # organic, no bait, no conflict, no challenge, no commercial
    "24 HOUR OVERNIGHT CHALLENGE IN TARGET!!!",             # performative, emotional_bait=1, challenge=1, commercial=1
    "Juicy Marriage Q&A",                                    # no child, no bait, no conflict, no challenge, no commercial
    "OUR 2 YEAR OLD GOES ON HIS FIRST DATE!!! **GUESS WITH WHO**",  # emotional_bait=1, no challenge, no commercial
    "Saying Goodbye to my DOG",                             # organic, no bait (calm title), no conflict, no challenge, no commercial
    "BREAK UP PRANK ON GIRLFRIEND!!! (GONE WRONG)",         # performative, emotional_bait=1, narrative_conflict=1, no challenge, no commercial
    "Seashell Tour on Marco Island Florida - Beautiful Haul", # organic, no bait, no conflict, no challenge, no commercial
    "WHO KNOWS MOM BETTER?! CHALLENGE",                     # performative, emotional_bait=1, challenge=1, no commercial
    "Testing VIRAL TikTok SLIME from Amazon!!!",            # performative, emotional_bait=1, no challenge, commercial=1
]

EXPECTED = [
    {"performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 1, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 1},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 1, "challenge_format": 0, "commercial_content": 0},
    {"performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 1, "commercial_content": 1},
    {"performative": -1, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 1, "challenge_format": 0, "commercial_content": 0},
    {"performative": 0, "emotional_bait": 0, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 1, "commercial_content": 0},
    {"performative": 1, "emotional_bait": 1, "narrative_conflict": 0, "challenge_format": 0, "commercial_content": 1},
]


def test_model(model_name):
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(TEST_TITLES)])
    user_msg = f"Classify these {len(TEST_TITLES)} video titles. Return ONLY a JSON array of {len(TEST_TITLES)} objects.\n\n{titles_text}"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.1,
        max_tokens=8000,
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
    
    results = json.loads(content)
    
    # Score
    total_fields = 0
    correct = 0
    errors = []
    for i, (pred, exp) in enumerate(zip(results, EXPECTED)):
        for field in ['performative', 'emotional_bait', 'narrative_conflict', 'challenge_format', 'commercial_content']:
            total_fields += 1
            p = pred.get(field, 0)
            e = exp[field]
            if field == 'performative':
                # Allow some flexibility for performative
                if p == e:
                    correct += 1
                else:
                    errors.append(f"  [{i+1}] '{TEST_TITLES[i][:50]}' — {field}: got {p}, expected {e}")
            else:
                if p == e:
                    correct += 1
                else:
                    errors.append(f"  [{i+1}] '{TEST_TITLES[i][:50]}' — {field}: got {p}, expected {e}")
    
    accuracy = correct / total_fields * 100
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"Accuracy: {correct}/{total_fields} ({accuracy:.1f}%)")
    print(f"Errors ({len(errors)}):")
    for e in errors[:15]:
        print(e)
    print(f"{'='*60}")
    return accuracy


if __name__ == '__main__':
    models = ["gpt-4.1-nano", "gemini-2.5-flash", "gpt-4.1-mini"]
    for m in models:
        try:
            test_model(m)
        except Exception as e:
            print(f"\n{m}: ERROR - {e}")
