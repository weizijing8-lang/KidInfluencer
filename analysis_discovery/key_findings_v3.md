# Key Findings v3: Child Labor Intensity × Emotional Manipulation

## Dataset
- 41,157 videos from 23 kidfluencer channels
- LLM-classified (gpt-4.1-nano): performative 52.2%, organic 36.6%, ambiguous 6.8%, no_child 4.4%
- Emotional exploitation flagged: 11.7% (4,805 videos)
- CV analysis: 2,100 thumbnails (OpenCV) + 270 LLM vision (gpt-4.1-mini)

## Core Finding 1: Performative content gets more views
- Performative median: 1,044,718 vs Organic median: 404,818
- Mann-Whitney U: p=0.00e+00 (highly significant)
- Within-channel control: performative +5.1% (p=0.0189*), organic -5.9% (p=0.0044**)
- Channel-level: Spearman ρ=0.427, p=0.042 (performative rate vs median views)

## Core Finding 2: Exploitation does NOT boost views
- Performative + no exploit: median 1,148,070
- Performative + exploit: median 1,190,744 (barely higher)
- Organic + no exploit: median 379,430
- Organic + exploit: median 478,946
- Channel-level: ρ=0.004, p=0.986 (exploit rate vs views - NO correlation)

## Core Finding 3: Content format matters hugely
- music_dance: +1264% view boost (43% performative, 1.8% exploit)
- roleplay: +243% (96% performative, 2.5% exploit)
- game: +193% (94% performative, 3.1% exploit)
- toy_play: +110% (86% performative, 0.3% exploit)
- drama: -32% (89% performative, 63.4% exploit!) ← high exploit, LOW views
- vlog: -54% (7% performative, 2.3% exploit)

## Core Finding 4: Thumbnail visual signals differ
- Performative: higher saturation (97.1 vs 85.2, p<0.0001), higher brightness (141 vs 135, p=0.0005)
- Performative: more faces detected (76.5% vs 71.5%, p=0.017)
- Organic: more large faces (30.8% vs 27.3%, ns), more smiles (17.2% vs 13.6%)
- LLM Vision: performative more "dramatic" (17.8% vs 10.9%) and "exciting" (25.9% vs 9.9%)
- LLM Vision: exploitation concern higher for performative (0.79 vs 0.56)

## Core Finding 5: Channel variation
- Highest performative: itsyeboi (96%), everleighrose (88%), brentrivera (87%)
- Lowest performative: theleray (22%), theweisslife (26%), cocomelon (37%)
- Highest exploit: piperrockelle (41.3%), acefamily (32.6%), jordanmatter (27.4%)
- Lowest exploit: vladandniki (0.4%), cocomelon (2.0%), ryansworld (2.3%)

## Key Narrative: "The Performativity Premium"
1. Algorithmic reward goes to performative content (+5.1% within-channel)
2. But emotional exploitation itself does NOT drive views (ρ=0.004)
3. The most rewarded formats (music, roleplay, game) are high-performative but LOW-exploitation
4. The highest exploitation format (drama, 63.4%) actually gets FEWER views (-32%)
5. This creates a "performativity premium" — children must work for content, but the exploitation is in the labor itself, not in emotional manipulation
