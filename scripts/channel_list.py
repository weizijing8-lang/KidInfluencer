"""
Channel list for the Kidfluencer Exploitation study.
Treatment group: Family vlog channels featuring children prominently.
Control group: Adult-only vloggers (no children featured).

Each entry: (short_name, channel_handle_or_id, category)
- category: "family" (treatment) or "adult" (control)
"""

CHANNELS = [
    # ============================================================
    # TREATMENT GROUP: Family Vlog Channels (children prominently featured)
    # ============================================================
    
    # --- Tier 1: High-profile / controversial family channels ---
    ("acefamily", "@TheACEFamily", "family"),
    ("labrantfam", "@TheLaBrantFam", "family"),
    ("ryansworld", "@RyansWorld", "family"),
    ("familyfunpack", "@FamilyFunPack", "family"),
    ("bratayley", "@Bratayley", "family"),
    ("8passengers", "@8passengers", "family"),  # Ruby Franke - convicted
    ("daddyofive", "@DaddyOFive", "family"),  # Michael Martin - convicted
    ("piperrockelle", "@PiperRockelle", "family"),
    
    # --- Tier 2: Large family vlog channels ---
    ("norrisnuts", "@TheNorrisNuts", "family"),
    ("itsyeboi", "@itsyeboi", "family"),  # FamilyOFive rebrand
    ("smellybellytv", "@SmellyBellyTV", "family"),
    ("thesuperherobuddy", "@TheEngineeringFamily", "family"),
    ("ehbee", "@EhBeeFamily", "family"),
    ("dailybumps", "@DailyBumps", "family"),
    ("thesacconejolys", "@SACCONEJOLYs", "family"),
    ("kkandbabyj", "@KKandbabyJ", "family"),
    ("thebramfam", "@TheBramFam", "family"),
    ("thefishfam", "@TheFishFam", "family"),
    ("theohana", "@TheOhana", "family"),
    ("tannerites", "@Tannerites", "family"),
    
    # --- Tier 3: Kid-centric channels (child is the star) ---
    ("likeNastya", "@LikeNastya", "family"),
    ("vladandniki", "@VladandNiki", "family"),
    ("dianakids", "@DianaKids", "family"),
    ("cocomelon", "@Cocomelon", "family"),
    ("kiddianashow", "@KidDianaShow", "family"),
    ("toysforkids", "@RyanToysReview", "family"),
    ("everleighrose", "@ForEverAndForAva", "family"),
    ("jojosiwaunofficial", "@itsjojosiwa", "family"),
    
    # --- Tier 4: Medium family channels ---
    ("theholdsworths", "@OfficiallyHoldsworths", "family"),
    ("thestaufferfamily", "@MYKAstauffer", "family"),  # Controversial adoption case
    ("familyfizz", "@FamilyFizz", "family"),
    ("thefranke", "@RubyFranke", "family"),  # Alt channel if exists
    ("bonniehoellein", "@BonnieHoellein", "family"),
    ("theweisslife", "@TheWeissLife", "family"),
    ("theleray", "@TheLeRoys", "family"),
    ("slyfoxhound", "@SlyFoxHound", "family"),
    
    # --- Tier 5: Additional family channels for statistical power ---
    ("thebeaulife", "@TheBeauLife", "family"),
    ("thedowneylife", "@TheDowneyLife", "family"),
    ("thejohnstons", "@Johnston7", "family"),
    ("themerrells", "@TheMerrellTwins", "family"),
    ("camilleandpavs", "@CamilleandPavs", "family"),
    ("thesquad", "@TheSquad", "family"),
    ("gavinmagnus", "@GavinMagnus", "family"),
    ("jordanmatter", "@JordanMatter", "family"),
    ("rebeccazamolo", "@RebeccaZamolo", "family"),
    ("shilohjolie", "@ShilohNelson", "family"),
    ("brentrivera", "@BrentRivera", "family"),
    ("lexirivera", "@LexiRivera", "family"),
    ("piersonwodzynski", "@Pierson", "family"),
    ("andrewdavila", "@AndrewDavila", "family"),
    
    # ============================================================
    # CONTROL GROUP: Adult-only vloggers (no children featured)
    # ============================================================
    
    # --- Travel / Food vloggers ---
    ("caseyneistat", "@casey", "adult"),
    ("markwiens", "@MarkWiens", "adult"),
    ("kara_and_nate", "@KaraandNate", "adult"),
    ("drewbinsky", "@DrewBinsky", "adult"),
    ("besteverfoods", "@BestEverFoodReviewShow", "adult"),
    ("mikechen", "@Strictlydum", "adult"),
    ("sonnysidevlog", "@BestEverFoodReviewShow", "adult"),
    ("indoorsmoker", "@SamTheLocalGuide", "adult"),
    
    # --- Lifestyle / Daily vloggers (adults only) ---
    ("emmachamberlain", "@emmachamberlain", "adult"),
    ("daviddobrik", "@DavidDobrik", "adult"),
    ("loganpaul", "@loganpaul", "adult"),
    ("jakepaul", "@jakepaul", "adult"),
    ("mrwhosetheboss", "@Mrwhosetheboss", "adult"),
    ("peterMcKinnon", "@PeterMcKinnon", "adult"),
    ("mattdavella", "@MattDAvella", "adult"),
    ("aliabdaal", "@aliabdaal", "adult"),
    
    # --- Comedy / Entertainment (adults only) ---
    ("mrbeast", "@MrBeast", "adult"),
    ("pewdiepie", "@PewDiePie", "adult"),
    ("markiplier", "@markiplier", "adult"),
    ("jacksepticeye", "@jacksepticeye", "adult"),
    ("dude_perfect", "@DudePerfect", "adult"),
    ("unspeakable", "@Unspeakable", "adult"),
    ("prestonplayz", "@PrestonPlayz", "adult"),
    
    # --- Fitness / Sports (adults only) ---
    ("jeffnippard", "@JeffNippard", "adult"),
    ("gregdoucette", "@GregDoucette", "adult"),
    ("atthleanx", "@ataborz", "adult"),
    ("natacha_oceane", "@NatachaOceane", "adult"),
    
    # --- Tech / Education (adults only) ---
    ("mkbhd", "@mkbhd", "adult"),
    ("linustechtips", "@LinusTechTips", "adult"),
    ("veritasium", "@veritasium", "adult"),
    ("smartereveryday", "@smartereveryday", "adult"),
    
    # --- Beauty / Fashion (adults only) ---
    ("jamescharles", "@jamescharles", "adult"),
    ("jeffreestar", "@jeffreestar", "adult"),
    ("nikkietutorials", "@NikkieTutorials", "adult"),
    
    # --- Music / Performance (adults only) ---
    ("roomieofficial", "@RoomieOfficial", "adult"),
    ("twosetviolin", "@TwoSetViolin", "adult"),
    
    # --- Additional adult vloggers ---
    ("grahamstephan", "@GrahamStephan", "adult"),
    ("andymation", "@Andymation", "adult"),
    ("colinfurze", "@colinfurze", "adult"),
    ("simoneGiertz", "@SimoneGiertz", "adult"),
    ("tomscott", "@TomScottGo", "adult"),
    ("johnnharris", "@johnnharris", "adult"),
    ("wendover", "@Wendoverproductions", "adult"),
    ("halfasinteresting", "@halfasinteresting", "adult"),
    ("kurzgesagt", "@kurzgesagt", "adult"),
    ("3blue1brown", "@3blue1brown", "adult"),
]

# Quick summary
family_channels = [c for c in CHANNELS if c[2] == "family"]
adult_channels = [c for c in CHANNELS if c[2] == "adult"]
print(f"Total channels: {len(CHANNELS)}")
print(f"  Family (treatment): {len(family_channels)}")
print(f"  Adult (control): {len(adult_channels)}")
