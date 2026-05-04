# Samba Paper (CIKM 2022) - Key Notes

## Title: "Samba: Identifying Inappropriate Videos for Young Children on YouTube"

## Key Points from Abstract:
- Metadata-based approaches achieve high accuracy BUT have significant misclassifications due to reliability of input features
- They propose Samba: fusion model using BOTH metadata AND video subtitles
- Dataset: 70K videos
- Samba achieves 95% accuracy, outperforming metadata-only by at least 7%
- They release their dataset

## CCS Concepts:
- Security and privacy → Human and societal aspects of security and privacy
- Computing methodologies → Machine learning

## What this means for our work:
- Samba focuses on CONTENT SAFETY (inappropriate vs appropriate for children)
- NOT on detecting child labor or kidfluencer exploitation
- Their metadata features: thumbnails, title, comments
- Our problem is DIFFERENT: we're detecting exploitation OF children, not content inappropriate FOR children
- This is a key distinction - no one has done metadata-only detection of child LABOR indicators

## Gap we can fill:
- Samba = "is this video safe for kids to watch?"
- Our system = "is this video exploiting a child who is IN the video?"
- Completely different detection target
