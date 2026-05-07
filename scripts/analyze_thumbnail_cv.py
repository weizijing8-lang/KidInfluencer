"""
Analyze the Ninja Battle Robot thumbnail using CV techniques:
- Face detection
- Emotion/expression detection
- Color saturation analysis
- Object detection signals
"""
import cv2
import numpy as np
from PIL import Image
import json

img_path = '/home/ubuntu/KidInfluencer/thumbnails_test/ninja_robot.jpg'

# Load image
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("=" * 70)
print("THUMBNAIL CV ANALYSIS: We Found A Giant Ninja Battle Robot!")
print("=" * 70)

# ============================================================
# 1. FACE DETECTION
# ============================================================
print("\n📸 1. FACE DETECTION")
print("─" * 50)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"  Faces detected: {len(faces)}")
for i, (x, y, w, h) in enumerate(faces):
    print(f"  Face {i+1}: position=({x},{y}), size={w}x{h}")

# ============================================================
# 2. MOUTH DETECTION (open mouth = surprise/shock)
# ============================================================
print("\n👄 2. MOUTH OPENNESS ANALYSIS")
print("─" * 50)

# For each detected face, check if mouth region has high contrast (open mouth)
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

for i, (x, y, w, h) in enumerate(faces):
    face_roi = gray[y:y+h, x:x+w]
    # Lower half of face = mouth region
    mouth_region = face_roi[int(h*0.6):, :]
    
    # High contrast in mouth region suggests open mouth
    mouth_std = np.std(mouth_region)
    mouth_range = int(np.max(mouth_region)) - int(np.min(mouth_region))
    
    # Detect smiles/open mouths
    smiles = mouth_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=5)
    
    print(f"  Face {i+1}: mouth_contrast={mouth_std:.1f}, range={mouth_range}, smiles_detected={len(smiles)}")

# ============================================================
# 3. COLOR SATURATION ANALYSIS
# ============================================================
print("\n🎨 3. COLOR SATURATION ANALYSIS")
print("─" * 50)

saturation = hsv[:, :, 1]
sat_mean = np.mean(saturation)
sat_std = np.std(saturation)
sat_high_pct = np.mean(saturation > 150) * 100  # % of highly saturated pixels

print(f"  Mean saturation: {sat_mean:.1f}/255")
print(f"  Saturation std:  {sat_std:.1f}")
print(f"  High saturation pixels (>150): {sat_high_pct:.1f}%")
print(f"  Interpretation: {'HIGH (clickbait-style)' if sat_mean > 100 else 'MODERATE' if sat_mean > 70 else 'LOW'}")

# ============================================================
# 4. BRIGHTNESS & CONTRAST
# ============================================================
print("\n💡 4. BRIGHTNESS & CONTRAST")
print("─" * 50)

brightness = np.mean(gray)
contrast = np.std(gray)
print(f"  Mean brightness: {brightness:.1f}/255")
print(f"  Contrast (std):  {contrast:.1f}")

# ============================================================
# 5. TEXT OVERLAY DETECTION (via edge density in typical text regions)
# ============================================================
print("\n📝 5. VISUAL COMPLEXITY (edge density)")
print("─" * 50)

edges = cv2.Canny(gray, 50, 150)
edge_density = np.mean(edges > 0) * 100
print(f"  Edge density: {edge_density:.1f}%")
print(f"  Interpretation: {'HIGH (complex/busy thumbnail)' if edge_density > 15 else 'MODERATE' if edge_density > 8 else 'SIMPLE'}")

# ============================================================
# 6. DOMINANT COLORS
# ============================================================
print("\n🔴 6. DOMINANT COLORS")
print("─" * 50)

# Check for red/yellow (attention-grabbing colors common in clickbait)
h_channel = hsv[:, :, 0]
s_channel = hsv[:, :, 1]
v_channel = hsv[:, :, 2]

# Red hue (0-10 or 170-180)
red_mask = ((h_channel < 10) | (h_channel > 170)) & (s_channel > 100) & (v_channel > 100)
red_pct = np.mean(red_mask) * 100

# Yellow hue (20-35)
yellow_mask = ((h_channel > 20) & (h_channel < 35)) & (s_channel > 100) & (v_channel > 100)
yellow_pct = np.mean(yellow_mask) * 100

# Purple/pink (130-160)
purple_mask = ((h_channel > 130) & (h_channel < 160)) & (s_channel > 50) & (v_channel > 50)
purple_pct = np.mean(purple_mask) * 100

print(f"  Red pixels:    {red_pct:.1f}%")
print(f"  Yellow pixels: {yellow_pct:.1f}%")
print(f"  Purple pixels: {purple_pct:.1f}%")
print(f"  Attention colors (red+yellow): {red_pct + yellow_pct:.1f}%")

# ============================================================
# 7. COMPOSITE SIGNALS
# ============================================================
print("\n" + "=" * 70)
print("📊 COMPOSITE SIGNAL SUMMARY")
print("=" * 70)

signals = {
    'faces_detected': len(faces),
    'has_multiple_faces': len(faces) >= 2,
    'high_saturation': sat_mean > 90,
    'high_edge_density': edge_density > 12,
    'attention_colors': (red_pct + yellow_pct) > 10,
    'purple_effects': purple_pct > 5,
}

print(f"\n  Signal                    Value       Interpretation")
print(f"  {'─' * 55}")
print(f"  Faces detected:           {signals['faces_detected']}           {'Multiple children' if signals['has_multiple_faces'] else 'Single/no face'}")
print(f"  High saturation:          {signals['high_saturation']}        {'Clickbait-style colors' if signals['high_saturation'] else 'Natural'}")
print(f"  High edge density:        {signals['high_edge_density']}        {'Busy/complex thumbnail' if signals['high_edge_density'] else 'Simple'}")
print(f"  Attention colors:         {signals['attention_colors']}        {'Red/Yellow dominant' if signals['attention_colors'] else 'Neutral palette'}")
print(f"  Purple/pink effects:      {signals['purple_effects']}        {'VFX/energy effects' if signals['purple_effects'] else 'No effects'}")

# Overall assessment
exploit_signals = sum([
    signals['has_multiple_faces'],
    signals['high_saturation'],
    signals['high_edge_density'],
    signals['attention_colors'],
    signals['purple_effects'],
])

print(f"\n  Exploitation visual signals: {exploit_signals}/5")
print(f"  Assessment: {'HIGHLY PRODUCED/CLICKBAIT' if exploit_signals >= 3 else 'MODERATELY PRODUCED' if exploit_signals >= 2 else 'ORGANIC-LOOKING'}")

# ============================================================
# 8. WHAT A HUMAN SEES (for comparison)
# ============================================================
print("\n" + "=" * 70)
print("👁️ WHAT A HUMAN OBSERVER SEES IN THIS THUMBNAIL:")
print("=" * 70)
print("""
  - 3 children wearing colored masks (ninja costumes) 
  - ALL 3 children have mouths wide open (exaggerated shock/surprise)
  - A large robot figure on the right side
  - Purple energy/lightning VFX effects
  - A UFO in the background
  - Highly saturated, colorful composition
  - Clearly STAGED/PERFORMATIVE (costumes + VFX + posed expressions)
  
  Human judgment: 
    ✅ Performative (costumes, staged poses, VFX)
    ✅ Emotional bait (exaggerated shock expressions)
""")
