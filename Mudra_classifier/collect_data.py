import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_PATH = './Bharatanatyam-Mudra-Dataset'
MODEL_PATH = 'hand_landmarker.task'
OUTPUT_FILE = 'mudra_data.csv'

# 1. Initialize Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

data = []

print(f"Starting relative extraction from: {DATA_PATH}")

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_path) or label.startswith('.'):
        continue

    print(f"Processing Mudra: {label}...")

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        full_img_path = os.path.join(label_path, img_name)
        
        try:
            image = mp.Image.create_from_file(full_img_path)
            detection_result = detector.detect(image)

            if detection_result.hand_landmarks:
                all_coords = []
                
                for hand_landmarks in detection_result.hand_landmarks:
                    # THE FIX: Use the wrist (landmark 0) as the origin (0,0,0)
                    wrist = hand_landmarks[0]
                    
                    for lm in hand_landmarks:
                        # Store distance from wrist instead of screen position
                        all_coords.extend([
                            lm.x - wrist.x, 
                            lm.y - wrist.y, 
                            lm.z - wrist.z
                        ])
                
                # Padding to 126 features
                if len(all_coords) < 126:
                    all_coords.extend([0.0] * (126 - len(all_coords)))
                
                final_row = all_coords[:126]
                final_row.append(label)
                data.append(final_row)

        except Exception as e:
            continue

# 4. Save to CSV
if data:
    columns = [f'coord_{i}' for i in range(126)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! New relative dataset saved as: {OUTPUT_FILE}")