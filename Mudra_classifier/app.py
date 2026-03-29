import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load your trained Mudra model
with open('mudra_model.p', 'rb') as f:
    model_dict = pickle.load(f)
clf = model_dict['model']

# 2. Setup MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Open Camera with Mac-specific flag
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Mirror the frame for easier dancing/practice
    frame = cv2.flip(frame, 1)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        all_coords = []
        for hand_landmarks in result.hand_landmarks:
            # Get the wrist (landmark 0) as the reference point
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            wrist_z = hand_landmarks[0].z

            for lm in hand_landmarks:
                # Subtract wrist coordinates from every other landmark
                all_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

        if len(all_coords) < 126:
            all_coords.extend([0.0] * (126 - len(all_coords)))
        
        # Get probabilities instead of just the top result
        probabilities = clf.predict_proba([np.array(all_coords[:126])])
        confidence = np.max(probabilities)
        prediction = clf.predict([np.array(all_coords[:126])])

        # Only show the name if confidence is above 80%
        if confidence > 0.80:
            display_text = f"{prediction[0]} ({int(confidence*100)}%)"
            color = (0, 255, 0) # Green for high confidence
        else:
            display_text = "Analyzing..."
            color = (0, 0, 255) # Red for uncertain

    cv2.imshow('Bharatanatyam AI Monitor', frame)
    cv2.setWindowProperty('Bharatanatyam AI Monitor', cv2.WND_PROP_TOPMOST, 1) # Force to front
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()