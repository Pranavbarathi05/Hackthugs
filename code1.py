import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

DATA_DIR = 'gesture_data'
LABEL = 'more'
  # ← change this for each gesture
SAMPLES = 100  # ← number of samples per gesture

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
collected = 0

while collected < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmark coordinates
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            npy_path = os.path.join(DATA_DIR, f'{LABEL}_{collected}.npy')
            np.save(npy_path, np.array(data))
            collected += 1
            print(f'Collected {collected}/{SAMPLES} for {LABEL}')

    cv2.imshow("Collecting", image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()