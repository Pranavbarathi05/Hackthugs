import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('gesture_model.h5')
labels = np.load('labels.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data)
            index = np.argmax(prediction)
            label = labels[index]

            cv2.putText(image, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Real-time Gesture", image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
