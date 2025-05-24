from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import base64
import io
from PIL import Image
from gtts import gTTS
import tempfile 
from deep_translator import GoogleTranslator
import os

app = Flask(__name__)
CORS(app)

model = load_model('gesture_model.h5')
labels = np.load('labels.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json.get('image')
        lang = request.json.get('lang', 'en')  # Default to English

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                data = np.array(data).reshape(1, -1)

                prediction = model.predict(data, verbose=0)
                index = np.argmax(prediction)
                label = labels[index]
                translated = GoogleTranslator(source='auto', target=lang).translate(label)
                print("Translated:", translated)
                # Convert label to voice
                tts = gTTS(text=translated, lang=lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    with open(fp.name, "rb") as audio_file:
                        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

                return jsonify({
                    'label': label,
                    'audio': f"data:audio/mpeg;base64,{audio_base64}"
                })

        return jsonify({'label': 'No hand detected'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/collect', methods=['POST'])
def collect():
    try:
        label = request.json.get('label')
        image_data = request.json.get('image')

        if not image_data or not label:
            return jsonify({'error': 'Missing image or label'}), 400

        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                os.makedirs('gesture_data', exist_ok=True)

                # Count how many files for this label
                existing_files = [f for f in os.listdir('gesture_data') if f.startswith(label)]
                file_path = os.path.join('gesture_data', f'{label}_{len(existing_files)}.npy')
                np.save(file_path, np.array(data))
                return jsonify({'message': f'Saved {file_path}'})

        return jsonify({'label': 'No hand detected'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Server started..........")
    app.run(port=5000)
    