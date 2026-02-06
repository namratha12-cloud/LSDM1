from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_resume_text

app = Flask(__name__)

# Configuration
MODEL_PATH = 'resume_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
LE_PATH = 'label_encoder.pkl'
MAX_LEN = 300

# Load assets at startup
print("Loading model and assets...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(LE_PATH, 'rb') as f:
    le = pickle.load(f)
print("Assets loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        resume_text = data.get('resume', '')
        
        if not resume_text:
            return jsonify({'error': 'No resume text provided'}), 400
            
        # Preprocess
        cleaned_text = clean_resume_text(resume_text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        category = le.inverse_transform([predicted_class_index])[0]
        
        return jsonify({
            'category': category,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
