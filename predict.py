import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_resume_text

def predict_category(resume_text, model_path='resume_lstm_model.h5', tokenizer_path='tokenizer.pkl', le_path='label_encoder.pkl'):
    # Load assets
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    
    # Preprocess
    cleaned_text = clean_resume_text(resume_text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)
    predicted_class_index = np.argmax(prediction)
    category = le.inverse_transform([predicted_class_index])[0]
    
    return category

if __name__ == "__main__":
    sample_resume = """
    Software Engineer with experience in Python, Django, and React. 
    Developed several web applications and worked on machine learning projects.
    Skills: Python, SQL, JavaScript, HTML, CSS.
    """
    category = predict_category(sample_resume)
    print(f"Predicted Category: {category}")
