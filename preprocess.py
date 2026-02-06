import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_resume_text
import pickle

def load_and_preprocess_data(filepath, max_words=10000, max_len=300):
    df = pd.read_csv(filepath)
    
    # Handle different column names
    text_col = 'Resume' if 'Resume' in df.columns else 'Resume_str'
    
    # Clean Resume Text
    df['cleaned_resume'] = df[text_col].apply(lambda x: clean_resume_text(x))
    
    # Label Encoding
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category'])
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_resume'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_resume'])
    
    # Padding
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = df['Category_Encoded'].values
    
    # Save objects for inference
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    return X, y, le.classes_, tokenizer

if __name__ == "__main__":
    X, y, classes, tokenizer = load_and_preprocess_data('UpdatedResumeDataSet.csv')
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
