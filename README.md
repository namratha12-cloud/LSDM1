# LSTM Resume Classifier
  An end-to-end deep learning project that uses LSTM neural networks to classify resumes into job categories.

# Project Overview
  This project implements a complete pipeline for resume classification:

- **Data Preprocessing:** Text cleaning, tokenization, and sequence padding
- **Model Architecture:** LSTM-based neural network with embedding layer
- **Web Application:** Flask-based UI with premium glassmorphic design
- **Trained Model:** Ready-to-use model trained on 2,484 resumes across 24 job categories

# Files Included
  - Core Python Scripts
  - app.py - Flask web application server
  -  model.py - LSTM model architecture definition
  - train.py - Training script
  -  preprocess.py - Data preprocessing pipeline
  -  predict.py - Standalone prediction script
  -  utils.py - Text cleaning utilities

# Web Application
  - templates/index.html - Frontend UI
  - static/style.css - Premium CSS styling

# Trained Assets
 -  resume_lstm_model.h5 - Trained LSTM model (~60 MB)
 - tokenizer.pkl - Fitted tokenizer for text processing
 - label_encoder.pkl - Label encoder for categories

# Installation
  - Install Dependencies
  - pip install flask tensorflow pandas scikit-learn
  - Run the Web Application
  - python app.py
  - Access the Application Open your browser and navigate to: http://127.0.0.1:5000

# Usage
  - Web Interface
  - Paste your resume text into the textarea
  - Click "Classify Resume"
  - View the predicted job category and confidence score
  - Command Line Prediction
  - from predict import predict_category

  - resume_text = "Your resume text here..."
  - category = predict_category(resume_text)
  - print(f"Predicted Category: {category}")

# Retraining the Model
  python train.py

# Model Performance
- **Dataset:** 2,484 resumes
- **Categories:** 24 job categories (HR, IT, Engineering, Finance, etc.)
- **Architecture:** Embedding → SpatialDropout1D → LSTM → Dense → Softmax
- **Training:** 10 epochs with validation split

# Job Categories
The model classifies resumes into the following categories:
|
|--ACCOUNTANT
|--ADVOCATE
|--AGRICULTURE
|--APPAREL
|--ARTS
|--AUTOMOBILE
|--AVIATION
|--BANKING
|--BPO
|--BUSINESS-DEVELOPMENT
|--CHEF
|--CONSTRUCTION
|--CONSULTANT
|--DESIGNER
|--DIGITAL-MEDIA
|--ENGINEERING
|--FINANCE
|--FITNESS
|--HEALTHCARE
|--HR
|--INFORMATION-TECHNOLOGY
|--PUBLIC-RELATIONS
|--SALES
|--TEACHER

# Technical Details
- **Framework:** TensorFlow/Keras
- **Web Framework:** Flask
- **Text Processing:** Custom regex-based cleaning
- **Sequence Length:** 300 tokens (max)
- **Vocabulary Size:** 10,000 words
- **Embedding Dimension:** 128

# License
This project is provided as-is for educational and commercial use.

# Author
Built using LSTM deep learning architecture for resume classification.
