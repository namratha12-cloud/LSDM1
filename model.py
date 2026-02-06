from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

def build_lstm_model(vocab_size, num_classes, embedding_dim=128, input_length=300):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_lstm_model(10000, 25)
    model.summary()
