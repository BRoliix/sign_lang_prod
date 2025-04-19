# src/models/translator_model.py
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Dropout

class SignLanguageTranslator:
    def __init__(self, data_dir="data/processed/landmarks", model_path=None):
        self.data_dir = data_dir
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model parameters
        self.max_text_length = 100
        self.vocab_size = 10000
        self.embedding_dim = 256
        self.lstm_units = 512
        
        # Initialize tokenizer
        self.tokenizer = None
        self.model = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_data(self):
        """Prepare data for training"""
        # Load dataset CSV
        dataset_csv = os.path.join(self.data_dir, "dataset.csv")
        if not os.path.exists(dataset_csv):
            raise FileNotFoundError(f"Dataset CSV not found at {dataset_csv}")
        
        df = pd.read_csv(dataset_csv)
        
        # Filter out rows with empty annotations
        df = df[df['annotation'].notna() & (df['annotation'] != "")]
        
        if len(df) == 0:
            raise ValueError("No valid data found with annotations")
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(df['annotation'])
        
        # Save tokenizer
        with open(os.path.join(self.model_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(df['annotation'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_text_length, padding='post')
        
        # Load landmarks
        X = []
        for landmark_path in df['landmark_path']:
            try:
                landmarks = np.load(landmark_path)
                X.append(landmarks)
            except Exception as e:
                print(f"Error loading {landmark_path}: {str(e)}")
        
        if len(X) == 0:
            raise ValueError("No valid landmark data found")
        
        # Pad landmarks to same length
        max_length = max(len(seq) for seq in X)
        landmark_dim = X[0].shape[1]
        X_padded = np.zeros((len(X), max_length, landmark_dim))
        for i, seq in enumerate(X):
            X_padded[i, :len(seq)] = seq
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, X_padded, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val, landmark_dim
    
    def build_model(self, input_shape, output_shape):
        """Build a sequence-to-sequence model"""
        # Encoder
        encoder_inputs = Input(shape=(self.max_text_length,))
        encoder_embedding = Embedding(
            self.vocab_size, self.embedding_dim, input_length=self.max_text_length
        )(encoder_inputs)
        encoder_lstm = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True)
        )(encoder_embedding)
        encoder_outputs, state_h, state_c = LSTM(
            self.lstm_units, return_state=True
        )(encoder_lstm)
        
        # Decoder
        decoder_inputs = Input(shape=(None, output_shape[1]))
        decoder_lstm = LSTM(
            self.lstm_units, return_sequences=True
        )(decoder_inputs, initial_state=[state_h, state_c])
        decoder_dense = Dense(output_shape[1])(decoder_lstm)
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], decoder_dense)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        # Prepare data
        X_train, X_val, y_train, y_val, landmark_dim = self.prepare_data()
        
        # Create decoder inputs (shifted by one time step)
        decoder_input_train = np.zeros((X_train.shape[0], y_train.shape[1], landmark_dim))
        decoder_input_train[:, 1:] = y_train[:, :-1]
        
        decoder_input_val = np.zeros((X_val.shape[0], y_val.shape[1], landmark_dim))
        decoder_input_val[:, 1:] = y_val[:, :-1]
        
        # Build model
        self.model = self.build_model(X_train.shape, y_train.shape)
        
        # Train model
        history = self.model.fit(
            [X_train, decoder_input_train], y_train,
            validation_data=([X_val, decoder_input_val], y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save model
        self.model.save(os.path.join(self.model_dir, "translator_model.h5"))
        
        return history
    
    def load_model(self, model_path):
        """Load the model and tokenizer"""
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
    
    def translate(self, text):
        """Translate text to sign language landmarks"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please train or load a model first.")
        
        # Tokenize text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_text_length, padding='post')
        
        # Get output shape from model
        output_shape = self.model.output_shape
        landmark_dim = output_shape[-1]
        
        # Create initial decoder input
        decoder_input = np.zeros((1, 1, landmark_dim))
        
        # Generate prediction
        max_length = 100  # Maximum sequence length to generate
        predicted_sequence = []
        
        for _ in range(max_length):
            # Predict next frame
            output = self.model.predict([padded_sequence, decoder_input], verbose=0)
            predicted_frame = output[0, -1, :]
            predicted_sequence.append(predicted_frame)
            
            # Update decoder input
            new_decoder_input = np.zeros((1, decoder_input.shape[1] + 1, landmark_dim))
            new_decoder_input[0, :-1] = decoder_input
            new_decoder_input[0, -1] = predicted_frame
            decoder_input = new_decoder_input
        
        return np.array(predicted_sequence)
