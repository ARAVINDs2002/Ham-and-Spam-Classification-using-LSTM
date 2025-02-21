import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1️⃣ Load dataset (for tokenization)
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")
data = data[['v1', 'v2']].dropna()
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2️⃣ Tokenizer (same as training)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['message'])  

# 3️⃣ Define Model Architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')
])

# ✅ FIX: Build Model Before Loading Weights
model.build(input_shape=(None, 50))  # Explicitly build the model

# 4️⃣ Compile Model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# 5️⃣ Load Pre-Trained Weights
model.load_weights("spam_classifier_weights.weights.h5")
print("✅ Model weights loaded successfully!")

# 6️⃣ Test the Model with Random Samples
test_messages = [
    "Congratulations! You have won a free lottery. Click here to claim your prize!",
    "Hey, are we still meeting for coffee tomorrow?",
    "You have been selected for a free iPhone. Reply NOW!",
    "Urgent: Your bank account is at risk. Verify your details immediately.",
    "Let's catch up this weekend for a movie."
]

# Convert messages to sequences
test_sequences = tokenizer.texts_to_sequences(test_messages)
test_padded = pad_sequences(test_sequences, maxlen=50, padding='post')

# Predict
predictions = model.predict(test_padded)

# Display Results
for i, msg in enumerate(test_messages):
    label = "Spam" if predictions[i] > 0.5 else "Ham"
    print(f"📩 Message: {msg}\n🔍 Prediction: {label} (Score: {predictions[i][0]:.4f})\n")
