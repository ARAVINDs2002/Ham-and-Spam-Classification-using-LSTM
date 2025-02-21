import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")
data = data[['v1', 'v2']].dropna()
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['message'])
sequences = tokenizer.texts_to_sequences(data['message'])
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['label'], test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')
])

# ✅ Optimized Learning Rate
optimizer = Adam(learning_rate=0.0003)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save_weights("spam_classifier_weights.weights.h5")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# ✅ Function to Smooth the Curve using Moving Average
def smooth_curve(data, smoothing_factor=0.8):
    smoothed = []
    last = data[0]  # Initialize with the first value
    for point in data:
        smoothed_value = last * smoothing_factor + point * (1 - smoothing_factor)
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed

# ✅ Plot Smoothed Curves
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(smooth_curve(history.history['loss']), label='Training Loss', color='blue')
plt.plot(smooth_curve(history.history['val_loss']), label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss (Smoothed)')
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(smooth_curve(history.history['accuracy']), label='Training Accuracy', color='blue')
plt.plot(smooth_curve(history.history['val_accuracy']), label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy (Smoothed)')
plt.legend()

plt.show()
