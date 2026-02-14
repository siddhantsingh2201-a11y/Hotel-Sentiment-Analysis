import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras Imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Data
print("Loading data...")
data_path = os.path.join('data', 'processed_reviews.csv')
if not os.path.exists(data_path):
    data_path = os.path.join('..', 'data', 'processed_reviews.csv')

df = pd.read_csv(data_path)
df['cleaned_review'] = df['cleaned_review'].fillna('')

# 2. Hyperparameters
MAX_NB_WORDS = 5000       # Top 5000 most frequent words
MAX_SEQUENCE_LENGTH = 250 # Pad/Cut reviews to 250 words
EMBEDDING_DIM = 100       # Vector size for each word

# 3. Tokenization (Text -> Integers)
print("Tokenizing text...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cleaned_review'].values)
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

X = tokenizer.texts_to_sequences(df['cleaned_review'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pd.get_dummies(df['Sentiment']).values # One-hot encoding for Keras

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Sentiment']
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")

# 5. Define LSTM Model Architecture
print("Building model...")
model = Sequential()

# Layer 1: Embedding
# Converts integer sequences into dense vectors of fixed size.
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

# Layer 2: SpatialDropout1D
# Helps prevent overfitting by dropping entire 1D feature maps.
model.add(SpatialDropout1D(0.2))

# Layer 3: LSTM
# The core logic. 100 units. dropout/recurrent_dropout handle regularization.
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

# Layer 4: Output
# 2 units (Positive, Negative) with Softmax activation
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 6. Train
# EarlyStopping prevents overfitting by stopping when validation loss stops improving
epochs = 5
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# 7. Evaluate
print("\nEvaluating model...")
accr = model.evaluate(X_test, y_test)
print(f'Test set\n  Loss: {accr[0]:0.3f}\n  Accuracy: {accr[1]:0.3f}')

# Generate Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nClassification Report (LSTM):")
print(classification_report(y_test_classes, y_pred_classes))

# Save Confusion Matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (LSTM)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_lstm.png')

# 8. Save Model & Tokenizer
os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.h5')
joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')
print("Model and Tokenizer saved.")