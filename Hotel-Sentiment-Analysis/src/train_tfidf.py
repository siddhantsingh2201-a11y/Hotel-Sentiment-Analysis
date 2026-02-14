import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data
# We use the cleaned data we created in the previous step
data_path = os.path.join('data', 'processed_reviews.csv')
if not os.path.exists(data_path):
    # Fallback if running from src directory
    data_path = os.path.join('..', 'data', 'processed_reviews.csv')

df = pd.read_csv(data_path)

# Handle missing values in 'cleaned_review' just in case
df['cleaned_review'] = df['cleaned_review'].fillna('')

X = df['cleaned_review']
y = df['Sentiment']

# 2. Split Data (80% Train, 20% Test)
# Stratify ensures we keep the same proportion of pos/neg in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 3. Vectorization (TF-IDF)
# max_features=5000 keeps only the top 5k frequent words to reduce noise
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) 

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Model Training (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluation
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save Confusion Matrix Plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (TF-IDF + Logistic Regression)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_tfidf.png')
print("\nConfusion Matrix saved as 'confusion_matrix_tfidf.png'")

# 7. Save Model & Vectorizer
# We need both to make predictions on new data later!
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/tfidf_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print("\nModel and Vectorizer saved to 'models/' directory.")