import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("../data/training_prompts.csv")

prompts = data["prompt"]
labels = data["label"]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(prompts)

# Split dataset for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train classifier
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved:")
print("classifier/model.pkl")
print("classifier/vectorizer.pkl")