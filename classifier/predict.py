import os
import joblib

# Paths to model files
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load models only once
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def classify_prompt(prompt: str) -> str:
    """
    Classify the prompt into a task category.

    Returns:
        coding | math | general
    """

    try:
        vector = vectorizer.transform([prompt])
        prediction = model.predict(vector)[0]

        return prediction

    except Exception as e:
        print(f"Classifier error: {e}")

        # safe fallback
        return "general"