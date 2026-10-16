from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODEL = joblib.load(BASE/"models"/"sentiment_model.joblib")
VEC = joblib.load(BASE/"models"/"topic_vectorizer.joblib")
KMEANS = joblib.load(BASE/"models"/"topic_kmeans.joblib")

TOPIC_NAMES = {
    0: "Teaching & Explanation",
    1: "Assignments & Workload",
    2: "Practical Learning",
    3: "Course Content",
    4: "Interaction & Support",
    5: "Course Delivery"
}

def analyze_feedback(text: str):
    sentiment = MODEL.predict([text])[0]
    probabilities = MODEL.predict_proba([text])[0]
    confidence = float(max(probabilities))
    cluster = int(KMEANS.predict(VEC.transform([text]))[0])
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "topic_cluster": cluster,
        "topic": TOPIC_NAMES.get(cluster, f"Topic {cluster}")
    }
