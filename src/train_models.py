from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed" / "clean_feedback.csv"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    if "sentiment_label" not in df.columns:
        # Fallback rule-based labels from rating if external/unlabeled data is supplied.
        df["sentiment_label"] = df["rating"].map(
            lambda x: "positive" if x >= 4 else ("neutral" if x == 3 else "negative")
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["feedback_text_clean"], df["sentiment_label"],
        test_size=0.2, random_state=42, stratify=df["sentiment_label"]
    )

    sentiment_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, ngram_range=(1,2),
            min_df=2, sublinear_tf=True
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    sentiment_model.fit(X_train, y_train)
    pred = sentiment_model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    (REPORTS/"sentiment_report.txt").write_text(
        f"Accuracy: {acc:.4f}\n\n{report}"
    )
    joblib.dump(sentiment_model, MODELS/"sentiment_model.joblib")

    # Topic discovery / clustering
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df["feedback_text_clean"])
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["topic_cluster"] = kmeans.fit_predict(X)
    joblib.dump(vectorizer, MODELS/"topic_vectorizer.joblib")
    joblib.dump(kmeans, MODELS/"topic_kmeans.joblib")
    df.to_csv(DATA, index=False)

    print(f"Sentiment accuracy: {acc:.4f}")
    print("Models saved.")

if __name__ == "__main__":
    main()
