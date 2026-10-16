from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "student_feedback.csv"
OUT = BASE / "data" / "processed" / "clean_feedback.csv"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv(RAW)
    required = ["feedback_id","student_year","semester","department",
                "course","rating","feedback_text","submitted_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.drop_duplicates(subset=["feedback_text"]).copy()
    df["feedback_text_clean"] = df["feedback_text"].map(clean_text)
    df = df[df["feedback_text_clean"].str.len() >= 10].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "feedback_text_clean"])
    df["rating"] = df["rating"].astype(int)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} cleaned records -> {OUT}")

if __name__ == "__main__":
    main()
