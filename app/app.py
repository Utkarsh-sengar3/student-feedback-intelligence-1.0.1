import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import analyze_feedback

st.set_page_config(
    page_title="Student Feedback Intelligence",
    page_icon="🎓",
    layout="wide"
)

DATA = ROOT/"data"/"processed"/"clean_feedback.csv"

st.markdown("""
# 🎓 Student Feedback Intelligence
### AI-powered sentiment, topic and feedback analytics
""")

if not DATA.exists() or not (ROOT/"models"/"sentiment_model.joblib").exists():
    st.error("Models/data are not ready. Run: python run_pipeline.py")
    st.stop()

df = pd.read_csv(DATA)

# Sidebar filters
st.sidebar.header("Filters")
departments = st.sidebar.multiselect(
    "Department", sorted(df["department"].unique()),
    default=sorted(df["department"].unique())
)
courses = st.sidebar.multiselect(
    "Course", sorted(df["course"].unique()),
    default=sorted(df["course"].unique())
)

filtered = df[
    df["department"].isin(departments) &
    df["course"].isin(courses)
].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Feedback", len(filtered))
c2.metric("Positive", int((filtered.sentiment_label=="positive").sum()))
c3.metric("Neutral", int((filtered.sentiment_label=="neutral").sum()))
c4.metric("Negative", int((filtered.sentiment_label=="negative").sum()))

st.divider()

left, right = st.columns(2)
with left:
    counts = filtered["sentiment_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
    fig = px.pie(counts, names="sentiment", values="count", hole=.45,
                 title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

with right:
    by_course = filtered.groupby("course")["rating"].mean().sort_values().reset_index()
    fig = px.bar(by_course, x="rating", y="course", orientation="h",
                 title="Average Rating by Course")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feedback Intelligence")
topic_counts = filtered["topic_cluster"].value_counts().sort_index().reset_index()
topic_counts.columns = ["topic_cluster", "count"]
st.dataframe(topic_counts, use_container_width=True)

st.subheader("🔎 Analyze New Feedback")
text = st.text_area(
    "Paste student feedback",
    placeholder="Example: The instructor explains concepts well, but we need more practical examples."
)
if st.button("Analyze Feedback", type="primary"):
    if not text.strip():
        st.warning("Please enter feedback text.")
    else:
        result = analyze_feedback(text)
        a,b,c = st.columns(3)
        a.metric("Sentiment", result["sentiment"].title())
        b.metric("Confidence", f"{result['confidence']:.1%}")
        c.metric("Detected Topic", result["topic"])

st.subheader("Recent Feedback")
show_cols = ["feedback_id","department","course","rating","feedback_text","sentiment_label"]
st.dataframe(filtered[show_cols].head(50), use_container_width=True)
