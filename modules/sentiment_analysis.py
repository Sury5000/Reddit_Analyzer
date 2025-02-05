import os
import pandas as pd
import spacy
import streamlit as st
from transformers import pipeline
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

SENTIMENT_FILE = "data/reddit_sentiment_analysis.csv"
DATA_FILE = "data/reddit_data.csv"

@st.cache_resource
def load_models():
    return (
        spacy.load("en_core_web_trf"),
        pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    )

nlp, sentiment_pipeline = load_models()

def analyze_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]
    return result["label"], result["score"]

def extract_aspects(comment):
    doc = nlp(comment[:512])
    aspects = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1 and chunk.root.pos_ in ["NOUN", "PROPN"]:
            aspects.append(chunk.text.lower())
    return aspects

def generate_aspect_summary(aspect, comments):
    aspect_related_comments = [c for c in comments if aspect in c.lower()]
    sentiments = [analyze_sentiment(c) for c in aspect_related_comments]

    if not sentiments:
        return "No strong opinions."

    positive_comments = [c for c, s in zip(aspect_related_comments, sentiments) if s[0] == "POSITIVE"]
    negative_comments = [c for c, s in zip(aspect_related_comments, sentiments) if s[0] == "NEGATIVE"]

    summary = []
    if positive_comments:
        summary.append(f"**Liked for:**\n{positive_comments[0]}")
    if negative_comments:
        summary.append(f"\n**Criticism:**\n{negative_comments[0]}")

    return "\n\n".join(summary)  # 🔹 Clean structure with spacing

def perform_aspect_sentiment_analysis(comments):
    all_aspects = []
    aspect_summaries = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(extract_aspects, comments)
        all_aspects = [aspect for sublist in results for aspect in sublist]

    aspect_counts = Counter(all_aspects)
    top_aspects = [aspect for aspect, count in aspect_counts.most_common(5) if count > 2]

    for aspect in top_aspects:
        aspect_summaries[aspect.capitalize()] = generate_aspect_summary(aspect, comments)

    return aspect_summaries

def perform_sentiment_analysis():
    if not os.path.exists(DATA_FILE):
        st.error("⚠️ No data available. Please fetch Reddit data first.")
        return None

    df = pd.read_csv(DATA_FILE)
    if "comment_body" not in df.columns:
        st.error("⚠️ The dataset must have a 'comment_body' column.")
        return None

    if os.path.exists(SENTIMENT_FILE):
        return pd.read_csv(SENTIMENT_FILE)

    st.info("🔍 Performing sentiment analysis...")

    df["sentiment"], df["score"] = zip(*df["comment_body"].astype(str).apply(analyze_sentiment))
    df.to_csv(SENTIMENT_FILE, index=False)
    return df

def display_sentiment_analysis():

    df = perform_sentiment_analysis()

    if df is None:
        return

    st.subheader("🔹 Most Upvoted Comments")
    upvoted_pos = df[df["sentiment"] == "POSITIVE"].nlargest(1, "comment_score")["comment_body"].values
    upvoted_neg = df[df["sentiment"] == "NEGATIVE"].nlargest(1, "comment_score")["comment_body"].values
    st.write(f"**Positive:** {upvoted_pos[0] if upvoted_pos else 'No positive comments found'}")
    st.write(f"**Negative:** {upvoted_neg[0] if upvoted_neg else 'No negative comments found'}")

    st.subheader("🔹 Aspect-Based Sentiment Analysis")
    aspect_results = perform_aspect_sentiment_analysis(df["comment_body"].astype(str).tolist())

    for aspect, summary in aspect_results.items():
        st.markdown(f"**{aspect.capitalize()}**\n\n{summary}\n", unsafe_allow_html=True)  # 🔹 Structured Output
