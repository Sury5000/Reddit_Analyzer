# File: modules/visualizations.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
import re
import os

# --- FILE PATHS ---
SENTIMENT_FILE = "data/reddit_sentiment_analysis.csv"
REDDIT_DATA_FILE = "data/reddit_data.csv"

@st.cache_data
def clear_visualization_cache():
    """Clears cached visualization data to avoid displaying outdated results."""
    st.cache_data.clear()

def plot_sentiment_distribution():
    """Plots the distribution of sentiments in the data with percentages."""
    try:
        if not os.path.exists(SENTIMENT_FILE):
            st.warning("⚠️ Sentiment analysis file not found. Please run sentiment analysis first.")
            return

        clear_visualization_cache()
        sentiment_df = pd.read_csv(SENTIMENT_FILE)

        if 'sentiment' not in sentiment_df.columns:
            raise ValueError("The sentiment file must have a 'sentiment' column.")

        sentiment_counts = sentiment_df['sentiment'].value_counts()
        total_count = sentiment_counts.sum()
        sentiment_percentages = (sentiment_counts / total_count) * 100

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
        plt.title("Sentiment Distribution in Reddit Data", fontsize=14, pad=15)
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Count", fontsize=12)

        for i, (sentiment, count) in enumerate(zip(sentiment_counts.index, sentiment_counts.values)):
            ax.text(i, count + 5, f"{sentiment} ({count} - {sentiment_percentages[sentiment]:.2f}%)",
                    ha="center", fontsize=10, fontweight="bold")

        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error plotting sentiment distribution: {str(e)}")

def generate_word_cloud(selected_sentiment=None):
    """Generates a word cloud from Reddit comments."""
    try:
        if not os.path.exists(SENTIMENT_FILE):
            st.warning("⚠️ Sentiment analysis file not found. Please run sentiment analysis first.")
            return

        clear_visualization_cache()
        sentiment_df = pd.read_csv(SENTIMENT_FILE)

        if 'comment_body' not in sentiment_df.columns or 'sentiment' not in sentiment_df.columns:
            raise ValueError("The sentiment file must have 'comment_body' and 'sentiment' columns.")

        if selected_sentiment and selected_sentiment != "ALL":
            sentiment_df = sentiment_df[sentiment_df['sentiment'] == selected_sentiment]
            if sentiment_df.empty:
                raise ValueError(f"No comments found for sentiment '{selected_sentiment}'.")

        all_comments = " ".join(sentiment_df['comment_body'].dropna())

        def clean_text(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            return text.lower()

        cleaned_comments = clean_text(all_comments)

        if not cleaned_comments.strip():
            raise ValueError("No valid text available for word cloud.")

        wordcloud = WordCloud(
            background_color="white",
            width=800,
            height=400,
            colormap="viridis",
            collocations=False
        ).generate(cleaned_comments)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Sentiment: {selected_sentiment}" if selected_sentiment else "Word Cloud of Reddit Comments", fontsize=16)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error generating word cloud: {str(e)}")

def plot_engagement_metrics():
    """Plot engagement metrics: Top 10 subreddits by post count."""
    try:
        if not os.path.exists(REDDIT_DATA_FILE):
            st.warning("⚠️ Reddit data file not found. Please fetch new data first.")
            return

        clear_visualization_cache()
        df = pd.read_csv(REDDIT_DATA_FILE)

        if 'subreddit' not in df.columns:
            raise ValueError("The Reddit data file must have a 'subreddit' column.")

        top_subreddits = df['subreddit'].value_counts().head(10)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(y=top_subreddits.index, x=top_subreddits.values, palette="viridis")
        plt.title("Top 10 Subreddits by Post Count", fontsize=14, pad=15)
        plt.xlabel("Number of Posts", fontsize=12)
        plt.ylabel("Subreddit", fontsize=12)

        for i, value in enumerate(top_subreddits.values):
            ax.text(value + 2, i, str(value), va="center", fontsize=10, fontweight="bold")

        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error plotting engagement metrics: {str(e)}")

def plot_hourly_post_activity():
    """Plot the distribution of Reddit posts by hour."""
    try:
        if not os.path.exists(REDDIT_DATA_FILE):
            st.warning("⚠️ Reddit data file not found. Please fetch new data first.")
            return

        clear_visualization_cache()
        df = pd.read_csv(REDDIT_DATA_FILE)

        if 'post_created_utc' not in df.columns:
            raise ValueError("The Reddit data file must have a 'post_created_utc' column.")

        df['hour'] = pd.to_datetime(df['post_created_utc'], errors='coerce').dt.hour

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="hour", palette="husl")
        plt.title("Hourly Distribution of Reddit Posts", fontsize=14, pad=15)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error plotting hourly post activity: {str(e)}")

def plot_sentiment_trend():
    """Plot sentiment trend over time."""
    try:
        if not os.path.exists(SENTIMENT_FILE):
            st.warning("⚠️ Sentiment analysis file not found. Please run sentiment analysis first.")
            return

        clear_visualization_cache()
        sentiment_df = pd.read_csv(SENTIMENT_FILE)

        if 'post_created_utc' not in sentiment_df.columns or 'sentiment' not in sentiment_df.columns:
            raise ValueError("The sentiment file must have 'post_created_utc' and 'sentiment' columns.")

        sentiment_df['post_created_utc'] = pd.to_datetime(sentiment_df['post_created_utc'], errors='coerce')
        sentiment_df['date'] = sentiment_df['post_created_utc'].dt.date
        sentiment_trend = sentiment_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

        plt.figure(figsize=(12, 6))
        sentiment_trend.plot(kind='line', marker='o', figsize=(12, 6), colormap="coolwarm")
        plt.title("Sentiment Trend Over Time", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Number of Comments", fontsize=12)
        plt.legend(title="Sentiment")
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error plotting sentiment trend: {str(e)}")

def display_visualizations():
    """Displays all visualizations for Reddit data analysis."""
    st.title("📊 Reddit Data Visualizations")

    plot_sentiment_distribution()
    generate_word_cloud("POSITIVE")
    plot_engagement_metrics()
    plot_hourly_post_activity()
    plot_sentiment_trend()
