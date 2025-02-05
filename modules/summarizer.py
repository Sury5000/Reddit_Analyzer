import os
import pandas as pd
import re
import threading
import torch
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DATA_FILE = "data/reddit_data.csv"
SUMMARY_FILE = "data/summarized_reddit_data.txt"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

summarization_status = {"is_processing": False, "result": None, "error": None}

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    return text.strip()

def cluster_comments(comments, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(comments)

    num_clusters = min(num_clusters, len(comments))  
    if num_clusters < 2:
        return [comments]  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    clustered_comments = {i: [] for i in range(num_clusters)}
    for comment, cluster in zip(comments, clusters):
        clustered_comments[cluster].append(comment)

    return list(clustered_comments.values())

def start_background_summarization():
    def summarize_data():
        global summarization_status
        summarization_status["is_processing"] = True
        summarization_status["error"] = None
        summarization_status["result"] = None
        
        try:
            summarized_text = summarize_content()
            summarization_status["result"] = summarized_text
        except Exception as e:
            summarization_status["error"] = str(e)
        finally:
            summarization_status["is_processing"] = False

    thread = threading.Thread(target=summarize_data)
    thread.start()

def get_summarization_status():
    return summarization_status["is_processing"], summarization_status["result"], summarization_status["error"]

def summarize_content(max_input_length=1024, max_summary_length=150, min_summary_length=50, num_clusters=5):
    if not os.path.exists(DATA_FILE):
        return "⚠️ No data available. Please fetch Reddit data first."

    data = pd.read_csv(DATA_FILE, encoding="utf-8")
    if "comment_body" not in data.columns:
        return "⚠️ The dataset must have a 'comment_body' column."

    data = data.nlargest(100, 'comment_score') if 'comment_score' in data.columns else data
    comments = data["comment_body"].dropna().astype(str).tolist()

    if not comments:
        return "⚠️ No valid text available for summarization."

    cleaned_comments = [clean_text(comment) for comment in comments if len(comment) > 20]
    clustered_comments = cluster_comments(cleaned_comments, num_clusters)

    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)

    summarized_chunks = []
    for cluster in clustered_comments:
        if not cluster:
            continue
        text_to_summarize = " ".join(cluster)[:max_input_length]
        summary = summarizer(text_to_summarize, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]["summary_text"]
        summarized_chunks.append(summary)

    summarized_text = " ".join(summarized_chunks)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        file.write(summarized_text.strip())

    return summarized_text.strip()
