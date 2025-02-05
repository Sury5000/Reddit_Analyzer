import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import streamlit as st
from sentence_transformers import util

# Paths to locally stored models
BERT_MODEL_PATH = "./models/bert_model"
QA_MODEL_PATH = "./models/qa_model"

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_qa_pipeline(model_path):
    return pipeline("question-answering", model=model_path, tokenizer=model_path)

def compute_embeddings(sentences, tokenizer, model):
    
    batch_size = 16  # Process sentences in batches
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pooling
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)

def get_top_k_unique_answers(question, sentences, tokenizer, embedding_model, qa_pipeline, top_k=10):
    
    question_embedding = compute_embeddings([question], tokenizer, embedding_model).squeeze()
    sentence_embeddings = compute_embeddings(sentences, tokenizer, embedding_model)

    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings).squeeze()
    top_indices = torch.argsort(similarities, descending=True)[:top_k * 2]  # Retrieve more to filter duplicates

    unique_answers = []
    seen_answers = set()

    for index in top_indices:
        context = sentences[index]
        result = qa_pipeline({"context": context, "question": question})
        answer = result["answer"].strip()

        if answer and answer.lower() not in seen_answers:
            unique_answers.append({"answer": answer, "context": context})
            seen_answers.add(answer.lower())

        if len(unique_answers) == top_k:
            break

    return unique_answers

def display_qa_bot():
    
    st.title("Reddit Q&A Bot")

    # Load Data
    data_path = "data/reddit_data.csv"
    if not os.path.exists(data_path):
        st.error("Data file not found. Please ensure 'reddit_data.csv' exists in the 'data' directory.")
        return

    df = pd.read_csv(data_path)
    if df.empty or 'post_content' not in df or 'comment_body' not in df:
        st.error("Invalid or empty data file. Ensure 'post_content' and 'comment_body' columns exist.")
        return

    sentences = df['post_content'].dropna().tolist() + df['comment_body'].dropna().tolist()

    # Load models
    st.write("Loading models...")
    try:
        tokenizer, embedding_model = load_embedding_model(BERT_MODEL_PATH)
        qa_pipeline = load_qa_pipeline(QA_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # User Input
    question = st.text_input("Enter your question:")
    if question:
        try:
            st.write("Finding relevant answers...")
            top_answers = get_top_k_unique_answers(question, sentences, tokenizer, embedding_model, qa_pipeline, top_k=5)

            if not top_answers:
                st.warning("No relevant answers found.")
            else:
                for idx, ans in enumerate(top_answers, 1):
                    st.write(f"**Answer {idx}:** {ans['answer']}")
                    st.write(f"**Context:** {ans['context']}")
                    st.write("---")

        except Exception as e:
            st.error(f"Error during Q/A: {str(e)}")

if __name__ == "__main__":
    display_qa_bot()
