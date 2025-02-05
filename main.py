import streamlit as st
from modules import reddit_data, summarizer, sentiment_analysis, visualizations, qa_bot
import os

st.sidebar.title("📌 Reddit Analyzer")
page = st.sidebar.radio("📂 Navigate", ["Home", "Summarization", "Sentiment Analysis", "Q/A Chatbot", "Visualizations"])

if page == "Home":
    st.title("🔍 Reddit Topic Analyzer - Data Collection")
    keyword = st.text_input("📌 Enter a keyword to search on Reddit", "")
    post_limit = st.slider("📥 Number of Posts", 10, 500, 100)
    max_comments = st.slider("💬 Max Comments per Post", 10, 100, 50)
    max_runtime = st.slider("⏳ Max Runtime (seconds)", 10, 300, 60)

    if st.button("🗑️ Clear Previous Data"):
        for file in ["data/reddit_data.csv", "data/reddit_sentiment_analysis.csv", "data/summarized_reddit_data.txt"]:
            if os.path.exists(file):
                os.remove(file)
        st.cache_data.clear()
        st.success("✅ Previous data cleared. Ready to fetch new data.")

    if st.button("📥 Fetch Data"):
        if not keyword.strip():
            st.error("⚠️ Please enter a valid keyword.")
        else:
            try:
                with st.spinner("🔄 Fetching data from Reddit..."):
                    data = reddit_data.fetch_reddit_data(
                        keyword=keyword,
                        post_limit=post_limit,
                        max_comments=max_comments,
                        max_runtime=max_runtime
                    )
                    reddit_data.save_data_to_csv(data)
                st.success("✅ New data fetched successfully!")

                st.info("🔍 Starting summarization...")
                summarizer.start_background_summarization()
                st.success("📄 Summarization started! Check the 'Summarization' page.")

            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")

elif page == "Summarization":
    st.title("📄 Summarized Reddit Data")
    try:
        is_processing, summary, error = summarizer.get_summarization_status()
        if is_processing:
            st.info("⏳ Summarization is in progress. You can explore other options...")
        elif error:
            st.error(f"❌ Error during summarization: {error}")
        elif summary:
            st.success("✅ Summarization completed!")
            st.write(summary)
        else:
            st.warning("⚠️ Summarization hasn't started. Please fetch data first.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

elif page == "Sentiment Analysis":
    st.title("📊 Sentiment Analysis")

    if st.button("🔍 Start Sentiment Analysis"):
        try:
            with st.spinner("⏳ Analyzing sentiments..."):
                sentiment_analysis.display_sentiment_analysis()
        except Exception as e:
            st.error(f"❌ Error during sentiment analysis: {str(e)}")

elif page == "Q/A Chatbot":
    st.title("🤖 Ask Questions About the Data")
    try:
        qa_bot.display_qa_bot()
    except Exception as e:
        st.error(f"❌ Error during Q/A: {str(e)}")

elif page == "Visualizations":
    try:
        visualizations.display_visualizations()
    except Exception as e:
        st.error(f"❌ Error displaying visualizations: {str(e)}")
