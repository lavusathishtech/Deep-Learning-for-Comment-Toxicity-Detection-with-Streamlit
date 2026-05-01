import streamlit as st
import pandas as pd
import numpy as np
import re
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Toxicity AI Dashboard", layout="wide")

# Custom CSS for 4D Animated Background
st.markdown("""
<style>
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}
.stMarkdown, .stHeader, p, h1, h2, h3, label {
    color: white !important;
}
.stButton>button {
    background-color: #ffffff33;
    color: white;
    border-radius: 20px;
    border: 1px solid white;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')

sentiment_model = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

st.title("✨ Toxicity Prediction Dashboard")

tab1, tab2 = st.tabs(["Real-time Analysis", "Bulk Insights"])

with tab1:
    st.header("Predict Toxicity & Sentiment")
    user_input = st.text_area("Enter comment for deep analysis:", placeholder="Type something here...")
    
    if st.button("Run AI Analysis"):
        if user_input:
            cleaned = clean_text(user_input)
            result = sentiment_model(cleaned)[0]
            label = result['label'].upper()
            score = result['score']
            
            # Visual Score Metric
            st.subheader(f"Result: {label}")
            st.progress(score)
            st.write(f"Confidence Level: {round(score*100, 2)}%")
            
            if label == 'NEGATIVE':
                st.error("⚠️ Potential Toxicity Detected")
            elif label == 'POSITIVE':
                st.success("✅ Friendly/Positive Content")
            else:
                st.info("ℹ️ Neutral Content")
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.header("Data Exploration")
    st.write("Upload a CSV to process multiple comments at once.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df_bulk = pd.read_csv(uploaded_file)
        st.dataframe(df_bulk.head(10))
