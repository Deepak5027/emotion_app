import streamlit as st
import joblib
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------------------------------------------------------
# Load ML Models
# -------------------------------------------------------------------
model = joblib.load("logistic_model.joblib")         # logistic regression model
vectorizer = joblib.load("vectorizer.joblib")        # tfidf vectorizer
label_encoder = joblib.load("label_encoder.joblib")  # label encoder

# -------------------------------------------------------------------
# Streamlit Page Settings
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Analyzer Dashboard",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔍 Emotion Detection", "📊 Text Analytics"]
)

# -------------------------------------------------------------------
# 🏠 HOME PAGE
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🧠 Emotion Analyzer Dashboard")
    st.write("Welcome! Use this dashboard to analyze emotions from text.")

    st.markdown("""
    ### 🚀 Features:
    - Predict emotion from text (Positive / Negative)
    - Generate word clouds
    - See keyword frequency
    - Sentiment score estimation
    """)

    st.image(
        "https://cdn.pixabay.com/photo/2017/01/31/14/44/artificial-intelligence-2024853_1280.png",
        use_column_width=True
    )

# -------------------------------------------------------------------
# 🔍 EMOTION DETECTION PAGE
# -------------------------------------------------------------------
elif page == "🔍 Emotion Detection":
    st.title("🔍 Emotion Detection")

    user_text = st.text_area("Enter text for emotion analysis:", height=200)

    if st.button("Predict Emotion"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Transform text & predict
            X = vectorizer.transform([user_text])
            pred = model.predict(X)[0]

            label = label_encoder.inverse_transform([pred])[0]

            st.success(f"### 🎭 Predicted Emotion: **{label}**")

            st.balloons()

# -------------------------------------------------------------------
# 📊 TEXT ANALYTICS PAGE
# -------------------------------------------------------------------
elif page == "📊 Text Analytics":
    st.title("📊 Text Analytics Tools")

    text = st.text_area("Paste long text to analyze patterns:", height=200)

    if st.button("Analyze Text"):
        if text.strip() == "":
            st.warning("Please enter text to analyze.")
        else:
            # ------------------ WORD CLOUD ------------------
            st.subheader("☁️ Word Cloud")

            words = " ".join(
                [w for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS]
            )

            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(words)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # ------------------ KEYWORD FREQUENCY ------------------
            st.subheader("📌 Keyword Frequency")

            words_list = [w.lower() for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS]
            freq = pd.Series(words_list).value_counts().head(10)

            st.bar_chart(freq)

            # ------------------ SENTIMENT SCORE ------------------
            st.subheader("📈 Sentiment Score (Model-based)")

            X = vectorizer.transform([text])
            proba = model.predict_proba(X)[0]

            score = proba[1]  # probability of positive

            st.metric("Sentiment Score", f"{score:.2f}")

