import streamlit as st
import joblib

st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("🧠 Emotion Detection from Journal Entry")
st.write("Enter your text below and the ML model will predict your emotion.")

# Text input
user_text = st.text_area("Write your journal text here:", height=200)

if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform text
        text_tfidf = tfidf.transform([user_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        emotion = "Positive" if prediction == 1 else "Negative"
        
        # Display Result
        st.subheader("Prediction Result")
        st.success(f"Emotion: **{emotion}**")


