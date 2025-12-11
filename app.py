import streamlit as st
import joblib

# Load saved model + vectorizer + label encoder
model = joblib.load("logistic_model.joblib")        # Logistic Regression
vectorizer = joblib.load("vectorizer.joblib")       # TF-IDF Vectorizer
label_encoder = joblib.load("label_encoder.joblib") # Encodes 0/1 → Positive/Negative

st.set_page_config(page_title="Emotion Detection App", layout="centered")

st.title("🧠 Emotion Detection from Journal Text")
st.write("Enter any journal entry or sentence, and the model will classify the emotion.")

# Text Input
text = st.text_area("Write your text here:", height=200)

# Predict Button
if st.button("Predict Emotion"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Transform text → vector
        X = vectorizer.transform([text])

        # Predict label
        pred = model.predict(X)[0]

        # Convert label → actual text emotion
        emotion = label_encoder.inverse_transform([pred])[0]

        # Display result
        st.success(f"Predicted Emotion: **{emotion}**")
        st.balloons()
