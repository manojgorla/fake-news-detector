import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# App title and description
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or paragraph below to check if it's real or fake.")

# Text input area
user_input = st.text_area("Enter News Text:")

# Predict button
if st.button("Check"):
    if user_input.strip():
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ This news seems REAL!")
        else:
            st.error("🚨 This news seems FAKE!")
    else:
        st.warning("Please enter some text first.")
