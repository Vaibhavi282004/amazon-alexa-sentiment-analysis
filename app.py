import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/best_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Amazon Alexa Sentiment Analysis",
    page_icon="🤖"
)

# Title
st.title("🤖 Amazon Alexa Sentiment Analysis")

st.write("Enter an Amazon Alexa review below and click Predict.")

# User input
review = st.text_area("Review")

# Predict button
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vector = tfidf.transform([review])
        prediction = model.predict(review_vector)

        st.write("Prediction:", prediction)
        st.write("Prediction value:", prediction[0])

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("☹️ Negative Review")
            
            
  