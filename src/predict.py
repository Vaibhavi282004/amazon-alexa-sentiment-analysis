import joblib

# Load the trained model
model = joblib.load("models/best_model.pkl")

# Load the TF-IDF vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 50)
print(" AMAZON ALEXA SENTIMENT ANALYSIS ")
print("=" * 50)

while True:

    review = input("\nEnter an Amazon Alexa Review: ")

    # Exit option
    if review.lower() == "exit":
        print("\nThank you for using the Sentiment Analyzer!")
        break

    # Convert review into TF-IDF features
    review_vector = tfidf.transform([review])

    # Predict
    prediction = model.predict(review_vector)

    # Display result
    if prediction[0] == 1:
        print("\n✅ Sentiment: POSITIVE")
    else:
        print("\n❌ Sentiment: NEGATIVE")