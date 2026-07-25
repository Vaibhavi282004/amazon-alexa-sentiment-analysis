import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_amazon_alexa.csv")

# Check for missing values
print("Missing values in clean_review:", df["clean_review"].isnull().sum())

# Replace missing values with empty strings
df["clean_review"] = df["clean_review"].fillna("")

# Features (X) and Target (y)
X = df["clean_review"]
y = df["feedback"]

# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Convert text into numerical features
X_tfidf = tfidf.fit_transform(X)

# Display matrix shape
print("\nTF-IDF Matrix Shape:")
print(X_tfidf.shape)

# Display first 20 feature names
print("\nFirst 20 Features:")
print(tfidf.get_feature_names_out()[:20])

# Save the vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n✅ TF-IDF Vectorizer saved successfully!")