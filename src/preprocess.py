import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download("stopwords")

# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv("data/amazon_alexa.tsv", sep="\t")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ----------------------------------------
# Check Missing Values
# ----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------------
# Feedback Distribution
# ----------------------------------------

print("\nFeedback Distribution:")
print(df["feedback"].value_counts())

# ----------------------------------------
# Load Stopwords
# ----------------------------------------

stop_words = set(stopwords.words("english"))

# ----------------------------------------
# Text Cleaning Function
# ----------------------------------------

def clean_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# ----------------------------------------
# Clean Reviews
# ----------------------------------------

df["clean_review"] = df["verified_reviews"].apply(clean_text)

# ----------------------------------------
# Show Sample Reviews
# ----------------------------------------

print("\nOriginal Review:")
print(df["verified_reviews"].iloc[0])

print("\nCleaned Review:")
print(df["clean_review"].iloc[0])

print("\nSample Data:")
print(df[["verified_reviews", "clean_review", "feedback"]].head(10))

# ----------------------------------------
# Save Cleaned Dataset
# ----------------------------------------
print("\nFeedback Distribution:")
print(df["feedback"].value_counts())
df.to_csv("data/cleaned_amazon_alexa.csv", index=False)

print("\n✅ Cleaned dataset saved successfully!")
print("Saved as: data/cleaned_amazon_alexa.csv")