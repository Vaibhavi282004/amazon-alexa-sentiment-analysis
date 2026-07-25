import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("data/amazon_alexa.tsv", sep="\t")

# -------------------------------
# First 5 rows
# -------------------------------
print("First 5 Rows:")
print(df.head())

# -------------------------------
# Dataset Shape
# -------------------------------
print("\nDataset Shape:")
print(df.shape)

# -------------------------------
# Feedback Distribution
# -------------------------------
feedback_counts = df["feedback"].value_counts()

print("\nFeedback Distribution:")
print(feedback_counts)

plt.figure(figsize=(6,4))
feedback_counts.plot(kind="bar")

plt.title("Feedback Distribution")
plt.xlabel("Feedback")
plt.ylabel("Count")

plt.show()

# -------------------------------
# Rating Distribution
# -------------------------------
plt.figure(figsize=(6,4))

df["rating"].value_counts().sort_index().plot(kind="bar")

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.show()

# -------------------------------
# Review Length
# -------------------------------
df["review_length"] = df["verified_reviews"].fillna("").apply(len)

plt.figure(figsize=(8,5))

plt.hist(df["review_length"], bins=30)

plt.title("Distribution of Review Length")

plt.xlabel("Number of Characters")

plt.ylabel("Frequency")

plt.show()

# -------------------------------
# Word Cloud
# -------------------------------
text = " ".join(df["verified_reviews"].fillna(""))

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Most Frequent Words in Reviews")

plt.show()

# -------------------------------
# Top Product Variations
# -------------------------------
plt.figure(figsize=(10,5))

df["variation"].value_counts().plot(kind="bar")

plt.title("Product Variations")

plt.xlabel("Variation")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()