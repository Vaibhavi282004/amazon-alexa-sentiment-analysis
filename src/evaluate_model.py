import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/cleaned_amazon_alexa.csv")

# Handle missing values
df["clean_review"] = df["clean_review"].fillna("")

X = df["clean_review"]
y = df["feedback"]

# -----------------------------
# Load TF-IDF Vectorizer
# -----------------------------

tfidf = joblib.load("models/tfidf_vectorizer.pkl")

X = tfidf.transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Load Best Model
# -----------------------------

model = joblib.load("models/best_model.pkl")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Accuracy:", accuracy)
print("=" * 50)

# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()