import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/cleaned_amazon_alexa.csv")

# Fill missing values
df["clean_review"] = df["clean_review"].fillna("")

X = df["clean_review"]
y = df["feedback"]

print("\nFeedback Distribution (Original):")
print(y.value_counts())


# ==========================================
# Dataset Info
# ==========================================

print("=" * 60)
print("Dataset Information")
print("=" * 60)

print("\nShape:", df.shape)
print("\nUnique Labels:", y.unique())


# ==========================================
# TF-IDF (Improved)
# ==========================================

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2
)

X = tfidf.fit_transform(X)

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n✅ TF-IDF Vectorizer Saved")


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Handle Imbalance using SMOTE
# ==========================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE (Balanced Data):")
print(pd.Series(y_train).value_counts())


# ==========================================
# Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
}


# ==========================================
# Training Loop
# ==========================================

best_model = None
best_accuracy = 0
best_name = ""

print("\n" + "=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\n🔹 {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nPrediction Distribution:")
    print(pd.Series(predictions).value_counts())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name


# ==========================================
# Save Best Model
# ==========================================

joblib.dump(best_model, "models/best_model.pkl")

print("\n" + "=" * 60)
print(f"🏆 Best Model: {best_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("✅ Model Saved")
print("✅ Vectorizer Saved")
print("=" * 60)