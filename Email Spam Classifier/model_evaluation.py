import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------------
# Load Dataset
# -----------------------------------

data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------
# Split Dataset
# -----------------------------------

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# Load Vectorizer and Model
# -----------------------------------

with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------------
# Transform Test Data
# -----------------------------------

X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# Make Predictions
# -----------------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------------
# Calculate Metrics
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)


# -----------------------------------
# Display Results
# -----------------------------------

print("\n================================")
print("EMAIL SPAM CLASSIFIER")
print("MODEL PERFORMANCE")
print("================================")

print(f"\nDataset Size: {len(data)} messages")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(matrix)

print("\n================================")
print("Evaluation completed successfully!")
print("================================")
