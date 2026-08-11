import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Starting Email Spam Classifier...")

# Load dataset
data_path = "dataset/SMSSpamCollection"

data = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Split data
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data split completed.")

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text converted into numerical features.")

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model training completed.")

# Prediction
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
with open("model/spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY!")
print("==============================")
