# Email-Spam-Classifier
📧 Machine Learning-based Email Spam Classifier using TF-IDF and Multinomial Naive Bayes, with a Streamlit web interface for real-time spam detection.

## 👩‍💻 Intern Details

**Name:** Oguboina Namratha  
**Intern ID:** CITS7786  
**Project:** Email Spam Classifier  
**Project Type:** Machine Learning / NLP Internship Project

## 📌 Project Overview

Email Spam Classifier is a machine learning based application that
classifies text messages as either **Spam** or **Not Spam**.

The project uses Natural Language Processing techniques to convert
text into numerical features and a machine learning algorithm to
perform classification.

The application provides a simple web interface where users can
enter a message and instantly receive a prediction.

---

## 🎯 Objectives

- Detect spam messages automatically.
- Apply Natural Language Processing to text data.
- Train a machine learning classification model.
- Provide a user-friendly web interface.
- Display the probability of a message being spam.
- Evaluate the performance of the trained model.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Git & GitHub

---

## 📊 Dataset

The project uses the **SMS Spam Collection** dataset.

The dataset contains **5,572 messages** classified into two categories:

- `ham` - Not Spam
- `spam` - Spam

---

## ⚙️ Methodology

The project follows these steps:

1. Load the dataset.
2. Clean and prepare the data.
3. Convert spam and ham labels into numerical values.
4. Split the dataset into training and testing sets.
5. Apply TF-IDF vectorization.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate the trained model.
8. Save the model and vectorizer.
9. Build a Streamlit web application.
10. Use the trained model to classify new messages.

---

## 🤖 Machine Learning Algorithm

### Multinomial Naive Bayes

Multinomial Naive Bayes is a classification algorithm commonly used
for text classification problems.

It estimates the probability that a message belongs to a particular
class based on the words present in the message.

---

## 🔢 Feature Extraction

### TF-IDF

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It converts text into numerical values that represent the importance
of words within the dataset.

The resulting numerical features are provided to the machine
learning model for classification.

---

## 📈 Model Performance

The trained model achieved:

| Metric | Score |
|--------|-------|
| Accuracy | 97.04% |
| Spam Precision | 100% |
| Spam Recall | 78% |
| Spam F1-Score | 88% |

### Confusion Matrix

```text
                 Predicted
              Not Spam   Spam

Actual
Not Spam         966       0
Spam              33     116
