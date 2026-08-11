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

🌐 Application Features
🏠 Home

Provides information about the project, dataset, Machine Learning
algorithm, and classification workflow.

🔍 Spam Detection

Users can enter or paste a message and receive:

Spam / Not Spam prediction
Spam probability
Visual prediction result
📊 Model Performance

Displays:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Dataset information
🧪 Example Messages

The application also provides sample messages that can be used to
test the classifier.

💻 Installation and Setup

Follow the steps below to run the project locally.

1. Clone the Repository

Open Command Prompt or Terminal and run:

git clone YOUR_GITHUB_REPOSITORY_URL

Then move into the project directory:

cd Email-Spam-Classifier
2. Check Python Installation

Make sure Python is installed.

Check the Python version using:

python --version

Python 3.x is recommended.

3. Install Required Libraries

Install all required dependencies using:

pip install -r requirements.txt

The required libraries are:

pandas
scikit-learn
streamlit
4. Train the Machine Learning Model

Run:

python train_model.py

This will:

Load the dataset.
Split the data into training and testing sets.
Apply TF-IDF vectorization.
Train the Multinomial Naive Bayes model.
Evaluate the model.
Save the trained model.

The following files will be created inside the model folder:

model/
├── spam_model.pkl
└── tfidf_vectorizer.pkl
5. Evaluate the Model

To view the model evaluation separately, run:

python model_evaluation.py

This displays:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
6. Run the Streamlit Application

Start the web application using:

python -m streamlit run app.py

After running the command, Streamlit will provide a local URL such
as:

http://localhost:8501

Open the URL in your web browser.

📁 Project Structure
Email-Spam-Classifier/
│
├── dataset/
│   └── SMSSpamCollection
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── train_model.py
├── model_evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
🧪 Example
Spam Message
Congratulations! You have won ₹50,000.
Click here to claim your prize!

Expected result:

🚨 SPAM DETECTED!
Normal Message
Hi, please send me the project report before tomorrow.

Expected result:

✅ NOT SPAM
🔒 Important Notes

The model was trained using the SMS Spam Collection dataset.
Therefore, although the application is titled Email Spam Classifier,
the current model primarily demonstrates spam classification on
text messages.

Future versions can be trained using larger email-specific datasets
for improved email spam detection.

🔮 Future Enhancements

Possible improvements include:

📧 Integration with real email services.
🛡️ Phishing email detection.
🌍 Support for multiple languages.
🧠 Advanced NLP techniques.
🤖 Deep Learning based classification.
🔍 Email header analysis.
📊 Prediction history and analytics.
☁️ Deployment as a cloud-based application.
📱 Responsive user interface.
📌 Conclusion

The Email Spam Classifier demonstrates the application of Machine
Learning and Natural Language Processing for automated spam detection.

The system uses TF-IDF feature extraction and a Multinomial Naive
Bayes classifier and achieved approximately 97.04% accuracy on the
test dataset.

A Streamlit-based web interface allows users to interact with the
trained model and obtain real-time spam predictions.
