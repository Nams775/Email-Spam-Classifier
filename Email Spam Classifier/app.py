import streamlit as st
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------------
# Load Model and Vectorizer
# -----------------------------------

with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


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
# Calculate Model Performance
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

X_test_tfidf = vectorizer.transform(X_test)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)


# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Sidebar Navigation
# -----------------------------------

st.sidebar.title("📧 Spam Classifier")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Spam Detection",
        "📊 Model Performance"
    ]
)


# ===================================
# HOME PAGE
# ===================================

if page == "🏠 Home":

    st.markdown(
        '<div class="title">📧 Email Spam Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning based Spam Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📨 Total Messages",
            "5,572"
        )

    with col2:
        st.metric(
            "🎯 Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col3:
        st.metric(
            "🤖 Algorithm",
            "Naive Bayes"
        )

    st.divider()

    st.subheader("📌 About the Project")

    st.write(
        """
        The Email Spam Classifier is a machine learning application
        designed to identify whether a message is Spam or Not Spam.

        The system uses Natural Language Processing techniques to
        convert text into numerical features using TF-IDF and then
        classifies the message using a Multinomial Naive Bayes model.
        """
    )

    st.subheader("⚙️ How It Works")

    st.write(
        """
        1. 📩 User enters an email or message.
        2. 🧹 The text is processed.
        3. 🔢 TF-IDF converts the text into numerical features.
        4. 🤖 The Naive Bayes model analyzes the features.
        5. 🚨 The system predicts Spam or Not Spam.
        6. 📊 A confidence score is displayed.
        """
    )

    st.info(
        "Use the sidebar to try the Spam Detector or view Model Performance."
    )


# ===================================
# SPAM DETECTION PAGE
# ===================================

elif page == "🔍 Spam Detection":

    st.title("🔍 Spam Detection")

    st.write(
        "Enter an email or message below to classify it."
    )

    example = st.selectbox(
        "🧪 Try an example",
        [
            "Select an example...",
            "Congratulations! You have won ₹50,000. Click here to claim your prize!",
            "Hi, please send me the project report before tomorrow.",
            "URGENT! You have been selected for a FREE gift. Claim now!",
            "Hey, are you coming to college tomorrow?"
        ]
    )

    message = st.text_area(
        "📩 Enter your email/message",
        value="" if example == "Select an example..." else example,
        height=200,
        placeholder="Paste your email or message here..."
    )

    if st.button("🔍 Check Email", use_container_width=True):

        if message.strip() == "":
            st.warning("⚠️ Please enter an email or message.")

        else:

            message_tfidf = vectorizer.transform([message])

            prediction = model.predict(message_tfidf)[0]

            probabilities = model.predict_proba(message_tfidf)[0]

            spam_probability = probabilities[1] * 100

            if prediction == 1:

                st.error("🚨 SPAM DETECTED!")

                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2f}%"
                )

                st.progress(
                    min(int(spam_probability), 100)
                )

                st.warning(
                    "This message contains patterns commonly "
                    "associated with spam."
                )

            else:

                st.success("✅ NOT SPAM")

                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2f}%"
                )

                st.progress(
                    min(int(spam_probability), 100)
                )

                st.info(
                    "This message appears to be legitimate."
                )


# ===================================
# MODEL PERFORMANCE PAGE
# ===================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance metrics of the trained Email Spam Classifier."
    )

    st.divider()

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎯 Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.divider()

    # Confusion Matrix

    st.subheader("🔢 Confusion Matrix")

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.divider()

    # Dataset Information

    st.subheader("📚 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Messages",
            len(data)
        )

    with col2:
        st.metric(
            "Spam Messages",
            int(data["label"].sum())
        )

    st.write(
        """
        **Feature Extraction:** TF-IDF Vectorization

        **Classification Algorithm:** Multinomial Naive Bayes

        **Training/Test Split:** 80% Training and 20% Testing
        """
    )

    st.divider()

    st.subheader("📈 Interpretation")

    st.write(
        f"""
        The model achieved an accuracy of **{accuracy * 100:.2f}%**.

        The precision of **{precision * 100:.2f}%** indicates that
        messages predicted as spam were highly likely to actually
        be spam.

        The recall of **{recall * 100:.2f}%** indicates how effectively
        the model identified actual spam messages.

        The F1-score of **{f1 * 100:.2f}%** provides a balance between
        precision and recall.
        """
    )


# -----------------------------------
# Footer
# -----------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Email Spam Classifier\nMachine Learning Internship Project"
)
