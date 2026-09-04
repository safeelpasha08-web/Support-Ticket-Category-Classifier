import streamlit as st
import joblib
import re
import os
import warnings

# Hide scikit-learn version warning
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "model",
    "customer_support_ticket_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    "model",
    "tfidf_vectorizer.pkl"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error("Unable to load the model.")
    st.code(str(e))
    st.stop()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# URGENCY DETECTION
# --------------------------------------------------

HIGH_URGENCY_KEYWORDS = [
    "urgent",
    "urgently",
    "immediately",
    "asap",
    "emergency",
    "critical",
    "blocked",
    "account locked",
    "lost money",
    "payment failed",
    "fraud",
    "hacked",
    "not working",
    "cannot access"
]

MEDIUM_URGENCY_KEYWORDS = [
    "problem",
    "issue",
    "error",
    "failed",
    "delay",
    "please help",
    "trouble",
    "unable",
    "cannot"
]


def detect_urgency(text):

    text = text.lower()

    for keyword in HIGH_URGENCY_KEYWORDS:

        if keyword in text:
            return "High"

    for keyword in MEDIUM_URGENCY_KEYWORDS:

        if keyword in text:
            return "Medium"

    return "Low"


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("🎫 Customer Support Ticket Classifier")

st.write(
    "Enter a customer support ticket to predict its category "
    "and urgency level."
)

st.divider()

ticket_text = st.text_area(
    "Customer Support Ticket",
    placeholder=(
        "Example: I was charged twice for my order "
        "and need a refund."
    ),
    height=180
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Ticket", use_container_width=True):

    if not ticket_text.strip():

        st.warning("Please enter a customer support ticket.")

    else:

        cleaned_ticket = clean_text(ticket_text)

        # Convert text into TF-IDF features
        ticket_vector = vectorizer.transform(
            [cleaned_ticket]
        )

        # Predict category
        prediction = model.predict(ticket_vector)[0]

        # Detect urgency
        urgency = detect_urgency(ticket_text)

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.success("Prediction completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📂 Predicted Category")

            st.info(str(prediction))

        with col2:

            st.subheader("🚨 Urgency")

            if urgency == "High":

                st.error("🔴 High")

            elif urgency == "Medium":

                st.warning("🟠 Medium")

            else:

                st.success("🟢 Low")

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                ticket_vector
            )[0]

            confidence = max(probabilities) * 100

            st.subheader("📊 Prediction Confidence")

            st.progress(
                min(int(confidence), 100)
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------

        st.divider()

        st.subheader("📝 Ticket")

        st.write(ticket_text)

        st.subheader("📌 Result Summary")

        st.write(
            f"**Category:** {prediction}"
        )

        st.write(
            f"**Urgency:** {urgency}"
        )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses a machine learning model "
        "to classify customer support tickets."
    )

    st.write("**Model:** Logistic Regression")

    st.write("**Features:** TF-IDF")

    st.write("**Categories:** 14")

    st.write("**Model Accuracy:** 95.49%")

    st.write("**Weighted F1:** 95.61%")