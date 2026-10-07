
"""
app.py

Professional Streamlit web interface for the Fake Job Posting Detector.

Run:
    .\venv\Scripts\python.exe -m streamlit run app.py
"""
import os
import json

import streamlit as st
import joblib
import pandas as pd
import numpy as np
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_FILE = "models/fake_job_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
METRICS_FILE = "models/metrics.json"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
    }

    .small-text {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Load model + vectorizer
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():

    if not os.path.exists(MODEL_FILE):

        st.error(
            f"❌ Model file not found: `{MODEL_FILE}`\n\n"
            "Please run `train_model.py` first."
        )

        st.stop()

    if not os.path.exists(VECTORIZER_FILE):

        st.error(
            f"❌ Vectorizer file not found: `{VECTORIZER_FILE}`\n\n"
            "Please run `train_model.py` first."
        )

        st.stop()

    model = joblib.load(MODEL_FILE)

    vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


@st.cache_data
def load_metrics():

    if not os.path.exists(METRICS_FILE):
        return None

    with open(METRICS_FILE, "r") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(job_text, model, vectorizer):

    job_tfidf = vectorizer.transform([job_text])

    prediction = model.predict(job_tfidf)[0]

    probabilities = model.predict_proba(job_tfidf)[0]

    if prediction == 1:

        return "FAKE", probabilities[1] * 100

    return "REAL", probabilities[0] * 100


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():

    model, vectorizer = load_artifacts()
    metrics = load_metrics()


    # -----------------------------------------------------------------------
    # Session state
    # -----------------------------------------------------------------------

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "job_text" not in st.session_state:
        st.session_state["job_text"] = ""


    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    st.markdown(
        '<div class="main-title">🔍 Fake Job Posting Detector</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning powered job posting analysis'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    total_checks = len(st.session_state["history"])

    fake_checks = sum(
        item["Prediction"] == "FAKE"
        for item in st.session_state["history"]
    )

    real_checks = sum(
        item["Prediction"] == "REAL"
        for item in st.session_state["history"]
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Checks",
            total_checks
        )

    with col2:
        st.metric(
            "Real Predictions",
            real_checks
        )

    with col3:
        st.metric(
            "Fake Predictions",
            fake_checks
        )


    # -----------------------------------------------------------------------
    # Model Performance section
    # -----------------------------------------------------------------------

    if metrics:

        st.subheader("📊 Model Performance")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with metric2:
        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with metric3:
        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    with metric4:
        st.metric(
            "F1-score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

    # ----------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------

    if "confusion_matrix" in metrics:
        st.subheader("🔢 Confusion Matrix")

        cm = metrics["confusion_matrix"]

        st.write(
            "The confusion matrix shows how the model "
        "classified Real and Fake job postings."
    )

    # Create a table for the confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Real", "Actual Fake"],
        columns=["Predicted Real", "Predicted Fake"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # Show individual results
    col1, col2 = st.columns(2)

    with col1:

        st.write("### Actual Real Jobs")

        st.write(
            f"✅ Correctly predicted Real: **{cm[0][0]}**"
        )

        st.write(
            f"❌ Incorrectly predicted Fake: **{cm[0][1]}**"
        )

    with col2:

        st.write("### Actual Fake Jobs")

        st.write(
            f"❌ Incorrectly predicted Real: **{cm[1][0]}**"
        )

        st.write(
            f"✅ Correctly predicted Fake: **{cm[1][1]}**"
        )




    # -----------------------------------------------------------------------
    # Input section
    # -----------------------------------------------------------------------

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📝 Job Posting")

        job_text = st.text_area(
            "Paste the job posting below",
            value=st.session_state["job_text"],
            height=350,
            placeholder=(
                "Paste the complete job posting here...\n\n"
                "Example:\n"
                "Software Developer\n"
                "Company Name\n"
                "Job description...\n"
                "Requirements..."
            )
        )


        # ---------------------------------------------------------------
        # Prediction button
        # ---------------------------------------------------------------

        if st.button(
            "🔍 Analyze Job Posting",
            type="primary",
            use_container_width=True
        ):

            if not job_text.strip():

                st.warning(
                    "⚠️ Please enter a job posting first."
                )

            else:

                with st.spinner("Analyzing job posting..."):

                    label, confidence = predict(
                        job_text,
                        model,
                        vectorizer
                    )


                # -------------------------------------------------------
                # Save history
                # -------------------------------------------------------

                st.session_state["history"].append({
                    "Prediction": label,
                    "Confidence": f"{confidence:.2f}%",
                    "Job Text": job_text[:200]
                })
                st.divider()


                # -------------------------------------------------------
                # Result
                # -------------------------------------------------------

                st.subheader("📊 Detection Result")


                if label == "FAKE":

                    st.error(
                        "⚠️ This job posting is likely **FAKE**."
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                    st.warning(
                        "The model detected patterns associated with "
                        "fraudulent job advertisements. Verify the "
                        "employer and avoid sharing sensitive information."
                    )


                else:

                    st.success(
                        "✅ This job posting is likely **REAL**."
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                    st.info(
                        "The model classified this posting as real. "
                        "Still verify the employer before sharing "
                        "personal or financial information."
                    )


    # -----------------------------------------------------------------------
    # Samples / information
    # -----------------------------------------------------------------------

    with right:

        st.subheader("📋 Quick Test")

        real_sample = (
            "Software Developer\n"
            "ABC Technologies\n"
            "We are looking for a Python developer.\n"
            "Required skills include Python, SQL and Git.\n"
            "Candidates should have good programming skills."
        )

        fake_sample = (
            "Work From Home! Earn $5000/week!\n"
            "No experience needed. No interview. Instant hiring.\n"
            "Send us your bank account and SSN to get started.\n"
            "Limited time offer! Apply now!!!"
        )


        if st.button(
            "✅ Load Real Example",
            use_container_width=True
        ):

            st.session_state["job_text"] = real_sample
            st.rerun()


        if st.button(
            "⚠️ Load Fake Example",
            use_container_width=True
        ):

            st.session_state["job_text"] = fake_sample
            st.rerun()


        st.divider()


        st.subheader("ℹ️ How It Works")

        st.write(
            """
            **1. Enter a job posting**

            Paste the job description into the text box.

            **2. Text processing**

            The system converts the text into numerical
            TF-IDF features.

            **3. Machine Learning**

            Logistic Regression analyzes the features.

            **4. Prediction**

            The system predicts whether the posting is
            likely real or fake.
            """
        )


    # -----------------------------------------------------------------------
    # Prediction history
    # -----------------------------------------------------------------------

    if st.session_state["history"]:

        st.divider()

        st.subheader("📋 Prediction History")

        for i, item in enumerate(
            reversed(st.session_state["history"]),
            start=1
        ):

            with st.expander(
                f"{i}. {item['Prediction']} — "
                f"{item['Confidence']}"
            ):

                st.write("**Job Posting Preview:**")

            st.write(
                item["Job Text"]
            )

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state["history"] = []

            st.rerun()


    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------

    st.divider()

    st.caption(
        "⚠️ This system provides a machine-learning prediction and "
        "should not be treated as definitive proof that a job is "
        "fraudulent or legitimate."
    )

    st.caption(
        "Built with Python · Streamlit · scikit-learn · "
        "TF-IDF · Logistic Regression"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
