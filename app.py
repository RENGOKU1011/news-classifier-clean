
import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="News Classification System",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "news_classifier.pkl"
    vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"

    if not model_path.is_file() or not vectorizer_path.is_file():
        st.error("Required model files are missing.")
        st.write(
            "Files found:",
            [p.name for p in BASE_DIR.iterdir()]
        )
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

st.title("📰 News Classification System")
st.write("Enter a news headline or article to predict its category.")

news_text = st.text_area(
    "Enter news text",
    placeholder="Example: The government announces new economic policies.",
    height=150
)

if st.button("Predict Category", type="primary"):
    if news_text.strip():
        features = vectorizer.transform([news_text.strip()])
        prediction = model.predict(features)[0]
        st.success(f"Predicted category: {prediction}")
    else:
        st.warning("Please enter some news text.")
