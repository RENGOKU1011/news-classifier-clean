
import streamlit as st
import joblib

st.set_page_config(
    page_title="News Classification System",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("news_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("📰 News Classification System")

st.write(
    "Enter a news headline or article below "
    "to predict its category."
)

news_text = st.text_area(
    "Enter news text",
    height=180,
    placeholder="Type or paste your news headline here..."
)

if st.button("Predict Category", type="primary"):
    if news_text.strip():
        text_features = vectorizer.transform([news_text])
        prediction = model.predict(text_features)[0]

        st.success("Prediction completed!")
        st.subheader("Predicted Category")
        st.write(f"**{prediction}**")
    else:
        st.warning("Please enter some news text first.")
