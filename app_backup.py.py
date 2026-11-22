import streamlit as st
import joblib

# Load trained sentiment model
model = joblib.load("retail_sentiment_model.joblib")

st.set_page_config(
	page_title="Retail Customer Intelligence",
	layout="wide"
)

st.title("Retail Customer Sentiment Analyzer")
st.write("Analyze customer feedback using NLP and Machine Learning.")

feedback = st.text_area("Enter Customer Feedback")

if st.button("Analyze Feedback"):
	if feedback.strip():
		sentiment = model.predict([feedback])[0]
		probabilities = model.predict_proba([feedback])[0]

		st.subheader("Predicted Sentiment")
		st.write(sentiment.upper())

		st.subheader("Sentiment Class Scores")
		scores = dict(zip(model.classes_, probabilities))
		st.bar_chart(scores)
	else:
		st.warning("Please enter customer feedback.")
