"""Retail Sentiment & CRM Intelligence | Educational Streamlit prototype.

Place retail_sentiment_model.joblib (trusted locally trained model) alongside this file.
Run: python -m streamlit run app.py
"""
from pathlib import Path
import re

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "retail_sentiment_model.joblib"
ROBERTA_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

st.set_page_config(page_title="Retail Customer Intelligence", layout="wide")
st.title("Retail Customer Intelligence Dashboard")
st.caption("Educational prototype: sentiment, intent, aspects and CRM next-best action")

@st.cache_resource
def load_baseline():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading RoBERTa model (first run may download weights)...")
def load_roberta():
    from transformers import pipeline
    return pipeline("text-classification", model=ROBERTA_NAME, device=-1)

INTENT_RULES = {
    "Return / refund": [r"\brefund\b", r"\breturn(?:ing|ed)?\b", r"money back", r"\breplacement\b"],
    "Switching risk": [r"switch(?:ing)? to another brand", r"never buy again", r"stop buying", r"cancel my subscription"],
    "Delivery complaint": [r"late delivery", r"delayed delivery", r"order (?:is|was) late", r"not delivered", r"delivery (?:is|was) slow"],
    "Complaint": [r"\bdefective\b", r"\bdamaged\b", r"\bbroken\b", r"poor service", r"unhelpful", r"\bdisappointed\b"],
    "Discount inquiry": [r"\bdiscounts?\b", r"\bcoupon\b", r"\bcashback\b", r"\boffer\b", r"\bpromo(?:tion)?\b"],
    "Purchase interest": [r"want to buy", r"planning to (?:buy|purchase)", r"\blooking to buy\b", r"\bbuy another\b", r"ready to order"],
    "Product information": [r"\bavailable\b", r"\bavailability\b", r"\bspecifications?\b", r"\bwhat is the price\b", r"\bwhen will\b"],
}
ASPECT_RULES = {
    "Product quality": ["product", "quality", "defective", "damaged", "broken", "performance", "battery", "refrigerator", "television"],
    "Delivery": ["delivery", "shipping", "shipment", "arrived", "courier"],
    "Price / offers": ["price", "cost", "expensive", "cheap", "discount", "offer", "cashback"],
    "Customer service": ["service", "support", "staff", "agent", "helpful", "unhelpful"],
    "Availability": ["stock", "inventory", "available", "availability"],
}
POS_WORDS = {"excellent", "amazing", "great", "love", "good", "helpful", "fast", "fantastic", "satisfied", "quickly"}
NEG_WORDS = {"terrible", "slow", "late", "bad", "poor", "defective", "damaged", "broken", "unhappy", "disappointed", "unhelpful", "expensive"}

def detect_intents(text):
    found = [label for label, patterns in INTENT_RULES.items() if any(re.search(p, text, re.I) for p in patterns)]
    return found or ["General feedback"]

def aspect_findings(text):
    # Explicitly a rough clause-based lexical heuristic, NOT a trained ABSA model.
    clauses = re.split(r"\b(?:but|however)\b|[.!?;]", text, flags=re.I)
    results = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        words = set(re.findall(r"\b\w+\b", clause.lower()))
        pos = len(words & POS_WORDS)
        neg = len(words & NEG_WORDS)
        if re.search(r"\bnot\s+(?:good|great|helpful|fast)\b", clause, re.I):
            neg += 2
        label = "Negative" if neg > pos else "Positive" if pos > neg else "Undetermined"
        for aspect, keywords in ASPECT_RULES.items():
            if words.intersection(keywords):
                results.append((aspect, label, clause))
    return results

def value_segment(purchase_value):
    if purchase_value >= 60000:
        return "High value"
    if purchase_value >= 25000:
        return "Medium value"
    return "Lower value"

def next_action(intents, sentiment, segment):
    if "Return / refund" in intents:
        return "Open a return/refund service case; verify policy and resolve before marketing."
    if "Switching risk" in intents:
        return "Escalate to retention/customer care; understand the concern before any offer."
    if "Delivery complaint" in intents or "Complaint" in intents:
        return "Create a support case and prioritize service recovery."
    if sentiment == "negative":
        return "Contact customer care for diagnostic follow-up; avoid promotional messaging."
    if "Purchase interest" in intents:
        return "Provide relevant product recommendations after eligibility and consent checks."
    if "Discount inquiry" in intents:
        return "Check promotion eligibility and expected incremental margin before offering a discount."
    if sentiment == "positive" and segment == "High value":
        return "Consider loyalty or premium cross-sell outreach if the customer opted in."
    return "Provide requested information; use the standard consent-based CRM journey."

with st.sidebar:
    st.header("CRM inputs")
    customer_id = st.text_input("Customer ID", "101")
    purchase_value = st.number_input("Purchase value (₹; illustrative)", min_value=0, value=45000, step=1000)
    purchase_frequency = st.number_input("Purchase frequency", min_value=0, value=8, step=1)
    compare_roberta = st.checkbox("Compare with RoBERTa", value=False, help="First run downloads model weights; can be slow on CPU.")

feedback = st.text_area("Enter customer feedback", "The product is excellent but the delivery was very slow.", height=110)
if st.button("Analyze customer", type="primary"):
    if not feedback.strip():
        st.warning("Enter customer feedback first.")
        st.stop()

    st.subheader("1. Sentiment model comparison")
    baseline_sentiment = None
    if not MODEL_PATH.exists():
        st.error(f"Baseline model file not found: {MODEL_PATH}. Save the trained pipeline to this folder first.")
    else:
        try:
            model = load_baseline()
            baseline_sentiment = str(model.predict([feedback])[0]).lower()
            st.write("**TF-IDF + Logistic Regression:**", baseline_sentiment.upper())
            probs = model.predict_proba([feedback])[0]
            for label, score in zip(model.classes_, probs):
                st.write(f"{str(label).capitalize()}: {score:.1%}")
                st.progress(float(score))  # no PyArrow/Altair chart dependency
            st.caption("Baseline was trained on 15 illustrative records; class scores are not calibrated confidence.")
        except Exception as exc:
            st.error(f"Could not load or run baseline model: {exc}")

    roberta_sentiment = None
    if compare_roberta:
        try:
            result = load_roberta()(feedback, truncation=True, max_length=512)[0]
            roberta_sentiment = str(result["label"]).lower()
            st.write("**RoBERTa:**", roberta_sentiment.upper(), f"(model score: {result['score']:.1%})")
            st.caption("RoBERTa is a general social-media sentiment model; not validated for this retailer.")
        except Exception as exc:
            st.warning(f"RoBERTa unavailable: {exc}. Baseline and CRM analysis remain usable.")

    st.subheader("2. Customer intent detection")
    intents = detect_intents(feedback)
    st.write(", ".join(intents))
    st.caption("Keyword/phrase rule baseline; may miss paraphrases and negation.")

    st.subheader("3. Aspect sentiment (demonstration heuristic)")
    findings = aspect_findings(feedback)
    if findings:
        for aspect, sentiment, evidence in findings:
            st.write(f"**{aspect}** — {sentiment} | Evidence: {evidence}")
    else:
        st.write("No defined aspects detected.")
    st.caption("Clause-level keyword heuristic, not a validated aspect-based sentiment model.")

    st.subheader("4. CRM customer segment")
    segment = value_segment(purchase_value)
    st.write(f"Customer: {customer_id} | Segment: **{segment}** | Purchase frequency: {purchase_frequency}")
    st.caption("Value thresholds are illustrative; not an estimate of CLV or churn probability.")

    st.subheader("5. Recommended next-best action")
    chosen_sentiment = roberta_sentiment or baseline_sentiment or "unknown"
    st.info(next_action(intents, chosen_sentiment, segment))
    st.caption("Human review and consent/eligibility checks are required before automated CRM actions.")
