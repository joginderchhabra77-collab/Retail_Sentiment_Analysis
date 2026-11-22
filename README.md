# AI-Powered Customer Sentiment & CRM Intelligence

## Project Overview

An end-to-end Natural Language Processing (NLP) and Machine Learning application designed to analyze retail customer feedback, identify customer sentiment and purchase intentions, and support personalized CRM engagement.

The project demonstrates how unstructured customer feedback can be converted into actionable retail, marketing, and customer-service intelligence.

## Business Objectives

- Analyze positive, negative, and neutral customer feedback.
- Identify purchase, discount, refund, complaint, and switching intentions.
- Extract feedback about product quality, pricing, delivery, and customer service.
- Integrate sentiment with customer purchase value.
- Generate recommended CRM and customer-retention actions.

## Technologies Used

Python, Pandas, NumPy, NLTK, spaCy, Scikit-learn, TF-IDF, Logistic Regression, Hugging Face Transformers, RoBERTa, Joblib, and Streamlit.

## Machine Learning Architecture

Customer Feedback → Text Preprocessing → TF-IDF → Logistic Regression → Sentiment Prediction

An optional advanced model uses RoBERTa for contextual sentiment classification and model comparison.

Rule-based intent and aspect detection extend the prototype into a customer intelligence workflow.

## Business Applications

**Retail:** Identify recurring issues in product quality, delivery, and customer experience.

**Marketing:** Identify purchase intentions and opportunities for personalized targeting.

**CRM:** Prioritize dissatisfied customers and support retention and loyalty decisions.

**Campaign Optimization:** Demonstrate how NLP-derived insights can contribute to targeting and campaign experimentation.

## How to Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

## Dataset and Results

The initial learning prototype uses a small synthetic retail dataset for training the baseline sentiment classifier.

Advanced customer intent, aspect extraction, and CRM recommendations are demonstration features.

Illustrative campaign scenarios explore conversion uplift, incremental contribution, and return on investment. These are hypothetical business examples, not verified deployment results.

## Future Enhancements

- Larger synthetic and independently labeled customer datasets
- Fine-tuning transformer models for retail sentiment and intent
- Multilingual and Hindi-English customer feedback
- Customer churn prediction and customer lifetime value
- Uplift modeling and controlled campaign experiments
- Enterprise CRM integration and model monitoring

## Project Status

Educational working prototype, with production-scale validation and enterprise integration planned as future enhancements.
![Customer Sentiment Dashboard](screenshots/customer_sentiment_dashboard.png)