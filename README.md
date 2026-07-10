# Real-Time SMS Spam Classifier

This repository contains an end-to-end Machine Learning project that classifies SMS messages into **Ham** (legitimate) or **Spam**. The system uses Natural Language Processing (NLP) for text preprocessing and a TF-IDF vectorization strategy paired with a Multinomial Naive Bayes classifier to ensure near-zero false positives.

---

## Project Architecture & Pipeline

- **Data Cleaning:** Handled structural real-world anomalies by dropping redundant unnamed columns, handling null fields, and removing duplicate entries to prevent training data leakage.
- **Exploratory Data Analysis (EDA):** Engineered text-profiling metrics (`num_characters`, `num_words`, `num_sentences`) via NLTK to statistically profile message lengths, visualizing distributions with Seaborn and word importance via WordClouds.
- **Text Preprocessing Pipeline:** Developed a custom modular preprocessing function executing:
  - Case normalization (lowercasing).
  - Tokenization via NLTK.
  - Alphanumeric filtering and punctuation removal.
  - Stopword filtering using the NLTK corpus.
  - Stemming via `PorterStemmer` to reduce tokens to their root forms.
- **Vectorization:** Implemented a `TfidfVectorizer` to capture relative word importances across the text corpus.
- **Model Evaluation:** Benchmarked three distinct variants of Naive Bayes classifiers (Gaussian, Bernoulli, and Multinomial).

---

## Model Evaluation & Results

In a spam detection system, **Precision** is prioritized over overall accuracy. A false negative (a spam message slipping into the inbox) is a minor inconvenience, whereas a false positive (a critical, legitimate text misclassified as spam) can cause significant data loss or missing crucial alerts.

The **Multinomial Naive Bayes (MNB)** model was chosen as the production model due to its exceptional precision score on the test set:

- **Accuracy:** 96.13%
- **Precision:** 99.07% *(Only 1 legitimate message misclassified out of the test sample)*

### Confusion Matrix (Test Set)

```text
[[888   1]  <- [True Ham,  False Spam]
 [ 39 106]] <- [False Ham, True Spam]
```

---

## Tech Stack

- **Programming Language:** Python
- **Data Processing & EDA:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, WordCloud
- **Natural Language Processing (NLP):** NLTK (Tokenization, Stopwords, PorterStemmer)
- **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LabelEncoder`, Evaluation Metrics)
- **Model Serialization:** Pickle
- **Web Application:** Streamlit

---

## Key Learnings

- **Precision-Driven Optimization:** Learned to prioritize precision over accuracy for spam detection, minimizing false positives to prevent legitimate messages from being incorrectly classified as spam.
- **Text Feature Engineering:** Gained hands-on experience converting unstructured text into numerical feature vectors using the TF-IDF weighting scheme.
- **Handling Imbalanced Data:** Analyzed datasets where legitimate messages significantly outnumber spam messages and evaluated models accordingly.
- **End-to-End ML Pipeline:** Built a complete machine learning workflow, including preprocessing, feature extraction, model training, evaluation, serialization with Pickle, and deployment using Streamlit.
- **Model Selection:** Compared Gaussian, Bernoulli, and Multinomial Naive Bayes classifiers and selected the production model based on precision and overall performance.
