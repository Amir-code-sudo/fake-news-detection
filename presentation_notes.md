# Presentation Guide: Fake News Detection using ML

This guide will help you explain the project during your PPIS final presentation.

## 1. Project Overview
- **Goal:** Classify news articles as **Real** or **Fake**.
- **Algorithms:** Multinomial **Naive Bayes**, **Logistic Regression**, **SVM** (Support Vector Machine).
- **Vectorization:** **TF-IDF** (Term Frequency-Inverse Document Frequency).
- **Features:**
  - Supports **3 classification algorithms** with live comparison.
  - Displays **evaluation metrics** (Accuracy, Precision, Recall, F1-Score).
  - Supports **file upload** (.txt files).

## 2. Key Academic Concepts

### TF-IDF (vs CountVectorizer)
- **TF** = How often a word appears in THIS document.
- **IDF** = How rare the word is across ALL documents.
- **TF-IDF** = TF × IDF → Words that are frequent in one doc but rare overall get higher weight.
- **Why better than CountVectorizer?** CountVectorizer just counts words; TF-IDF weights them by importance.

### Naive Bayes
- Based on **Bayes' Theorem**: P(Class|Text) = [P(Text|Class) × P(Class)] / P(Text)
- **"Naive"** = Assumes each word is independent of others.
- **Pros:** Fast, simple, works well with text data.
- **Cons:** Independence assumption is unrealistic.

### Logistic Regression
- Learns a **decision boundary** using the **sigmoid function**.
- **Output:** Probability between 0 and 1.
- **Pros:** Interpretable, handles high-dimensional data well.
- **Cons:** Assumes linear decision boundary.

### SVM (Support Vector Machine)
- Finds the **optimal hyperplane** that separates classes with maximum margin.
- **Support Vectors** = Data points closest to the decision boundary.
- **Pros:** Effective in high-dimensional spaces, robust to overfitting.
- **Cons:** Slower training, no native probability output.

## 3. The Workflow
1. **Input:** User types or uploads a news article (.txt).
2. **Preprocessing:** Lowercase → Remove special characters → Remove stopwords.
3. **TF-IDF Vectorization:** Convert clean text to numerical feature vector.
4. **Classification:** Selected model predicts Real or Fake.
5. **Result:** Display prediction + confidence percentage.

## 4. Model Comparison (Key Talking Point!)
- The app trains **all 3 models** on the same data with **80/20 train-test split**.
- Shows **4 metrics** for each model:
  - **Accuracy:** Overall correct predictions.
  - **Precision:** Of those predicted Fake, how many were actually Fake?
  - **Recall:** Of all actual Fake news, how many did we catch?
  - **F1-Score:** Harmonic mean of Precision and Recall (balanced metric).
- **Best model** is highlighted with a crown 👑.

## 5. Example Demo
- **Real:** "NASA launched a new satellite to monitor climate patterns" → Real (high confidence).
- **Fake:** "SHOCKING Scientists prove drinking bleach cures all diseases" → Fake (high confidence).
- **Tricky:** Try short or ambiguous text to discuss model limitations.

## 6. Potential Questions & Answers
- **Q: Why TF-IDF instead of CountVectorizer?**
  A: TF-IDF considers word importance, not just frequency.

- **Q: Why does SVM not show probability?**
  A: LinearSVC uses decision_function; we convert it to pseudo-probability using sigmoid.

- **Q: How to improve the model?**
  A: Use a larger dataset (Kaggle), add n-grams, try deep learning (LSTM/BERT).

---
*Good luck with your presentation!*
