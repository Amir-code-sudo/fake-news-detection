# Fake News Detection System 🔍

AI-powered system for detecting fake news articles using 5 Machine Learning algorithms with bilingual support (English & Arabic).

## Features
- 🤖 **5 ML Models**: Naive Bayes, Logistic Regression, SVM, Random Forest, Decision Tree
- 🌍 **Bilingual**: English + Arabic + Egyptian dialect
- 📰 **Live News**: Fetch from BBC, CNN, Google News, Al Arabiya, BBC Arabic
- 📊 **Interactive Dashboard**: Charts, Confusion Matrices, Radar Chart, Pie Chart
- 🔬 **Explainable AI**: Shows WHY the model made its decision
- 🔄 **Compare All Models**: Analyze same article with all 5 models
- 📦 **Batch Analysis**: Analyze multiple articles at once
- 📤 **CSV Upload & Retrain**: Upload new data and retrain models live
- 🌙 **Dark Mode** & 🌐 **Arabic/English toggle**
- ✅ **30 Unit Tests** with pytest

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Run Tests
```bash
python -m pytest test_project.py -v
```

## Tech Stack
- **Backend**: Python, Flask, scikit-learn, pandas
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **NLP**: TF-IDF Vectorization with bilingual Unicode support
