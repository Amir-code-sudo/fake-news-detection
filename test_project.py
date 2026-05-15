"""
Test Suite for Fake News Detection System
Using pytest framework - TDD approach
Run: python -m pytest test_project.py -v
"""
import pytest
import json
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import FakeNewsDetector
from app import app


# ==========================================
#  MODEL TESTS
# ==========================================

class TestFakeNewsDetector:
    """Test the ML model core functionality"""

    @classmethod
    def setup_class(cls):
        """Train model once for all tests"""
        cls.detector = FakeNewsDetector()
        cls.detector.train()

    # --- Test 1: Model trains successfully ---
    def test_model_is_trained(self):
        assert self.detector.is_trained is True

    # --- Test 2: All 5 models are loaded ---
    def test_all_models_loaded(self):
        expected = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree']
        for model_name in expected:
            assert model_name in self.detector.models, f"Missing model: {model_name}"

    # --- Test 3: Vectorizer is initialized ---
    def test_vectorizer_exists(self):
        assert self.detector.vectorizer is not None

    # --- Test 4: Predict returns correct format ---
    def test_predict_returns_tuple(self):
        result = self.detector.predict("NASA launched a new satellite")
        assert len(result) == 3  # (prediction, confidence, reasons)

    # --- Test 5: Prediction is Real or Fake ---
    def test_predict_valid_label(self):
        pred, conf, reasons = self.detector.predict("Test news article")
        assert pred in ['Real', 'Fake']

    # --- Test 6: Confidence is between 0 and 100 ---
    def test_confidence_range(self):
        pred, conf, reasons = self.detector.predict("Test news article")
        assert 0 <= conf <= 100

    # --- Test 7: Fake news detected correctly ---
    def test_detect_fake_news(self):
        fake_text = "SHOCKING BREAKING NEWS bleach cures all diseases government hiding truth urgent exposed secret"
        pred, conf, reasons = self.detector.predict(fake_text)
        assert pred == 'Fake', f"Expected Fake, got {pred}"

    # --- Test 8: Real news detected correctly ---
    def test_detect_real_news(self):
        real_text = "The Federal Reserve announced a 0.25 percent interest rate increase following their quarterly meeting"
        pred, conf, reasons = self.detector.predict(real_text)
        assert pred == 'Real', f"Expected Real, got {pred}"

    # --- Test 9: Arabic fake news detection ---
    def test_detect_arabic_fake(self):
        fake_ar = "عاجل صدمة اكتشاف ان شرب البنزين يقوي المناعة فضيحة كبرى الحكومة تخبي الحقيقة"
        pred, conf, reasons = self.detector.predict(fake_ar)
        assert pred == 'Fake', f"Expected Fake for Arabic fake news, got {pred}"

    # --- Test 10: Arabic real news detection ---
    def test_detect_arabic_real(self):
        real_ar = "اعلن البنك المركزي المصري رفع سعر الفائدة بمقدار 2 بالمئة خلال اجتماع لجنة السياسة النقدية"
        pred, conf, reasons = self.detector.predict(real_ar)
        assert pred == 'Real', f"Expected Real for Arabic real news, got {pred}"

    # --- Test 11: All models give valid predictions ---
    def test_all_models_predict(self):
        text = "Scientists discovered a new species in the Amazon rainforest"
        for model_name in self.detector.models:
            pred, conf, reasons = self.detector.predict(text, model_name)
            assert pred in ['Real', 'Fake'], f"{model_name} gave invalid prediction"
            assert 0 <= conf <= 100, f"{model_name} confidence out of range"

    # --- Test 12: Compare all models ---
    def test_compare_all(self):
        results = self.detector.compare_all("Test article for comparison")
        assert len(results) == 5
        for r in results:
            assert 'model' in r
            assert 'prediction' in r
            assert 'confidence' in r

    # --- Test 13: Batch analysis ---
    def test_batch_analyze(self):
        texts = ["First news article", "Second news article", "Third one"]
        results = self.detector.batch_analyze(texts)
        assert len(results) == 3

    # --- Test 14: Reasons/Explanation returned ---
    def test_explanation_returned(self):
        pred, conf, reasons = self.detector.predict("SHOCKING bleach cures cancer", "Logistic Regression")
        assert isinstance(reasons, list)
        if len(reasons) > 0:
            assert 'word' in reasons[0]
            assert 'impact' in reasons[0]
            assert 'direction' in reasons[0]

    # --- Test 15: Top features extraction ---
    def test_top_features(self):
        features = self.detector.get_top_features(n=5)
        assert len(features) > 0

    # --- Test 16: Evaluation results ---
    def test_evaluation_results(self):
        evaluation = self.detector.get_evaluation()
        assert 'models' in evaluation
        assert 'best_model' in evaluation
        assert 'dataset' in evaluation
        assert evaluation['best_model'] is not None

    # --- Test 17: Dataset info ---
    def test_dataset_info(self):
        info = self.detector.dataset_info
        assert info['total'] > 0
        assert info['train_size'] > 0
        assert info['test_size'] > 0

    # --- Test 18: Confusion matrices ---
    def test_confusion_matrices(self):
        assert len(self.detector.confusion_matrices) == 5
        for name, cm in self.detector.confusion_matrices.items():
            assert 'matrix' in cm
            assert len(cm['matrix']) == 2
            assert len(cm['matrix'][0]) == 2

    # --- Test 19: Preprocessing works ---
    def test_preprocessing(self):
        text = "Hello!! World @#$ 123  مرحبا"
        clean = self.detector.preprocess_text(text)
        assert '@' not in clean
        assert '#' not in clean
        assert '!' not in clean

    # --- Test 20: Empty text handling ---
    def test_empty_text(self):
        pred, conf, reasons = self.detector.predict("")
        assert pred in ['Real', 'Fake']


# ==========================================
#  API TESTS
# ==========================================

class TestAPI:
    """Test Flask API endpoints"""

    @classmethod
    def setup_class(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    # --- Test 21: Home page loads ---
    def test_home_page(self):
        res = self.client.get('/')
        assert res.status_code == 200

    # --- Test 22: Analyze endpoint ---
    def test_analyze_endpoint(self):
        res = self.client.post('/analyze',
            data=json.dumps({'text': 'Test news', 'model': 'SVM'}),
            content_type='application/json')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'prediction' in data
        assert 'confidence' in data

    # --- Test 23: Analyze with no text ---
    def test_analyze_no_text(self):
        res = self.client.post('/analyze',
            data=json.dumps({'text': '', 'model': 'SVM'}),
            content_type='application/json')
        assert res.status_code == 400

    # --- Test 24: Models endpoint ---
    def test_models_endpoint(self):
        res = self.client.get('/models')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'models' in data
        assert 'best_model' in data

    # --- Test 25: Features endpoint ---
    def test_features_endpoint(self):
        res = self.client.get('/features')
        assert res.status_code == 200

    # --- Test 26: Compare all endpoint ---
    def test_compare_all_endpoint(self):
        res = self.client.post('/compare-all',
            data=json.dumps({'text': 'Test news article'}),
            content_type='application/json')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'results' in data
        assert len(data['results']) == 5

    # --- Test 27: Batch endpoint ---
    def test_batch_endpoint(self):
        res = self.client.post('/batch-analyze',
            data=json.dumps({'texts': ['News 1', 'News 2'], 'model': 'SVM'}),
            content_type='application/json')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'results' in data

    # --- Test 28: Stats endpoint ---
    def test_stats_endpoint(self):
        res = self.client.get('/stats')
        assert res.status_code == 200
        data = json.loads(res.data)
        assert 'total' in data
        assert 'real' in data
        assert 'fake' in data

    # --- Test 29: Fetch news endpoint ---
    def test_fetch_news_endpoint(self):
        res = self.client.post('/fetch-news',
            data=json.dumps({'source': 'BBC'}),
            content_type='application/json')
        assert res.status_code == 200

    # --- Test 30: Invalid source ---
    def test_fetch_invalid_source(self):
        res = self.client.post('/fetch-news',
            data=json.dumps({'source': 'INVALID_SOURCE'}),
            content_type='application/json')
        assert res.status_code == 400


# ==========================================
#  Run Summary
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("  FAKE NEWS DETECTION - TEST SUITE")
    print("  Running 30 tests...")
    print("=" * 60)
    pytest.main([__file__, '-v', '--tb=short'])
