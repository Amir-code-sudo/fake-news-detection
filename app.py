from flask import Flask, render_template, request, jsonify
from model import FakeNewsDetector
import feedparser
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)
detector = FakeNewsDetector()
detector.train()

# stats counter
stats = {'total': 0, 'real': 0, 'fake': 0}

RSS_FEEDS = {
    'BBC': 'http://feeds.bbci.co.uk/news/rss.xml',
    'CNN': 'http://rss.cnn.com/rss/edition.rss',
    'Google News': 'https://news.google.com/rss?hl=en&gl=US&ceid=US:en',
    'Al Arabiya': 'https://www.alarabiya.net/feed/rss2/ar.xml',
    'BBC Arabic': 'https://feeds.bbci.co.uk/arabic/rss.xml',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model', 'Logistic Regression')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction, confidence, reasons = detector.predict(text, model_name)
    stats['total'] += 1
    stats['real' if prediction == 'Real' else 'fake'] += 1
    return jsonify({
        'prediction': prediction,
        'confidence': f"{confidence:.2f}",
        'model_used': model_name,
        'text_preview': text[:200],
        'reasons': reasons
    })

@app.route('/compare-all', methods=['POST'])
def compare_all():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text'}), 400
    results = detector.compare_all(text)
    return jsonify({'results': results, 'text_preview': text[:200]})

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    data = request.json
    texts = data.get('texts', [])
    model_name = data.get('model', 'Logistic Regression')
    if not texts:
        return jsonify({'error': 'No texts'}), 400
    results = detector.batch_analyze(texts[:20], model_name)
    for r in results:
        stats['total'] += 1
        stats['real' if r['prediction'] == 'Real' else 'fake'] += 1
    return jsonify({'results': results})

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'CSV files only'}), 400
    path = os.path.join('uploads', f.filename)
    os.makedirs('uploads', exist_ok=True)
    f.save(path)
    try:
        detector.train(csv_path=path)
        return jsonify({
            'message': 'Model retrained successfully',
            'dataset': detector.dataset_info,
            'balance': detector.balance_info,
            'evaluation': detector.get_evaluation()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/models')
def get_models():
    return jsonify(detector.get_evaluation())

@app.route('/features')
def get_features():
    return jsonify(detector.get_top_features(n=10))

@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    data = request.json
    source = data.get('source', 'BBC')
    url = RSS_FEEDS.get(source)
    if not url:
        return jsonify({'error': 'Unknown source'}), 400
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:12]:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            text = title
            if summary:
                clean_summary = BeautifulSoup(summary, 'html.parser').get_text()
                text = title + '. ' + clean_summary
            articles.append({
                'title': title,
                'text': text[:500],
                'link': entry.get('link', ''),
                'source': source
            })
        return jsonify({'articles': articles, 'source': source})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-url', methods=['POST'])
def fetch_url():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        feed = feedparser.parse(url)
        if feed.entries:
            articles = []
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = title
                if summary:
                    clean = BeautifulSoup(summary, 'html.parser').get_text()
                    text = title + '. ' + clean
                articles.append({
                    'title': title,
                    'text': text[:500],
                    'link': entry.get('link', ''),
                    'source': url
                })
            return jsonify({'articles': articles, 'source': url, 'type': 'rss'})

        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'article'], limit=15):
            text = tag.get_text(strip=True)
            if len(text) > 20:
                link_tag = tag.find('a')
                link = link_tag['href'] if link_tag and link_tag.get('href') else ''
                if link and not link.startswith('http'):
                    link = url.rstrip('/') + '/' + link.lstrip('/')
                articles.append({
                    'title': text[:200],
                    'text': text[:500],
                    'link': link,
                    'source': url
                })
        if not articles:
            paragraphs = soup.find_all('p')
            full_text = ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
            if full_text:
                articles.append({
                    'title': soup.title.string if soup.title else 'Page Content',
                    'text': full_text[:1000],
                    'link': url,
                    'source': url
                })
        return jsonify({'articles': articles, 'source': url, 'type': 'scrape'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
