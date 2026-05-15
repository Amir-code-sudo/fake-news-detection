document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const analyzeBtn = document.getElementById('analyze-btn');
    const compareBtn = document.getElementById('compare-btn');
    const batchBtn = document.getElementById('batch-btn');
    const textInput = document.getElementById('text-input');
    const fileUpload = document.getElementById('file-upload');
    const csvUpload = document.getElementById('csv-upload');
    const csvStatus = document.getElementById('csv-status');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultBadge = document.getElementById('result-badge');
    const resultModelName = document.getElementById('result-model-name');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const resultPreview = document.getElementById('result-preview');
    const reasonsContainer = document.getElementById('reasons-container');
    const reasonsList = document.getElementById('reasons-list');
    const compareCard = document.getElementById('compare-card');
    const compareResults = document.getElementById('compare-results');
    const batchCard = document.getElementById('batch-card');
    const batchResults = document.getElementById('batch-results');
    const modelBtns = document.querySelectorAll('.model-btn');
    const fetchBtn = document.getElementById('fetch-btn');
    const fetchUrlBtn = document.getElementById('fetch-url-btn');
    const customUrlInput = document.getElementById('custom-url');
    const newsList = document.getElementById('news-list');
    const sourceBtns = document.querySelectorAll('.source-btn');
    const darkBtn = document.getElementById('dark-btn');
    const langBtn = document.getElementById('lang-btn');

    let selectedModel = 'Logistic Regression';
    let selectedSource = 'BBC';
    let accChart = null, radChart = null, pieChart = null;
    let isArabic = false;

    // Translations
    const tr = {
        page_title: ['Fake News Detection', 'كشف الأخبار المزيفة'],
        analyze_title: ['Analyze News Article', 'تحليل مقال إخباري'],
        upload_csv: ['Upload CSV (Retrain)', 'رفع CSV (إعادة تدريب)'],
        paste_placeholder: ['Paste a news article here (English or Arabic)...', 'الصق مقال إخباري هنا (عربي أو إنجليزي)...'],
        select_algo: ['Select Algorithm:', 'اختر الخوارزمية:'],
        analyze_btn: ['Analyze Article', 'تحليل المقال'],
        compare_btn: ['Compare All Models', 'مقارنة كل الموديلات'],
        batch_btn: ['Batch Analysis', 'تحليل جماعي'],
        classified_by: ['Classified by', 'مصنف بواسطة'],
        confidence: ['Confidence', 'درجة الثقة'],
        why_sub: ['(Words that influenced the decision)', '(الكلمات التي أثرت على القرار)'],
        compare_title: ['All Models Comparison', 'مقارنة كل الموديلات'],
        batch_title: ['Batch Analysis Results', 'نتائج التحليل الجماعي'],
        live_title: ['Live News Feed', 'الأخبار الحية'],
        fetch_btn: ['Fetch News', 'جلب الأخبار'],
        url_placeholder: ['Or paste any news website URL here...', 'أو الصق رابط أي موقع إخباري هنا...'],
        pie_title: ['Dataset Distribution', 'توزيع البيانات'],
        acc_title: ['Model Accuracy Comparison', 'مقارنة دقة الموديلات'],
        radar_title: ['Metrics Radar Chart', 'رسم بياني راداري للمقاييس'],
        cm_title: ['Confusion Matrices', 'مصفوفات الارتباك'],
        kw_title: ['Top Keywords', 'أهم الكلمات الدالة'],
        table_title: ['Performance Table', 'جدول الأداء'],
        th_algo: ['Algorithm', 'الخوارزمية'],
        th_acc: ['Accuracy', 'الدقة'],
        th_prec: ['Precision', 'الإحكام'],
        th_recall: ['Recall', 'الاستدعاء'],
        analyzed: ['analyzed', 'تم تحليلها'],
    };

    function applyLang() {
        const idx = isArabic ? 1 : 0;
        document.documentElement.dir = isArabic ? 'rtl' : 'ltr';
        document.body.dir = isArabic ? 'rtl' : 'ltr';
        document.getElementById('page-title').textContent = tr.page_title[idx];
        document.querySelectorAll('[data-t]').forEach(el => {
            const key = el.dataset.t;
            if (tr[key]) el.textContent = tr[key][idx];
        });
        document.querySelectorAll('[data-t-placeholder]').forEach(el => {
            const key = el.dataset.tPlaceholder;
            if (tr[key]) el.placeholder = tr[key][idx];
        });
    }

    // Dark mode
    darkBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        darkBtn.textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
    });

    // Language toggle
    langBtn.addEventListener('click', () => {
        isArabic = !isArabic;
        applyLang();
    });

    // Model selector
    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModel = btn.dataset.model;
        });
    });

    // Source selector
    sourceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sourceBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedSource = btn.dataset.source;
        });
    });

    // File upload (txt)
    fileUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => { textInput.value = ev.target.result; };
        reader.readAsText(file);
    });

    // CSV upload + retrain
    csvUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        csvStatus.className = 'csv-status loading';
        csvStatus.textContent = 'Uploading and retraining models...';
        csvStatus.classList.remove('hidden');
        const fd = new FormData();
        fd.append('file', file);
        try {
            const res = await fetch('/upload-csv', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) {
                csvStatus.className = 'csv-status error';
                csvStatus.textContent = 'Error: ' + data.error;
            } else {
                let msg = '✅ Model retrained! Dataset: ' + data.dataset.total + ' articles (' + data.dataset.real_count + ' Real, ' + data.dataset.fake_count + ' Fake)';
                if (data.balance && !data.balance.balanced) {
                    csvStatus.className = 'csv-status error';
                    msg += ' ⚠️ Warning: Dataset is unbalanced (' + data.balance.ratio + '%)! Add more ' + (data.balance.real > data.balance.fake ? 'Fake' : 'Real') + ' news for better accuracy.';
                } else {
                    csvStatus.className = 'csv-status success';
                    if (data.balance) msg += ' | Balance: ' + data.balance.ratio + '% ✅';
                }
                csvStatus.textContent = msg;
                loadAll();
            }
        } catch(e) {
            csvStatus.className = 'csv-status error';
            csvStatus.textContent = 'Upload failed';
        }
    });

    // Analyze single
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) { alert('Please enter text.'); return; }
        analyzeBtn.disabled = true;
        resultCard.classList.add('hidden');
        compareCard.classList.add('hidden');
        batchCard.classList.add('hidden');
        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text, model: selectedModel})
            });
            const data = await res.json();
            if (data.error) alert(data.error);
            else showResult(data);
            updateStats();
        } catch(e) { alert('Server error'); }
        analyzeBtn.disabled = false;
    });

    // Compare all models
    compareBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) { alert('Please enter text.'); return; }
        compareBtn.disabled = true;
        resultCard.classList.add('hidden');
        batchCard.classList.add('hidden');
        try {
            const res = await fetch('/compare-all', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text})
            });
            const data = await res.json();
            showCompare(data);
        } catch(e) { alert('Server error'); }
        compareBtn.disabled = false;
    });

    // Batch analysis
    batchBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) { alert('Enter multiple articles separated by new lines.'); return; }
        const texts = text.split('\n').map(t => t.trim()).filter(t => t.length > 10);
        if (!texts.length) { alert('Enter at least one article per line.'); return; }
        batchBtn.disabled = true;
        resultCard.classList.add('hidden');
        compareCard.classList.add('hidden');
        try {
            const res = await fetch('/batch-analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({texts, model: selectedModel})
            });
            const data = await res.json();
            showBatch(data);
            updateStats();
        } catch(e) { alert('Server error'); }
        batchBtn.disabled = false;
    });

    function showResult(data) {
        const fake = data.prediction === 'Fake';
        const conf = parseFloat(data.confidence);
        resultCard.classList.remove('hidden');
        resultIcon.textContent = fake ? '\u26A0' : '\u2705';
        resultIcon.className = 'result-icon' + (fake ? ' fake' : '');
        resultBadge.textContent = fake ? 'FAKE NEWS' : 'REAL NEWS';
        resultBadge.className = 'result-badge ' + (fake ? 'fake' : 'real');
        resultModelName.textContent = data.model_used;
        confidenceValue.textContent = conf.toFixed(1) + '%';
        confidenceFill.style.width = '0%';
        confidenceFill.className = 'confidence-fill' + (fake ? ' fake' : '');
        setTimeout(() => { confidenceFill.style.width = conf + '%'; }, 80);
        resultPreview.textContent = data.text_preview || '';
        if (data.reasons && data.reasons.length > 0) {
            reasonsContainer.classList.remove('hidden');
            reasonsList.innerHTML = '';
            data.reasons.forEach(r => {
                const span = document.createElement('span');
                span.className = `reason-token ${r.direction}`;
                span.textContent = `${r.word} (${r.impact.toFixed(2)})`;
                reasonsList.appendChild(span);
            });
        } else { reasonsContainer.classList.add('hidden'); }
        resultCard.scrollIntoView({behavior: 'smooth', block: 'center'});
    }

    function showCompare(data) {
        compareCard.classList.remove('hidden');
        compareResults.innerHTML = '';
        data.results.forEach(r => {
            const fake = r.prediction === 'Fake';
            const div = document.createElement('div');
            div.className = 'compare-item ' + (fake ? 'fake-result' : 'real-result');
            div.innerHTML = `<h4>${r.model}</h4><div class="pred ${fake?'fake':'real'}">${r.prediction}</div><div class="conf">${r.confidence}%</div>`;
            compareResults.appendChild(div);
        });
        compareCard.scrollIntoView({behavior: 'smooth', block: 'center'});
    }

    function showBatch(data) {
        batchCard.classList.remove('hidden');
        batchResults.innerHTML = '';
        data.results.forEach(r => {
            const fake = r.prediction === 'Fake';
            const div = document.createElement('div');
            div.className = 'batch-item ' + (fake ? 'fake-result' : 'real-result');
            div.innerHTML = `<span class="batch-text" dir="auto">${r.text}</span><span class="batch-tag ${fake?'fake':'real'}">${r.prediction} ${r.confidence}%</span>`;
            batchResults.appendChild(div);
        });
        batchCard.scrollIntoView({behavior: 'smooth', block: 'center'});
    }

    // Stats
    async function updateStats() {
        try {
            const res = await fetch('/stats');
            const data = await res.json();
            document.getElementById('stat-total').innerHTML = data.total + ' <small>' + (isArabic ? 'تم تحليلها' : 'analyzed') + '</small>';
            document.getElementById('stat-real').innerHTML = data.real + ' <small>Real</small>';
            document.getElementById('stat-fake').innerHTML = data.fake + ' <small>Fake</small>';
        } catch(e) {}
    }

    // Fetch news
    fetchBtn.addEventListener('click', async () => {
        fetchBtn.disabled = true;
        newsList.innerHTML = '<p class="loading-text">Fetching news...</p>';
        try {
            const res = await fetch('/fetch-news', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source: selectedSource})
            });
            const data = await res.json();
            if (data.error) newsList.innerHTML = '<p class="loading-text">Error: ' + data.error + '</p>';
            else renderNews(data.articles);
        } catch(e) { newsList.innerHTML = '<p class="loading-text">Failed to fetch</p>'; }
        fetchBtn.disabled = false;
    });

    fetchUrlBtn.addEventListener('click', async () => {
        const url = customUrlInput.value.trim();
        if (!url) { alert('Enter a URL'); return; }
        fetchUrlBtn.disabled = true;
        newsList.innerHTML = '<p class="loading-text">Fetching...</p>';
        try {
            const res = await fetch('/fetch-url', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url})
            });
            const data = await res.json();
            if (data.error) newsList.innerHTML = '<p class="loading-text">Error: ' + data.error + '</p>';
            else renderNews(data.articles);
        } catch(e) { newsList.innerHTML = '<p class="loading-text">Failed</p>'; }
        fetchUrlBtn.disabled = false;
    });

    function renderNews(articles) {
        if (!articles.length) { newsList.innerHTML = '<p class="loading-text">No articles found</p>'; return; }
        newsList.innerHTML = '';
        articles.forEach(art => {
            const div = document.createElement('div');
            div.className = 'news-item';
            div.innerHTML = `<div class="news-text"><p class="news-title" dir="auto">${art.title}</p>${art.link ? '<a href="'+art.link+'" target="_blank" class="news-link">Open</a>' : ''}</div><button class="check-btn" data-text="${art.text.replace(/"/g, '&quot;')}">Check</button>`;
            newsList.appendChild(div);
        });
        document.querySelectorAll('.check-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const text = btn.dataset.text;
                btn.disabled = true;
                btn.textContent = '...';
                try {
                    const res = await fetch('/analyze', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, model: selectedModel}) });
                    const data = await res.json();
                    const fake = data.prediction === 'Fake';
                    btn.textContent = data.prediction + ' ' + parseFloat(data.confidence).toFixed(0) + '%';
                    btn.classList.add(fake ? 'fake' : 'real');
                    btn.parentElement.classList.add(fake ? 'news-fake' : 'news-real');
                    updateStats();
                } catch(e) { btn.textContent = 'Error'; }
                btn.disabled = true;
            });
        });
    }

    // Load all charts
    async function loadAll() {
        try {
            const [mRes, fRes] = await Promise.all([fetch('/models'), fetch('/features')]);
            const mData = await mRes.json();
            const fData = await fRes.json();
            renderBarChart(mData);
            renderRadar(mData);
            renderPie(mData);
            renderCM(mData);
            renderTable(mData);
            renderKeywords(fData);
        } catch(e) { console.error(e); }
    }

    function renderBarChart(data) {
        if (!data.models) return;
        const names = Object.keys(data.models);
        const accs = names.map(n => data.models[n].accuracy);
        const colors = ['#4a6cf7','#7c3aed','#ec4899','#22c55e','#f59e0b'];
        if (accChart) accChart.destroy();
        accChart = new Chart(document.getElementById('accuracyChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ label: 'Accuracy %', data: accs, backgroundColor: colors, borderRadius: 6 }] },
            options: { responsive: true, plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true, max: 105, ticks: { callback: v=>v+'%' } }, x: { ticks: { font: { weight: '600' } } } }
            }
        });
    }

    function renderRadar(data) {
        if (!data.models) return;
        const names = Object.keys(data.models);
        const colors = ['rgba(74,108,247,0.7)','rgba(124,58,237,0.7)','rgba(236,72,153,0.7)','rgba(34,197,94,0.7)','rgba(245,158,11,0.7)'];
        const bg = ['rgba(74,108,247,0.1)','rgba(124,58,237,0.1)','rgba(236,72,153,0.1)','rgba(34,197,94,0.1)','rgba(245,158,11,0.1)'];
        const ds = names.map((n,i) => ({
            label: n,
            data: [data.models[n].accuracy, data.models[n].precision, data.models[n].recall, data.models[n].f1, data.models[n].cv_mean],
            borderColor: colors[i], backgroundColor: bg[i], pointRadius: 3
        }));
        if (radChart) radChart.destroy();
        radChart = new Chart(document.getElementById('radarChart'), {
            type: 'radar',
            data: { labels: ['Accuracy','Precision','Recall','F1','CV Mean'], datasets: ds },
            options: { responsive: true, scales: { r: { beginAtZero: true, max: 105, ticks: { display: false } } } }
        });
    }

    function renderPie(data) {
        if (!data.dataset) return;
        if (pieChart) pieChart.destroy();
        pieChart = new Chart(document.getElementById('pieChart'), {
            type: 'doughnut',
            data: {
                labels: ['Real News', 'Fake News'],
                datasets: [{ data: [data.dataset.real_count, data.dataset.fake_count], backgroundColor: ['#22c55e','#ef4444'], borderWidth: 0 }]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
        });
    }

    function renderCM(data) {
        const grid = document.getElementById('cm-grid');
        if (!data.confusion_matrices) return;
        grid.innerHTML = '';
        for (const [name, cm] of Object.entries(data.confusion_matrices)) {
            const m = cm.matrix;
            const best = name === data.best_model;
            const div = document.createElement('div');
            div.className = 'cm-item' + (best ? ' best' : '');
            div.innerHTML = `<h3>${name}${best?' (Best)':''}</h3>
                <table class="cm-table"><thead><tr><th></th><th>Pred Real</th><th>Pred Fake</th></tr></thead>
                <tbody><tr><td class="cm-label">Real</td><td class="cm-tp">${m[0][0]}</td><td class="cm-fp">${m[0][1]}</td></tr>
                <tr><td class="cm-label">Fake</td><td class="cm-fn">${m[1][0]}</td><td class="cm-tn">${m[1][1]}</td></tr></tbody></table>`;
            grid.appendChild(div);
        }
    }

    function renderTable(data) {
        const tbody = document.getElementById('metrics-tbody');
        const info = document.getElementById('dataset-info');
        if (!data.models) return;
        tbody.innerHTML = '';
        for (const [n, m] of Object.entries(data.models)) {
            const best = n === data.best_model;
            const tr = document.createElement('tr');
            if (best) tr.classList.add('best-row');
            tr.innerHTML = `<td>${n}${best?' *':''}</td><td>${m.accuracy}%</td><td>${m.precision}%</td><td>${m.recall}%</td><td>${m.f1}%</td><td>${m.cv_mean}%</td><td>${m.cv_std}%</td>`;
            tbody.appendChild(tr);
        }
        if (data.dataset) {
            const d = data.dataset;
            info.innerHTML = `<div class="info-chip">Total: <span>${d.total}</span></div><div class="info-chip">Train: <span>${d.train_size}</span></div><div class="info-chip">Test: <span>${d.test_size}</span></div><div class="info-chip">Real: <span>${d.real_count}</span></div><div class="info-chip">Fake: <span>${d.fake_count}</span></div>`;
        }
    }

    function renderKeywords(data) {
        const el = document.getElementById('keywords-content');
        const m = data['Logistic Regression'] || data[Object.keys(data)[0]];
        if (!m) return;
        let html = '<div class="keywords-grid">';
        if (m.fake_words) {
            html += '<div class="kw-col"><h3 class="kw-fake">Fake News Indicators</h3>';
            m.fake_words.forEach((w,i) => { html += `<div class="kw-item fake"><span class="kw-rank">${i+1}</span><span class="kw-word" dir="auto">${w.word}</span><span class="kw-score">${w.score.toFixed(3)}</span></div>`; });
            html += '</div>';
        }
        if (m.real_words) {
            html += '<div class="kw-col"><h3 class="kw-real">Real News Indicators</h3>';
            m.real_words.forEach((w,i) => { html += `<div class="kw-item real"><span class="kw-rank">${i+1}</span><span class="kw-word" dir="auto">${w.word}</span><span class="kw-score">${w.score.toFixed(3)}</span></div>`; });
            html += '</div>';
        }
        html += '</div>';
        el.innerHTML = html;
    }

    loadAll();
    updateStats();
});
