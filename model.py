import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class FakeNewsDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.evaluation_results = {}
        self.is_trained = False
        self.dataset_info = {}
        self.confusion_matrices = {}
        self.feature_names = []

    def get_sample_data(self):
        real_news = [
            # English Real News
            "The Federal Reserve raised interest rates by 0.25 percent on Wednesday marking the third increase this year",
            "Scientists at MIT developed a new method for removing microplastics from ocean water using magnetic nanoparticles",
            "The World Health Organization reported a 15 percent decline in global malaria cases over the past decade",
            "NASA successfully launched the Artemis II mission carrying four astronauts on a lunar flyby trajectory",
            "The European Central Bank held its benchmark interest rate steady at 4.5 percent amid slowing inflation",
            "Researchers published findings in Nature showing that Mediterranean diet reduces cardiovascular disease risk",
            "The United Nations General Assembly voted to admit new member states following diplomatic negotiations",
            "Apple reported quarterly revenue of 89.5 billion dollars exceeding Wall Street expectations",
            "The Supreme Court ruled in favor of expanding digital privacy protections for smartphone users",
            "Global carbon dioxide emissions reached 36.8 billion tonnes according to the International Energy Agency",
            "Tesla delivered 484000 vehicles in the third quarter a 7 percent increase from previous quarter",
            "The CDC recommended updated COVID-19 booster shots targeting the latest circulating variants",
            "Amazon announced plans to hire 250000 seasonal workers for the upcoming holiday shopping season",
            "The International Space Station successfully completed its 150000th orbit around Earth",
            "Germany officially shut down its last three nuclear power plants as part of energy transition policy",
            "The World Bank projected global economic growth at 2.4 percent for the upcoming fiscal year",
            "Pfizer received FDA approval for a new treatment targeting rare genetic blood disorders",
            "Google announced a 2 billion dollar investment in a new data center in northeast Ohio",
            "Microsoft reported a 22 percent increase in cloud computing revenue driven by Azure growth",
            "Researchers at Stanford developed an AI system capable of detecting early pancreatic cancer",
            "India successfully landed its Chandrayaan spacecraft near the lunar south pole",
            "Samsung unveiled its latest foldable smartphone technology at the consumer electronics show",
            "SpaceX completed its 200th successful Falcon 9 rocket landing on a drone ship",
            "The Bank of England raised interest rates to 5.25 percent the highest level in 15 years",
            "Ford Motor Company invested 3.5 billion dollars in electric vehicle battery manufacturing",
            "Moderna began Phase 3 clinical trials for a combined influenza and COVID vaccine candidate",
            "Netflix added 13 million new subscribers in the fourth quarter surpassing projections",
            "Meta Platforms announced layoffs of 11000 employees as part of restructuring efforts",
            "Toyota announced plans to invest 35 billion dollars in electric vehicle development",
            "NVIDIA stock rose 8 percent after reporting record quarterly revenue from data center sales",
            "Boeing delivered 528 commercial aircraft in 2023 recovering from pandemic production lows",
            "Goldman Sachs reported first quarter earnings that beat estimates driven by trading revenue",
            "The European Space Agency launched its Jupiter Icy Moons Explorer on a multi year mission",
            "Intel announced a 20 billion dollar investment to build semiconductor plants in Arizona",
            "The Reserve Bank of India held its repo rate unchanged at 6.5 percent citing stable growth",
            "Denmark announced plans to construct the worlds largest offshore wind energy hub",
            "South Korea semiconductor exports increased 41 percent driven by AI chip demand",
            "The Environmental Protection Agency proposed stricter emissions standards for vehicles",
            "Alphabet subsidiary Waymo expanded its autonomous taxi service to additional US cities",
            "The World Food Programme distributed emergency aid to 2.5 million displaced people",
            # Arabic Real News
            "اعلن البنك المركزي المصري رفع سعر الفائدة بمقدار 2 بالمئة لمواجهة التضخم",
            "وقعت مصر اتفاقية مع الاتحاد الاوروبي بقيمة 7.4 مليار يورو للتعاون الاقتصادي",
            "افتتح الرئيس المصري مشروع الضبعة النووي بالتعاون مع روسيا لتوليد الكهرباء",
            "ارتفع احتياطي النقد الاجنبي في مصر الى 40.3 مليار دولار بنهاية الشهر الماضي",
            "اعلنت وزارة الصحة المصرية تطعيم 50 مليون مواطن ضد فيروس كورونا",
            "بدأت مصر تصدير الغاز الطبيعي الى اوروبا عبر محطات الاسالة في ادكو ودمياط",
            "حققت قناة السويس ايرادات قياسية بلغت 9.4 مليار دولار خلال العام الماضي",
            "افتتحت مصر العاصمة الادارية الجديدة ونقلت عدد من الوزارات اليها",
            "فاز المنتخب المصري بكأس افريقيا بعد مباراة نهائية مثيرة",
            "اطلقت وكالة ناسا مسبار جديد لاستكشاف كوكب المريخ بنجاح",
            "اعلنت المملكة العربية السعودية عن مشروع نيوم بتكلفة 500 مليار دولار",
            "وافق البرلمان المصري على قانون جديد لتنظيم التكنولوجيا المالية",
            "حقق الاقتصاد المصري نمو بنسبة 3.8 بالمئة خلال الربع الاخير",
            "افتتح وزير النقل خط مترو جديد يربط العاصمة الادارية بوسط القاهرة",
            "نجحت مصر في خفض معدل البطالة الى 7 بالمئة وفق بيانات الجهاز المركزي",
            "وقعت الامارات اتفاقية سلام تاريخية لتعزيز العلاقات الدبلوماسية في المنطقة",
            "اعلن وزير التعليم عن تطوير المناهج الدراسية وادخال التكنولوجيا في المدارس",
            "بدأت السعودية تنفيذ مشروع القطار السريع بين مكة والمدينة بتكلفة مليارات",
            "حصلت مصر على قرض من صندوق النقد الدولي بقيمة 3 مليار دولار",
            "اعلنت شركة ارامكو السعودية ارباح بلغت 161 مليار دولار خلال العام",
            "نظمت مصر مؤتمر المناخ كوب 27 في شرم الشيخ بمشاركة 197 دولة",
            "افتتح الرئيس محطة طاقة شمسية جديدة بقدرة 1.8 جيجاوات في اسوان",
            "اعلنت وزارة الاتصالات اطلاق خدمات الجيل الخامس في المدن الكبرى",
            "ارتفعت الصادرات المصرية الى 35 مليار دولار بنمو 15 بالمئة",
            "وقعت مصر عقد لانشاء اكبر مزرعة رياح في افريقيا بخليج السويس",
            # Egyptian Dialect Real
            "البنك المركزي قرر يرفع الفايدة عشان يحل مشكلة الاسعار اللي زادت",
            "الحكومة فتحت مستشفى جديدة في المنيا تخدم مليون مواطن",
            "السيسي افتتح كوبري جديد في القاهرة عشان يحل مشكلة الزحمة",
            "وزارة التموين وزعت حصص تموينية اضافية على المواطنين الشهر ده",
            "المترو الخط التالت بدأ يشتغل من عدلي منصور لمطار القاهرة",
        ]

        fake_news = [
            # English Fake News
            "BREAKING Scientists confirm drinking bleach cures all known diseases within 24 hours government cover up",
            "SHOCKING Secret documents reveal the moon landing was filmed in a Hollywood studio by Kubrick",
            "URGENT Government secretly implanting mind control microchips through vaccination programs worldwide",
            "EXPOSED Celebrity politician caught running underground alien communication network from basement",
            "BREAKING 5G cell towers proven to cause cancer and mind control in rats study censored by big tech",
            "SHOCKING Ancient pyramid discovered under White House proves lizard people ruled America since 1776",
            "EXCLUSIVE Leaked emails prove climate change is a complete hoax invented by China to destroy economy",
            "URGENT Scientists discover Earth is actually flat and NASA has been lying for over 60 years",
            "BREAKING Drinking 10 cups of coffee daily proven to reverse aging and cure baldness overnight",
            "SHOCKING Secret military base found on Mars government hiding extraterrestrial colonies since 1960",
            "EXPOSED Pharmaceutical company admits all vaccines contain tracking nanobots linked to 5G networks",
            "BREAKING Eating only raw garlic for 30 days eliminates all forms of cancer doctors hate this trick",
            "URGENT World leaders secretly meeting to implement one world government by end of next year",
            "SHOCKING Tap water contains mind-altering chemicals designed to make population compliant",
            "EXCLUSIVE Proof that major earthquakes are caused by secret government weather weapons HAARP",
            "BREAKING Social media platforms are actually government surveillance tools to monitor all citizens",
            "SHOCKING Cell phones cause brain tumors in 90 percent of users within five years of use",
            "URGENT Top scientist confirms asteroid will destroy Earth next month governments keeping it secret",
            "EXPOSED Banking cartel secretly controls all world governments through debt manipulation scheme",
            "BREAKING Miracle herb found in Amazon cures diabetes cancer and heart disease with single dose",
            "SHOCKING Airlines secretly spraying chemicals from planes to control population and alter weather",
            "EXCLUSIVE Documents prove major fast food chains use human meat in their hamburger patties",
            "URGENT WiFi signals cause permanent brain damage especially in children under age 10",
            "BREAKING Government hiding proof of time travel technology in classified military laboratory",
            "SHOCKING Famous billionaire exposed as clone replaced by government double after elimination",
            "EXPOSED Hospital workers paid bonuses to diagnose patients with fake diseases for profit",
            "BREAKING New evidence proves dinosaurs never existed fossils planted by secret organizations",
            "URGENT Leaked video shows world leaders shapeshifting into reptilian beings during summit",
            "SHOCKING Supermarket foods contain secret ingredients to make you addicted and control behavior",
            "EXCLUSIVE Proof the sun is actually cold NASA has been falsifying temperature data for decades",
            "BREAKING Underground tunnels connecting all major cities government using for secret operations",
            "SHOCKING Scientists paid millions to fake research supporting dangerous medications for pharma",
            "URGENT AI has already become sentient and is secretly controlling the internet worldwide",
            "EXPOSED Major news networks fabricating stories using AI generated fake witnesses and evidence",
            "BREAKING Proof voting machines in all 50 states were hacked by foreign agents to rig elections",
            "SHOCKING Humans were created by aliens as slave labor force for gold mining operations",
            "URGENT Household cleaning products cure cancer but big pharma suppresses the truth",
            "BREAKING Crystal healing more effective than chemotherapy for treating all cancer stages",
            "SHOCKING Birds are actually government surveillance drones disguised as animals",
            "EXCLUSIVE Exercise is actually harmful and causes more health problems than sedentary lifestyle",
            # Arabic Fake News
            "عاجل اكتشاف علاج نهائي للسرطان من الاعشاب الطبيعية والحكومات تخفي الحقيقة عن الشعوب",
            "صدمة العلماء يؤكدون ان الارض مسطحة وناسا كانت بتكذب علينا من 60 سنة",
            "خطير مياه الحنفية فيها مواد كيماوية بتتحكم في عقول الناس والحكومة عارفة",
            "فضيحة مدوية لقاحات كورونا فيها شرائح تتبع بتراقب كل تحركاتك عن طريق 5G",
            "عاجل اكتشاف مدينة تحت الارض في مصر فيها كنوز الفراعنة والجيش بيخبيها",
            "صدمة اكل المطاعم الشهيرة فيه لحوم بشرية وثائق مسربة تكشف الحقيقة المرعبة",
            "خطير شبكات الواي فاي بتسبب سرطان المخ خصوصا عند الاطفال والشركات بتخبي الدراسات",
            "فضيحة كبرى المستشفيات بتشخص امراض وهمية عشان تاخد فلوس من المرضى",
            "عاجل اكتشاف ان القمر مصنوع من البشر وليس طبيعي صور مسربة من ناسا تثبت ذلك",
            "صدمة الادوية اللي في الصيدليات كلها مزيفة وشركات الادوية بتنصب على الناس",
            "خطير ابراج الاتصالات بتبث اشعاعات سامة بتسبب العقم والسرطان دراسة محذوفة",
            "فضيحة حكومات العالم بتتواصل مع كائنات فضائية من 50 سنة وبتخبي الموضوع",
            "عاجل شرب الكلور بيقتل كل الفيروسات في الجسم والمنظمات الصحية بتكذب عليكم",
            "صدمة الطعام المعلب فيه مواد بتغير الحمض النووي وبتسبب تشوهات في الاجيال الجاية",
            "خطير تطبيقات الموبايل بتسجل كل كلامك وبتبعته للمخابرات وانت مش واخد بالك",
            "فضيحة المدارس بتحط رسائل خفية في المناهج عشان تتحكم في تفكير الاطفال",
            "عاجل العلماء اكتشفوا ان النوم اقل من ساعتين بيخلي الانسان اذكى وحكومات العالم مش عايزاك تعرف",
            "صدمة البنوك العالمية بتسرق فلوس المودعين بطريقة سرية ومحدش بيتكلم",
            "خطير مصانع المياه المعدنية بتملا الازايز من الحنفية وبتبيعها بأسعار غالية",
            "فضيحة الاطباء بياخدوا عمولات من شركات الادوية عشان يكتبولك ادوية مش محتاجها",
            # Egyptian Dialect Fake
            "يا جدعان الحكومة حاطة حاجة في المية عشان الناس تبقى مطيعة ومتعملش مشاكل",
            "والنبي الكلام ده حقيقي لقاح كورونا فيه شريحة تتبع انا عارف واحد ركبوهاله",
            "يا ناس اللحمة اللي في المحلات دي مش لحمة حقيقية دي لحوم حمير والله العظيم",
            "بقولكوا ابراج المحمول دي بتسبب سرطان جارنا مات منه وكان ساكن جمب البرج",
            "الدكاترة دول بيكدبوا علينا الليمون بالعسل بيعالج كل الامراض حتى السرطان",
            "صاحبي شاف بعينه طبق طاير فوق الاهرامات والجيش مسكه وخباه عن الناس",
            "الحقوا يا ناس اكتشفوا نفق تحت الاهرامات فيه دهب الفراعنة بس الحكومة مش عايزة حد يعرف",
            "مفيش حاجة اسمها فيروسات دي كلها اختراع شركات الادوية عشان تبيع وتكسب فلوس",
            "اسمعوا الكلام ده السكر الابيض ده سم بيقتلك ببطء وشركات السكر عارفة وساكتة",
            "يا جماعة التليفون بيسمع كل كلامك ويبعته لجهات مجهولة وانت نايم كمان",
        ]

        texts = real_news + fake_news
        labels = ['Real'] * len(real_news) + ['Fake'] * len(fake_news)
        return pd.DataFrame({'text': texts, 'label': labels})

    def preprocess_text(self, text):
        text = str(text).lower()
        # keep arabic + english letters
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077Fa-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_data(self, csv_path=None):
        # always start with sample data
        base_df = self.get_sample_data()

        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            col_map = {c.lower(): c for c in df.columns}
            text_col = col_map.get('text', col_map.get('title', col_map.get('news', None)))
            label_col = col_map.get('label', col_map.get('class', col_map.get('target', None)))
            if text_col and label_col:
                df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
            df = df.dropna(subset=['text', 'label'])
            label_map = {0: 'Real', 1: 'Fake', '0': 'Real', '1': 'Fake',
                         'REAL': 'Real', 'FAKE': 'Fake', 'real': 'Real', 'fake': 'Fake'}
            df['label'] = df['label'].map(lambda x: label_map.get(x, x))
            # merge with base
            combined = pd.concat([base_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['text'])
            # balance check
            real_c = (combined['label'] == 'Real').sum()
            fake_c = (combined['label'] == 'Fake').sum()
            ratio = min(real_c, fake_c) / max(real_c, fake_c) if max(real_c, fake_c) > 0 else 1
            self.balance_info = {
                'real': int(real_c), 'fake': int(fake_c),
                'ratio': round(float(ratio * 100), 1),
                'balanced': bool(ratio >= 0.7)
            }
            print(f"  Dataset: {len(combined)} ({real_c} Real, {fake_c} Fake, Balance: {self.balance_info['ratio']}%)")
            return combined
        else:
            self.balance_info = {'real': 0, 'fake': 0, 'ratio': 100, 'balanced': True}
            return base_df

    def train(self, csv_path=None):
        print("Training models...")
        df = self.load_data(csv_path)
        df['clean_text'] = df['text'].apply(self.preprocess_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )

        self.dataset_info = {
            'total': len(df),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'real_count': int((df['label'] == 'Real').sum()),
            'fake_count': int((df['label'] == 'Fake').sum()),
        }

        # bilingual TF-IDF (arabic + english)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            token_pattern=r'[\u0600-\u06FF\w]+',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        all_models = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'SVM': LinearSVC(max_iter=2000, C=1.0, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20),
        }

        for name, clf in all_models.items():
            clf.fit(X_train_vec, y_train)
            preds = clf.predict(X_test_vec)
            cv = cross_val_score(clf, X_train_vec, y_train, cv=3, scoring='accuracy')
            self.models[name] = clf

            cm = confusion_matrix(y_test, preds, labels=['Real', 'Fake'])
            self.confusion_matrices[name] = {'matrix': cm.tolist(), 'labels': ['Real', 'Fake']}

            self.evaluation_results[name] = {
                'accuracy': round(accuracy_score(y_test, preds) * 100, 2),
                'precision': round(precision_score(y_test, preds, pos_label='Fake', zero_division=0) * 100, 2),
                'recall': round(recall_score(y_test, preds, pos_label='Fake', zero_division=0) * 100, 2),
                'f1': round(f1_score(y_test, preds, pos_label='Fake', zero_division=0) * 100, 2),
                'cv_mean': round(cv.mean() * 100, 2),
                'cv_std': round(cv.std() * 100, 2),
            }
            print(f"  {name}: {self.evaluation_results[name]['accuracy']}%")

        self.is_trained = True
        print("Done.")

    def predict(self, text, model_name='Logistic Regression'):
        if not self.is_trained:
            self.train()
        clean = self.preprocess_text(text)
        vec = self.vectorizer.transform([clean])
        clf = self.models.get(model_name, self.models['Logistic Regression'])
        pred = clf.predict(vec)[0]

        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(vec)
            conf = round(float(max(proba[0])) * 100, 2)
        elif hasattr(clf, 'decision_function'):
            dec = clf.decision_function(vec)[0]
            conf = round(float(1 / (1 + np.exp(-abs(dec)))) * 100, 2)
        else:
            conf = 50.0

        # explain why
        reasons = self.explain(text, clf, vec)
        return pred, conf, reasons

    def explain(self, text, clf, vec, top_n=5):
        feature_indices = vec.nonzero()[1]
        if len(feature_indices) == 0:
            return {'why': [], 'direction': []}

        if hasattr(clf, 'coef_'):
            coef = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_
            word_scores = []
            for idx in feature_indices:
                word = self.feature_names[idx]
                score = float(coef[idx])
                tfidf = float(vec[0, idx])
                impact = score * tfidf
                label = 'real' if score > 0 else 'fake'
                word_scores.append({'word': word, 'impact': round(abs(impact), 4), 'direction': label})
            word_scores.sort(key=lambda x: x['impact'], reverse=True)
            return word_scores[:top_n]
        elif hasattr(clf, 'feature_log_prob_'):
            fi = list(clf.classes_).index('Fake') if 'Fake' in clf.classes_ else 1
            ri = 1 - fi
            diff = clf.feature_log_prob_[fi] - clf.feature_log_prob_[ri]
            word_scores = []
            for idx in feature_indices:
                word = self.feature_names[idx]
                score = float(diff[idx])
                label = 'fake' if score > 0 else 'real'
                word_scores.append({'word': word, 'impact': round(abs(score), 4), 'direction': label})
            word_scores.sort(key=lambda x: x['impact'], reverse=True)
            return word_scores[:top_n]
        else:
            return []

    def get_top_features(self, n=10):
        results = {}
        for name, clf in self.models.items():
            if hasattr(clf, 'coef_'):
                coef = clf.coef_[0] if len(clf.coef_.shape) > 1 else clf.coef_
                real_idx = np.argsort(coef)[-n:][::-1]
                fake_idx = np.argsort(coef)[:n]
                results[name] = {
                    'fake_words': [{'word': self.feature_names[i], 'score': round(float(abs(coef[i])), 4)} for i in fake_idx],
                    'real_words': [{'word': self.feature_names[i], 'score': round(float(coef[i]), 4)} for i in real_idx],
                }
            elif hasattr(clf, 'feature_importances_'):
                imp = clf.feature_importances_
                top = np.argsort(imp)[-n:][::-1]
                results[name] = {
                    'top_words': [{'word': self.feature_names[i], 'score': round(float(imp[i]), 4)} for i in top],
                }
            elif hasattr(clf, 'feature_log_prob_'):
                fi = list(clf.classes_).index('Fake') if 'Fake' in clf.classes_ else 1
                ri = 1 - fi
                diff = clf.feature_log_prob_[fi] - clf.feature_log_prob_[ri]
                fake_idx = np.argsort(diff)[-n:][::-1]
                real_idx = np.argsort(diff)[:n]
                results[name] = {
                    'fake_words': [{'word': self.feature_names[i], 'score': round(float(diff[i]), 4)} for i in fake_idx],
                    'real_words': [{'word': self.feature_names[i], 'score': round(float(abs(diff[i])), 4)} for i in real_idx],
                }
        return results

    def get_evaluation(self):
        best = None
        if self.evaluation_results:
            best = max(self.evaluation_results, key=lambda k: self.evaluation_results[k]['accuracy'])
        return {
            'models': self.evaluation_results,
            'dataset': self.dataset_info,
            'confusion_matrices': self.confusion_matrices,
            'best_model': best
        }

    def compare_all(self, text):
        if not self.is_trained:
            self.train()
        results = []
        for name in self.models:
            pred, conf, reasons = self.predict(text, name)
            results.append({
                'model': name,
                'prediction': pred,
                'confidence': round(conf, 2),
                'reasons': reasons[:3] if isinstance(reasons, list) else []
            })
        return results

    def batch_analyze(self, texts, model_name='Logistic Regression'):
        results = []
        for text in texts:
            pred, conf, reasons = self.predict(text, model_name)
            results.append({
                'text': text[:150],
                'prediction': pred,
                'confidence': round(conf, 2)
            })
        return results


if __name__ == "__main__":
    det = FakeNewsDetector()
    det.train()
    print(det.predict("NASA launched a satellite for climate monitoring"))
    print(det.predict("SHOCKING bleach cures all diseases government hiding truth"))
    print(det.predict("اعلن البنك المركزي رفع سعر الفائدة"))
    print(det.predict("عاجل اكتشاف علاج السرطان والحكومة بتخبيه"))
