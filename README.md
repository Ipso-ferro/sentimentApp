# 🧠 SentimentApp — Simple NLP Sentiment Classifier (TF-IDF + Logistic Regression)

A tiny, beginner-friendly web app that trains a sentiment model on plain text reviews and serves a UI + JSON API.  
It expects a folder of categories (e.g., `books/`, `dvd/`, `electronics/`), each containing `positive.review` and `negative.review`.

**Stack:** Python, scikit-learn, Flask • **Model:** TF-IDF (uni+bi-grams) → Logistic Regression • **Deploy:** AWS (App Runner or Elastic Beanstalk)

---

## ✨ Features

- Ingests reviews from files; each review is a line or a paragraph (blank-line separated)
- Trains a **TF-IDF** + **Logistic Regression** baseline with probabilities
- Saves model to `sentiment_model.joblib`
- Minimal **Flask** UI (Bootstrap) + **POST /classify** endpoint that returns:
  ```json
  { "sentiment": "positive", "confidence": 0.95 }
  ```
- "Auto-train on first request" if the model file is missing
- One-file trainer (`train.py`) and one-file app (`app.py`)

## 📁 Project Structure

```
backEnd_nlpSentimentApp/
├─ app.py
├─ train.py
├─ requirements.txt
├─ sentiment_model.joblib        # created after training (or auto-trained)
└─ data/
   ├─ books/
   │  ├─ positive.review
   │  └─ negative.review
   ├─ dvd/
   │  ├─ positive.review
   │  └─ negative.review
   ├─ electronics/
   │  ├─ positive.review
   │  └─ negative.review
   └─ kitchen_&_housewares/
      ├─ positive.review
      └─ negative.review
```

> **Note:** `unlabeled.review` files (if present) are ignored by this baseline.

**File format:** Each review may be one line or a block; blocks can be separated by a blank line.

## 🔧 Setup (Local)

**Requirements:**
- Python 3.10+ recommended
- macOS/Linux/WSL/Windows

```bash
cd backEnd_nlpSentimentApp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt** (keep it minimal):
```
flask
scikit-learn
joblib
```

## 🏋️ Train

Train once (creates `sentiment_model.joblib`):

```bash
python train.py
```

You'll see accuracy + a classification report.  
If you skip this, `app.py` will train automatically on first request (slower first run).

## 🚀 Run the App (Local)

```bash
python app.py
# open http://127.0.0.1:5000
```

### Endpoints

- **GET /** – HTML page with a textbox and a "Classify" button
- **POST /classify** – JSON body `{ "text": "..." }` → returns `{ "sentiment": "...", "confidence": 0.xx }`
- **POST /train** – Force a retrain from the current `data/` (handy after you add reviews)

**Example request:**

```bash
curl -s -X POST http://127.0.0.1:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"This camera is fantastic for the price"}'
```

## 🧪 NLP Technique (What the model does)

This is a classic, transparent baseline that performs surprisingly well on product reviews:

### Ingestion
- Reads `positive.review` as label 1 and `negative.review` as label 0 from each category folder
- Splits by blank lines; if too few chunks, falls back to per-line

### Vectorization: TF-IDF
```python
TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=50k, min_df=2, max_df=0.9)
```
- Uni-grams + bi-grams capture phrases like "not good"

### Classifier: Logistic Regression
- `class_weight="balanced"` (helps if positives/negatives are imbalanced)
- `max_iter=2000, solver="liblinear"`
- `predict_proba` gives calibrated-ish probabilities for confidence

### Eval
- `train_test_split(test_size=0.2, stratify=y)`
- Metrics: accuracy + precision/recall/F1 via `classification_report`

### Why this baseline?
Fast to train, low memory, interpretable features (you can inspect top coefficients), deploys easily. Easy to improve incrementally.

### Ideas to Improve Later
- Add tri-grams, prune stopwords, tune C regularization
- Use LinearSVC + CalibratedClassifierCV for potentially better margins with reliable probabilities
- Domain adaptation: train separate models per category or add a domain feature
- Modernize: fine-tune a small Transformer (e.g., DistilBERT) if you need higher accuracy and can tolerate larger footprint

## 🧭 Agile Plan (August 2025 Sprint)

**Goal:** Ship a working MVP that classifies sentiment from a browser and a JSON API, deployed on AWS.

**Sprint Timeline:** August 14-22, 2025 (2 weeks)

### User Stories
- As a user, can paste a comment and see the predicted sentiment and confidence in a simple web page
- As a developer,can post JSON to an API endpoint to get predictions programmatically
- As an operator, retrain the model after adding new data without redeploying code
- As a stakeholder,access the app via a public URL (HTTPS) on AWS

### Sprint Backlog

| # | Task | Status | Days |
|---|------|--------|------|
| 1 | Set up communication, project, virtual environment & dependencies (backEnd) | ✅ Complete | Aug 14 |
| 2 | Train model creating the new document .joblib to connect to the app.py | ✅ Complete | Aug 14-15 |
| 3 | Install dependencies | ✅ Complete | Aug 15 |
| 4 | Create machine learning | ✅ Complete | Aug 15 |
| 5 | Preprocess input text | ✅ Complete | Aug 16 |
| 6 | Test /predict with curl/Postman | ✅ Complete | Aug 16 |
| 7 | Flask Web Application in local | ✅ Complete | Aug 16 |
| 8 | Create environment to develop backend and split the project |✅ Complete| Aug 20 |
| 9 | Final code cleanup|✅ Complete| Aug 20 |
| 10 | Handle errors and edge cases|✅ Complete| Aug 21 |
| 11 | Write ethics.md |✅ Complete| Aug 21 |
| 12 | Record project demo video |✅ Complete| Aug 21 |
| 13 | Deploy the project in AWS |✅ Complete| Aug 21-22 |

### Sprint Goals by Week

**Week 1 (Aug 14-19):**
- ✅ Environment setup and dependency management
- ✅ Core ML model training and persistence
- 🔄 API endpoint development and testing
- 🔄 Local Flask application deployment

**Week 2 (Aug 20-22):**
- 📋 Code quality improvements and error handling
- 📋 Documentation and ethics guidelines
- 📋 Demo preparation and AWS deployment

### Definition of Done
- Model trains locally and saves `sentiment_model.joblib`
- POST /classify returns sentiment + confidence with HTTP 200
- Deployed URL reachable publicly; UI works end-to-end
- README explains setup, train, run, and deploy
- Basic error handling + sensible defaults; no secrets in repo
- Ethics documentation completed
- Demo video recorded

### Acceptance Criteria
- For input "awful support, totally broken and late" returns `{ "sentiment": "negative", ... }`
- For input "great value and super fast delivery" returns `{ "sentiment": "positive", ... }`
- First call on a fresh server finishes < 30s (auto-train), subsequent calls < 300ms (t2.small-ish)
- All API endpoints properly tested with curl/Postman

### Current Risks & Mitigation
- **Blocked items (ethics, demo, deployment)** → Prioritize unblocking dependencies; consider parallel work streams
- **Cold start/first-request training slow** → Pre-train locally and commit `sentiment_model.joblib` ✅
- **Data imbalance** → Keep `class_weight="balanced"`; monitor confusion matrix; add examples
- **Memory limits on small instances** → Cap `max_features` (50k is safe for small corpora)

## ☁️ Deploy on AWS

### Option A — App Runner (Docker, simplest managed)

**Dockerfile** in `backEnd_nlpSentimentApp/`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
```

**(Recommended)** Pre-train locally so the image includes `sentiment_model.joblib`:

```bash
python train.py
```

Push to ECR and create an App Runner service (port 8080).  
You'll get a public HTTPS URL. Open it; the HTML page is served from `/`.

### Option B — Elastic Beanstalk (no Docker)

Add `application.py` at repo root:

```python
from app import app as application
```

Deploy with EB CLI:

```bash
pip install awsebcli --upgrade --user
eb init  # choose Python 3.11 on AL2023
eb create sentiment-env --single
eb open
```

**Updates:** push a new image (App Runner) or `eb deploy` (EB).

## 🔐 Security & Privacy

- Do not log full texts in production (PII). If needed, log only length and a hash
- Add a max input size (e.g., 2–4 KB) to avoid abuse
- If exposing `/train`, consider protecting it (simple auth, IP allowlist, or remove in prod)

## 🧭 Roadmap

- Confidence calibration & threshold tuning
- Add a neutral class (requires labeled data)
- Export top features per class for explainability
- Batch classification endpoint
- Swap to LinearSVC + CalibratedClassifierCV or a small Transformer for higher accuracy
- GitHub Actions CI (lint, tests, Docker build, deploy)

## 🧪 Quick Test Snippets

```bash
# Local run
python app.py

# Classify via curl
curl -s -X POST http://127.0.0.1:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"great value and super fast delivery"}' | jq .

# Retrain after adding data
curl -X POST http://127.0.0.1:5000/train
```

## 📜 License

MIT (or your choice). Update this section to your actual license.

---

**Note:** If you want, I can also add a Dockerfile and a minimal GitHub Action (build → push to ECR → deploy to App Runner) tailored to your AWS account/region.
