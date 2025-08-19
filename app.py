# app.py - Flask Web Application for Sentiment Analysis
# This file creates a web interface for sentiment classification using machine learning

# STEP 1: Import required libraries
from __future__ import annotations  # Enable modern type hints
from pathlib import Path  # For file path operations
from typing import Dict  # For type hints

# Flask web framework imports
from flask import Flask, request, jsonify, Response
import joblib  # For saving/loading ML models

# STEP 2: Import machine learning libraries for training
import re  # Regular expressions for text processing
from sklearn.pipeline import Pipeline  # ML pipeline
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # Text to numbers conversion
from sklearn.linear_model import LogisticRegression  # Classification algorithm
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation

# STEP 3: Define file paths and constants
HERE = Path(__file__).parent  # Current directory where this file is located
DATA_ROOT = HERE / "data"  # Folder containing training data
MODEL_PATH = HERE / "sentiment_model.joblib"  # Where to save the trained model
LABELS = {0: "negative", 1: "positive"}  # Map numbers to sentiment labels
REVIEW_FILES = ("positive.review", "negative.review")  # Expected data file names

# STEP 4: Function to read review files and extract text data
def _read_reviews(path: Path, label: int):
    """Read reviews from a file and assign sentiment labels"""
    texts, labels = [], []  # Initialize empty lists
    if not path.exists():  # Check if file exists
        return texts, labels  # Return empty lists if no file

    # Read file content and normalize line endings
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    # Split text by blank lines (paragraph separation)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]

    # If few chunks found, split by individual lines instead
    if len(chunks) < 10:
        chunks = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    # Add each text chunk with its corresponding label
    for c in chunks:
        texts.append(c)  # Add review text
        labels.append(label)  # Add sentiment label (0=negative, 1=positive)
    return texts, labels


# STEP 5: Function to load all training data from multiple categories
def load_dataset(root: Path):
    """Load all review files from data directory structure"""
    X, y = [], []  # X = texts, y = labels

    # Look through all subdirectories (books, dvd, electronics, etc.)
    for category in root.glob("*"):
        if not category.is_dir():  # Skip if not a directory
            continue

        # Look for positive.review and negative.review files in each category
        for fname in REVIEW_FILES:
            # Determine label: positive files = 1, negative files = 0
            label = 1 if fname.startswith("positive") else 0

            # Read reviews from this file
            tx, lb = _read_reviews(category / fname, label)

            # Add all texts and labels to our dataset
            X.extend(tx)  # Add review texts
            y.extend(lb)  # Add corresponding labels

    # Check if we found any data
    if not X:
        raise RuntimeError(
            f"No reviews found under {root}. Expected {REVIEW_FILES} inside subfolders."
        )
    return X, y


# STEP 6: Create machine learning pipeline
def build_pipeline():
    """Build ML pipeline: text processing + classification"""
    return Pipeline(
        [
            # Step 1: Convert text to numerical features using TF-IDF
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,  # Convert to lowercase
                    ngram_range=(1, 2),  # Use single words and word pairs
                    max_features=50000,  # Keep top 50,000 most important words
                    min_df=2,  # Word must appear in at least 2 documents
                    max_df=0.9,  # Ignore words in more than 90% of documents
                ),
            ),
            # Step 2: Classify using Logistic Regression
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,  # Maximum training iterations
                    class_weight="balanced",  # Handle imbalanced data
                    solver="liblinear",  # Optimization algorithm
                ),
            ),
        ]
    )


# STEP 7: Train the machine learning model and save it
def train_and_save(data_root: Path = DATA_ROOT, model_path: Path = MODEL_PATH):
    """Complete training process: load data, train model, evaluate, save"""
    print(f"Loading data from: {data_root}")

    # Load all review data
    X, y = load_dataset(data_root)
    print(f"Loaded {len(X)} reviews.")

    # Split data: 80% for training, 20% for testing
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,  # 20% for testing
        random_state=42,  # For reproducible results
        stratify=y,  # Keep same ratio of positive/negative in both sets
    )

    # Create and train the model
    pipe = build_pipeline()
    print("Training model...")
    pipe.fit(X_tr, y_tr)  # Train on training data

    # Test the model performance
    print("Evaluating...")
    y_pred = pipe.predict(X_te)  # Make predictions on test data
    acc = accuracy_score(y_te, y_pred)  # Calculate accuracy
    print(f"Accuracy: {acc:.4f}")

    # Show detailed performance report
    print(classification_report(y_te, y_pred, target_names=[LABELS[0], LABELS[1]]))

    # Save the trained model to disk
    print(f"Saving model to: {model_path}")
    joblib.dump({"pipeline": pipe, "labels": LABELS}, model_path)


# STEP 8: Helper functions for using the trained model
# ---------------------- Model helpers --------------------------------------
def _ensure_model():
    """Make sure we have a trained model - train one if needed"""
    if not MODEL_PATH.exists():  # Check if model file exists
        if not DATA_ROOT.exists():  # Check if training data exists
            raise RuntimeError(f"Data folder not found: {DATA_ROOT}")
        # Train a new model if none exists
        train_and_save(DATA_ROOT, MODEL_PATH)


def _load_model():
    """Load the trained model from disk"""
    _ensure_model()  # Make sure model exists
    bundle = joblib.load(MODEL_PATH)  # Load saved model file
    pipeline = bundle["pipeline"]  # Extract the ML pipeline
    labels = bundle.get("labels", {0: "negative", 1: "positive"})  # Extract labels
    return pipeline, labels


# STEP 9: Main classification function
def classify(text: str) -> Dict[str, str | float]:
    """Classify text sentiment and return result with confidence"""
    pipeline, labels = _load_model()  # Load the trained model

    # Get prediction probabilities [P(negative), P(positive)]
    proba = pipeline.predict_proba([text])[0]
    pos_p = float(proba[1])  # Probability of positive sentiment

    # Determine sentiment based on probability
    sentiment = labels[1] if pos_p >= 0.5 else labels[0]  # positive if >= 50%

    # Calculate confidence (probability of predicted class)
    confidence = pos_p if sentiment == labels[1] else float(proba[0])

    return {"sentiment": sentiment, "confidence": round(confidence, 4)}


# STEP 10: Create Flask web application
# ---------------------- App bootstrap --------------------------------------
app = Flask(__name__)  # Create Flask app instance

# STEP 11: Define web routes (URLs that users can visit)
# --- GET / : serve a single HTML page -------------------------------------
@app.get("/")  # Handle requests to home page
def index():
    """Serve the main web page with sentiment analysis form"""
    # We return the entire HTML document as a string. It includes:
    # - Bootstrap (for quick styling)
    # - A simple form (textarea + button)
    # - A result area to show a pretty badge + raw JSON
    # - A small JS script that calls POST /classify and renders the result
    html = """
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8"/>
  <title>Sentiment Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <!-- Bootstrap CSS via CDN for styling -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="py-5">
  <div class="container" style="max-width: 760px;">
    <h1 class="mb-4">Sentiment Classifier</h1>

    <!-- Card container for the form + results -->
    <div class="card shadow-sm">
      <div class="card-body">
        <!-- The input form. We handle submit with JavaScript (no page reload). -->
        <form id="form" class="mb-3">
          <div class="mb-3">
            <label for="text" class="form-label">Enter a comment</label>
            <textarea id="text" class="form-control" rows="4"
              placeholder="e.g. 'awful support, totally broken and late'"></textarea>
          </div>
          <button id="btn" type="submit" class="btn btn-primary">
            <!-- Small spinner shown while waiting for the server response -->
            <span class="spinner-border spinner-border-sm me-2 d-none" id="spin"></span>
            Classify
          </button>
        </form>

        <!-- Pretty, human-friendly result area (Bootstrap alert) -->
        <div id="out" class="alert d-none" role="alert"></div>

        <!-- Optional: show the raw JSON response below -->
        <pre id="json" class="bg-light p-3 rounded d-none"></pre>
      </div>
    </div>
  </div>

  <!-- Bootstrap JS bundle (for components; not strictly required here) -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <script>
    // --- Grab references to DOM elements we'll interact with ---------------
    const form  = document.getElementById('form');
    const btn   = document.getElementById('btn');
    const ta    = document.getElementById('text');
    const out   = document.getElementById('out');
    const spin  = document.getElementById('spin');
    const jsonEl= document.getElementById('json');

    // Helper to show a Bootstrap alert with a message
    function showAlert(msg, type="info") {
      out.className = 'alert alert-' + type;
      out.textContent = msg;
      out.classList.remove('d-none');
    }

    // Pretty renderer: shows emoji + colored badge; also prints raw JSON below
    function renderSentiment(data) {
      const sentiment  = String((data && data.sentiment) || '').toLowerCase();
      const isPositive = sentiment === 'positive';
      const emoji      = isPositive ? '😊' : (sentiment === 'negative' ? '😠' : '🤔');

      // Pick alert color based on sentiment (green for positive, red for negative)
      out.className = 'alert ' + (isPositive ? 'alert-success' :
                                  (sentiment === 'negative' ? 'alert-danger' : 'alert-secondary'));
      out.hidden = false;

      // If your classify() returns a numeric confidence, show it
      const conf = (typeof data.confidence === 'number')
        ? `<div class="small text-muted">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>`
        : '';

      // Compose the pretty HTML
      out.innerHTML = `
        <div class="d-flex align-items-center gap-2">
          <span style="font-size:1.5rem">${emoji}</span>
          <div>
            <div class="fw-semibold">
              Sentiment:
              <span class="badge ${isPositive ? 'bg-success' :
                                    (sentiment === 'negative' ? 'bg-danger' : 'bg-secondary')} text-uppercase">
                ${sentiment || 'unknown'}
              </span>
            </div>
            ${conf}
          </div>
        </div>
      `;

      // Also show the raw JSON (useful for debugging)
      jsonEl.textContent = JSON.stringify(data || {}, null, 2);
      jsonEl.classList.remove('d-none');
    }

    // --- Form submit handler: call POST /classify with fetch ----------------
    form.addEventListener('submit', async (e) => {
      e.preventDefault();                 // stop the default form POST/reload
      out.classList.add('d-none');        // hide previous results
      jsonEl.classList.add('d-none');

      const text = ta.value.trim();       // get user input
      if (!text) { showAlert('Please enter some text.', 'warning'); return; }

      // Disable the button + show spinner while calling the backend
      btn.disabled = true; spin.classList.remove('d-none');
      try {
        // Send JSON {text: "..."} to our Flask endpoint
        const res = await fetch('/classify', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text })
        });

        // Parse JSON either way so we can show helpful errors
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);

        // Render a pretty result
        renderSentiment(data);
      } catch (err) {
        // Show any error message in a red alert
        showAlert('Error: ' + err.message, 'danger');
      } finally {
        // Re-enable the button and hide the spinner
        btn.disabled = false; spin.classList.add('d-none');
      }
    });
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


# STEP 12: API endpoint for sentiment classification
# --- POST /classify : server endpoint the JS calls -------------------------
@app.post("/classify")  # Handle POST requests to /classify
def classify_route():
    """API endpoint that receives text and returns sentiment analysis"""
    # Get JSON data from the request
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()  # Extract and clean the text

    # Validate input
    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        # Classify the text using our trained model
        result = classify(text)  # {"sentiment": "...", "confidence": 0.95}

        # Ensure result is properly formatted
        if not isinstance(result, dict):  # defensive fallback
            result = {"sentiment": str(result)}

        return jsonify(result)  # Return JSON response
    except Exception as e:
        # Return error if classification fails
        return jsonify({"error": "classification_failed", "detail": str(e)}), 500


# STEP 13: Optional endpoint to retrain the model
# Optional manual training endpoint (handy after adding new data)
@app.post("/train")  # Handle POST requests to /train
def train_now():
    """API endpoint to manually retrain the model with new data"""
    try:
        # Retrain the model with current data
        train_and_save(DATA_ROOT, MODEL_PATH)
        return jsonify({"message": f"Model trained and saved to {MODEL_PATH.name}"})
    except Exception as e:
        # Return error if training fails
        return jsonify({"error": "training_failed", "detail": str(e)}), 500


# STEP 14: Start the web server when script is run directly
# --- Production-friendly entrypoint (port 80 by default) -------------------
if __name__ == "__main__":
    import os
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),   # listen on all interfaces
        port=int(os.getenv("PORT", "80")),   # default to port 80
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
