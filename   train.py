# train.py - Standalone Training Script for Sentiment Analysis Model
# This file trains a machine learning model to classify text sentiment

# STEP 1: Import required libraries
from __future__ import annotations  # Enable modern type hints
from pathlib import Path  # For file path operations
from typing import List, Tuple  # For type hints
import re  # Regular expressions for text processing
import joblib  # For saving/loading ML models

# Machine learning libraries
from sklearn.pipeline import Pipeline  # ML pipeline
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # Text to numbers conversion
from sklearn.linear_model import LogisticRegression  # Classification algorithm
from sklearn.model_selection import train_test_split  # Split data for training/testing
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation

# STEP 2: Define file paths and constants
# Paths relative to this file (so it works no matter your working directory)
HERE = Path(__file__).parent  # Current directory where this file is located
DATA_ROOT = HERE / "data"  # Folder containing training data
MODEL_PATH = HERE / "sentiment_model.joblib"  # Where to save the trained model

# Define sentiment labels and expected file names
LABELS = {0: "negative", 1: "positive"}  # Map numbers to sentiment labels
REVIEW_FILES = ("positive.review", "negative.review")  # Expected data file names

# STEP 3: Function to read and process review files
def _read_reviews(path: Path, label: int) -> Tuple[List[str], List[int]]:
    """Read reviews from a file and assign sentiment labels"""
    texts, labels = [], []  # Initialize empty lists for texts and labels

    # Check if the file exists
    if not path.exists():
        return texts, labels  # Return empty lists if no file found

    # Read the entire file content
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    # Split by blank lines (assumes reviews are separated by blank lines)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]

    # If few chunks found, fall back to per-line splitting
    if len(chunks) < 10:
        chunks = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    # Process each text chunk
    for c in chunks:
        texts.append(c)  # Add review text
        labels.append(label)  # Add corresponding sentiment label

    return texts, labels


# STEP 4: Function to load complete dataset from directory structure
def load_dataset(root: Path) -> Tuple[List[str], List[int]]:
    """Load all review files from the data directory structure"""
    X, y = [], []  # X will store texts, y will store labels

    # Iterate through all subdirectories (e.g., books, dvd, electronics)
    for category in root.glob("*"):
        if not category.is_dir():  # Skip if not a directory
            continue

        # Look for positive.review and negative.review files in each category
        for fname in REVIEW_FILES:
            # Determine sentiment label based on filename
            label = 1 if fname.startswith("positive") else 0  # 1=positive, 0=negative

            # Read all reviews from this file
            tx, lb = _read_reviews(category / fname, label)

            # Add all texts and labels to our main dataset
            X.extend(tx)  # Extend with review texts
            y.extend(lb)  # Extend with corresponding labels

    # Validate that we found some data
    if not X:
        raise RuntimeError(
            f"No reviews found under {root}. Expected {REVIEW_FILES} inside subfolders."
        )

    return X, y


# STEP 5: Create the machine learning pipeline
def build_pipeline() -> Pipeline:
    """Build a machine learning pipeline for sentiment classification"""
    return Pipeline(
        [
            # Step 1: Text Feature Extraction using TF-IDF
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,  # Convert all text to lowercase
                    ngram_range=(1, 2),  # Use both single words and word pairs
                    max_features=50000,  # Keep only the 50,000 most important features
                    min_df=2,  # Word must appear in at least 2 documents
                    max_df=0.9,  # Ignore words that appear in >90% of documents
                ),
            ),
            # Step 2: Classification using Logistic Regression
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,  # Maximum number of training iterations
                    class_weight="balanced",  # Handle imbalanced classes automatically
                    solver="liblinear",  # Optimization algorithm (good for text data)
                ),
            ),
        ]
    )


# STEP 6: Main training function
def train_and_save(data_root: Path = DATA_ROOT, model_path: Path = MODEL_PATH):
    """Complete training workflow: load data, train model, evaluate, and save"""

    # Load all training data
    print(f"Loading data from: {data_root}")
    X, y = load_dataset(data_root)  # X = review texts, y = sentiment labels
    print(f"Loaded {len(X)} reviews.")

    # Split data into training and testing sets
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,  # Input data and labels
        test_size=0.2,  # Use 20% for testing, 80% for training
        random_state=42,  # Fixed seed for reproducible results
        stratify=y,  # Maintain same positive/negative ratio in both sets
    )

    # Create and train the machine learning model
    pipe = build_pipeline()  # Create the ML pipeline
    print("Training model...")
    pipe.fit(X_tr, y_tr)  # Train on training data

    # Evaluate model performance on test data
    print("Evaluating...")
    y_pred = pipe.predict(X_te)  # Make predictions on test set
    acc = accuracy_score(y_te, y_pred)  # Calculate accuracy
    print(f"Accuracy: {acc:.4f}")

    # Show detailed performance metrics
    print(classification_report(y_te, y_pred, target_names=[LABELS[0], LABELS[1]]))

    # Save the trained model to disk
    print(f"Saving model to: {model_path}")
    joblib.dump({"pipeline": pipe, "labels": LABELS}, model_path)


# STEP 7: Run training when script is executed directly
if __name__ == "__main__":
    # Execute the complete training process
    train_and_save()
    print("\nTraining completed! The model is ready to use.")
    print("You can now run app.py to start the web interface.")
