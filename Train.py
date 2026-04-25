#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -----------------------------
data = [
    ("hello how are you", "normal"),
    ("hi there", "normal"),
    ("limited offer buy now", "spam"),
    ("free money now", "spam"),
    ("what is your name", "question"),
    ("how does this work", "question"),
    ("click this link", "spam"),
    ("good morning", "normal"),
    ("dont use it","spam"),
    ("Test it","spam"),
    ("Hello","normal"),
    ("Send","spam"),
    ("Let's go","normal"),
    ("Try click","spam")

]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -----------------------------

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        max_features=5000,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# -----------------------------
print("🔹 Training model...")
pipeline.fit(X_train, y_train)
print("Model trained successfully")

# -----------------------------
y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
test_text = "Try send it"
prediction = pipeline.predict([test_text])[0]
print("\nTest Input:", test_text)
print("Prediction:", prediction)

# -----------------------------
MODEL_PATH = "nlp_model_pipeline.joblib"
joblib.dump(pipeline, MODEL_PATH)
print(f"\n Model saved as {MODEL_PATH}")





import joblib
model = joblib.load("nlp_model_pipeline.joblib")
prediction = model.predict(["hi"])[0]
print(prediction)