# train_fresh_model.py

"""
Train a fresh model with your current Python/NumPy environment
This will create compatible .pkl files
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime

print("🚀 Training fresh model with current environment...")

# Load your data
print("Loading data...")
df = pd.read_csv('data/liar_binary_combined.csv')

X = df['text'].values
y = df['label'].values

print(f"Loaded {len(df)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create vectorizer
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train ensemble model
print("Training ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42))
    ],
    voting='soft'
)

ensemble.fit(X_train_vec, y_train)

# Test it
y_pred = ensemble.predict(X_test_vec)
from sklearn.metrics import f1_score, accuracy_score

f1 = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained!")
print(f"   F1 Score: {f1:.4f}")
print(f"   Accuracy: {acc:.4f}")

# Save model
print("\nSaving model...")
os.makedirs('production_models/1.0.0', exist_ok=True)

joblib.dump(ensemble, 'production_models/1.0.0/model.pkl')
joblib.dump(vectorizer, 'production_models/1.0.0/vectorizer.pkl')

metadata = {
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'metrics': {
        'f1_score': float(f1),
        'accuracy': float(acc)
    },
    'model_type': 'VotingClassifier',
    'python_version': f"{np.__version__}",
    'environment': 'fresh_training'
}

with open('production_models/1.0.0/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Get file sizes
model_size = os.path.getsize('production_models/1.0.0/model.pkl') / 1024 / 1024
vec_size = os.path.getsize('production_models/1.0.0/vectorizer.pkl') / 1024 / 1024

print(f"\n✅ Fresh model saved!")
print(f"   Location: production_models/1.0.0/")
print(f"   Model size: {model_size:.2f} MB")
print(f"   Vectorizer size: {vec_size:.2f} MB")
