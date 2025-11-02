
from fake_news_ml_pipeline import FakeNewsDetectionPipeline
import joblib

# Re-run just the model training part (much faster than full pipeline)
pipeline = FakeNewsDetectionPipeline(random_state=42)

# Load your data
X, y = pipeline.load_and_preprocess_data(
    data_path='data/liar_binary_combined.csv',
    text_column='text',
    label_column='label'
)

# Build ensemble (this is fast, just a few seconds)
pipeline.build_ensemble_models(X, y)

# Now you have the objects:
ensemble_model = pipeline.ensemble_model
vectorizer = pipeline.vectorizer

# Export manually:
import os
import json
from datetime import datetime

os.makedirs('production_models/1.0.0', exist_ok=True)

print("Saving model...")
joblib.dump(ensemble_model, 'production_models/1.0.0/model.pkl')

print("Saving vectorizer...")
joblib.dump(vectorizer, 'production_models/1.0.0/vectorizer.pkl')

metadata = {
    'version': '1.0.0',
    'created_at': datetime.now().isoformat(),
    'metrics': {
        'f1_score': 0.596,
        'accuracy': 0.617
    }
}

with open('production_models/1.0.0/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model exported!")