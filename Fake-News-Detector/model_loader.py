# model_loader.py
import joblib
import json
from pathlib import Path


class ProductionModel:
    """Wrapper for production model with versioning"""

    def __init__(self, version='1.0.0'):
        self.version = version
        self.model_path = Path(f'production_models/{version}')

        # Load model and vectorizer
        self.model = joblib.load(self.model_path / 'model.pkl')
        self.vectorizer = joblib.load(self.model_path / 'vectorizer.pkl')

        # Load metadata
        with open(self.model_path / 'metadata.json') as f:
            self.metadata = json.load(f)

        print(f"✅ Loaded model version {version}")
        print(f"   F1 Score: {self.metadata['metrics']['f1_score']}")

    def predict(self, text):
        """Make prediction on single text"""
        # Vectorize
        features = self.vectorizer.transform([text])

        # Predict
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]

        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': float(max(proba)),
            'probabilities': {
                'real': float(proba[0]),
                'fake': float(proba[1])
            }
        }

    def predict_batch(self, texts):
        """Make predictions on multiple texts"""
        features = self.vectorizer.transform(texts)
        predictions = self.model.predict(features)
        probas = self.model.predict_proba(features)

        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            results.append({
                'text': texts[i][:50] + '...',
                'prediction': 'fake' if pred == 1 else 'real',
                'confidence': float(max(proba))
            })
        return results


# Test it
if __name__ == '__main__':
    model = ProductionModel()

    # Test prediction
    test_text = "Breaking: Shocking revelation about government conspiracy!"
    result = model.predict(test_text)
    print(f"\nTest Prediction:")
    print(f"Text: {test_text}")
    print(f"Result: {result}")