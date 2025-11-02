# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from model_loader import ProductionModel
from datetime import datetime
import time

# Initialize FastAPI
app = FastAPI(
    title="Fake News Detection API",
    description="Detect fake news using ML",
    version="1.0.0"
)

# Load model on startup
model = None


@app.on_event("startup")
async def startup_event():
    global model
    model = ProductionModel(version='1.0.0')
    print("✅ Model loaded and ready!")


# Request/Response models
class PredictionRequest(BaseModel):
    text: str

    @validator('text')
    def text_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters')
        if len(v) > 5000:
            raise ValueError('Text too long (max 5000 characters)')
        return v


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    processing_time_ms: float
    timestamp: str
    model_version: str


# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Fake News Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Check if API is healthy"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict if news text is fake or real

    - **text**: The news article or statement to classify

    Returns prediction with confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Make prediction
        result = model.predict(request.text)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Return response
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            model_version='1.0.0'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
class BatchRequest(BaseModel):
    texts: list[str]

    @validator('texts')
    def validate_batch(cls, v):
        if len(v) > 50:
            raise ValueError('Batch size cannot exceed 50')
        return v


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """Predict multiple texts at once (max 50)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = model.predict_batch(request.texts)
        return {
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload