"""
FastAPI Backend for CERT-EU Ticket Classification System
"""

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the code directory to Python path to import our ML modules
sys.path.append("/app/code")

from database import (
    get_db, create_tables, Ticket, Prediction, Feedback, User, 
    AmbiguousBatch, AmbiguousTicket, ModelMetric, CategoryKeyword,
    SystemConfig, TicketCreate, PredictionCreate, FeedbackCreate,
    TicketResponse, PredictionResponse, DashboardStats, QueuePerformance
)
# from time_series_predictor import TimeSeriesPredictor

# Import ML components (with error handling for missing model)
try:
    from data_processor import DataProcessor
    from model_trainer import HybridTransformerModel
    ML_AVAILABLE = True
    print("ML modules loaded successfully")
except ImportError as e:
    print(f"Warning: ML modules not available: {e}")
    print("Backend will run without ML functionality")
    DataProcessor = None
    HybridTransformerModel = None
    ML_AVAILABLE = False

import joblib

app = FastAPI(
    title="CERT-EU Ticket Classification API",
    description="AI-powered ticket classification system for CERT-EU",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for ML model
model = None
processor = None
class_names = None

# Time series predictor (disabled for now)
# time_series_predictor = TimeSeriesPredictor()

# Initialize ML components
def load_model():
    """Load the trained model and processor"""
    global model, processor, class_names
    
    if not ML_AVAILABLE:
        print("ML modules not available - running in demo mode")
        # Set up demo data
        class_names = ['CTI', 'DFIR::incidents', 'DFIR::phishing', 'OFFSEC::CVD', 'OFFSEC::Pentesting', 'SMS', 'Trash']
        model = None
        processor = None
        return
    
    try:
        # Load from the models directory
        model_path = Path("/app/code/models")
        processor_path = model_path / "data_processor.pkl"
        class_names_path = model_path / "class_names.txt"
        
        if not processor_path.exists() or not class_names_path.exists():
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        # Load processor
        processor = DataProcessor()
        processor.load_processor(str(processor_path))
        
        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Load model
        model = HybridTransformerModel(
            num_labels=len(class_names),
            num_numerical_features=17,  # Standard feature count from your code
            use_gpu=False  # Set to True if GPU available
        )
        
        # Try to find model directory
        model_dir = model_path / "hybrid_roberta_model"
        if model_dir.exists():
            model.load_model(str(model_dir))
            print(f"Model loaded successfully from {model_dir}")
        else:
            raise FileNotFoundError("Model directory not found")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # Set up demo mode
        class_names = ['CTI', 'DFIR::incidents', 'DFIR::phishing', 'OFFSEC::CVD', 'OFFSEC::Pentesting', 'SMS', 'Trash']
        model = None
        processor = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on startup"""
    create_tables()
    try:
        load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Model not loaded: {e}")
        print("Some endpoints may not work without the trained model")

# Authentication (simplified for now)
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - replace with proper JWT in production"""
    # For now, return a default user
    return {"user_id": str(uuid.uuid4()), "email": "admin@cert-eu.eu", "role": "admin"}

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "class_names": class_names if class_names else []
    }

# Dashboard endpoints
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    
    # Get total tickets
    total_tickets = db.query(Ticket).count()
    
    # Get total predictions
    total_predictions = db.query(Prediction).count()
    
    # Get routing decisions
    auto_routed = db.query(Prediction).filter(Prediction.routing_decision == "auto_route").count()
    human_verify = db.query(Prediction).filter(Prediction.routing_decision == "human_verify").count()
    manual_triage = db.query(Prediction).filter(Prediction.routing_decision == "manual_triage").count()
    
    # Get average confidence
    avg_confidence_result = db.query(Prediction).with_entities(
        func.avg(Prediction.confidence_score)
    ).scalar()
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
    
    # Get feedback stats
    total_feedback = db.query(Feedback).count()
    corrections_needed = db.query(Feedback).filter(Feedback.is_correct == False).count()
    
    return DashboardStats(
        total_tickets=total_tickets,
        total_predictions=total_predictions,
        auto_routed=auto_routed,
        human_verify=human_verify,
        manual_triage=manual_triage,
        avg_confidence=avg_confidence,
        total_feedback=total_feedback,
        corrections_needed=corrections_needed
    )

@app.get("/api/dashboard/queue-performance", response_model=List[QueuePerformance])
async def get_queue_performance(db: Session = Depends(get_db)):
    """Get performance metrics by queue"""
    
    # This would typically be done with a SQL query, but using Python for now
    predictions = db.query(Prediction).all()
    feedbacks = db.query(Feedback).all()
    
    # Group by queue
    queue_stats = {}
    for pred in predictions:
        queue = pred.predicted_queue
        if queue not in queue_stats:
            queue_stats[queue] = {
                'total_predictions': 0,
                'confidence_scores': [],
                'auto_routed_count': 0,
                'human_verify_count': 0,
                'manual_triage_count': 0,
                'feedback_count': 0,
                'corrections_count': 0
            }
        
        queue_stats[queue]['total_predictions'] += 1
        queue_stats[queue]['confidence_scores'].append(float(pred.confidence_score))
        
        if pred.routing_decision == 'auto_route':
            queue_stats[queue]['auto_routed_count'] += 1
        elif pred.routing_decision == 'human_verify':
            queue_stats[queue]['human_verify_count'] += 1
        else:
            queue_stats[queue]['manual_triage_count'] += 1
    
    # Add feedback data
    for feedback in feedbacks:
        pred = db.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
        if pred:
            queue = pred.predicted_queue
            if queue in queue_stats:
                queue_stats[queue]['feedback_count'] += 1
                if not feedback.is_correct:
                    queue_stats[queue]['corrections_count'] += 1
    
    # Convert to response format
    results = []
    for queue, stats in queue_stats.items():
        avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
        error_rate = (stats['corrections_count'] / stats['feedback_count'] * 100) if stats['feedback_count'] > 0 else 0.0
        
        results.append(QueuePerformance(
            predicted_queue=queue,
            total_predictions=stats['total_predictions'],
            avg_confidence=avg_confidence,
            auto_routed_count=stats['auto_routed_count'],
            human_verify_count=stats['human_verify_count'],
            manual_triage_count=stats['manual_triage_count'],
            total_feedback=stats['feedback_count'],
            corrections_count=stats['corrections_count'],
            error_rate_percent=error_rate
        ))
    
    return sorted(results, key=lambda x: x.total_predictions, reverse=True)

# Ticket prediction endpoint
@app.post("/api/predict")
async def predict_ticket(
    ticket_data: TicketCreate,
    db: Session = Depends(get_db)
):
    """Predict queue for a single ticket"""
    
    if not model or not processor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Create ticket in database
        db_ticket = Ticket(
            ticket_id=ticket_data.ticket_id,
            title=ticket_data.title,
            content=ticket_data.content,
            created_date=ticket_data.created_date,
            email_address=ticket_data.email_address,
            raw_data=ticket_data.raw_data
        )
        db.add(db_ticket)
        db.commit()
        db.refresh(db_ticket)
        
        # Prepare features for prediction
        df = pd.DataFrame([{
            'title': ticket_data.title,
            'content': ticket_data.content,
            'created_date': ticket_data.created_date or datetime.now(),
            'email_address': ticket_data.email_address or 'unknown@example.com'
        }])
        
        # Extract features
        df_features = processor.extract_features(df)
        text_features, numerical_features, _ = processor.prepare_features(df_features, is_training=False)
        
        # Make prediction
        start_time = datetime.now()
        predictions = model.predict(text_features, numerical_features)
        probabilities = model.predict_proba(text_features, numerical_features)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        predicted_queue = class_names[predictions[0]]
        confidence_score = float(np.max(probabilities[0]))
        all_probabilities = {class_names[i]: float(prob) for i, prob in enumerate(probabilities[0])}
        
        # Determine routing decision based on confidence
        if confidence_score >= 0.85:  # High confidence
            routing_decision = "auto_route"
        elif confidence_score >= 0.65:  # Medium confidence
            routing_decision = "human_verify"
        else:  # Low confidence
            routing_decision = "manual_triage"
        
        # Store prediction
        db_prediction = Prediction(
            ticket_id=db_ticket.id,
            predicted_queue=predicted_queue,
            confidence_score=confidence_score,
            all_probabilities=all_probabilities,
            model_version="hybrid_roberta_v1",
            processing_time_ms=int(processing_time),
            features_used={
                'text_length': len(ticket_data.title + " " + ticket_data.content),
                'has_urls': 'http' in ticket_data.content.lower(),
                'has_attachments': any(ext in ticket_data.content.lower() for ext in ['.pdf', '.doc', '.xlsx'])
            },
            routing_decision=routing_decision
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return {
            "ticket_id": db_ticket.ticket_id,
            "predicted_queue": predicted_queue,
            "confidence_score": confidence_score,
            "all_probabilities": all_probabilities,
            "routing_decision": routing_decision,
            "processing_time_ms": int(processing_time),
            "prediction_id": str(db_prediction.id)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/api/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Predict queue for multiple tickets from JSONL file"""
    
    if not model or not processor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read and parse JSONL file
        content = await file.read()
        lines = content.decode('utf-8').strip().split('\n')
        tickets_data = [json.loads(line) for line in lines if line.strip()]
        
        results = []
        for ticket_data in tickets_data:
            # Create ticket
            ticket_create = TicketCreate(
                ticket_id=ticket_data.get('ticket_id', f"batch_{uuid.uuid4()}"),
                title=ticket_data.get('title', ''),
                content=ticket_data.get('content', ''),
                created_date=datetime.fromisoformat(ticket_data.get('created_date', datetime.now().isoformat())) if ticket_data.get('created_date') else None,
                email_address=ticket_data.get('email_address'),
                raw_data=ticket_data
            )
            
            # Make prediction (reuse the single prediction logic)
            prediction_result = await predict_ticket(ticket_create, db)
            results.append(prediction_result)
        
        return {
            "total_processed": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Low confidence tickets endpoint
@app.get("/api/tickets/low-confidence")
async def get_low_confidence_tickets(
    limit: int = 50,
    threshold: float = 0.65,
    db: Session = Depends(get_db)
):
    """Get tickets with low confidence scores for human review"""
    
    predictions = db.query(Prediction).filter(
        Prediction.confidence_score < threshold
    ).limit(limit).all()
    
    results = []
    for pred in predictions:
        ticket = db.query(Ticket).filter(Ticket.id == pred.ticket_id).first()
        if ticket:
            results.append({
                "prediction_id": str(pred.id),
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "content": ticket.content[:500] + "..." if len(ticket.content) > 500 else ticket.content,
                "predicted_queue": pred.predicted_queue,
                "confidence_score": float(pred.confidence_score),
                "all_probabilities": pred.all_probabilities,
                "routing_decision": pred.routing_decision,
                "created_at": ticket.created_at.isoformat()
            })
    
    return {
        "total_count": len(results),
        "threshold": threshold,
        "tickets": results
    }

# Feedback endpoint
@app.post("/api/feedback")
async def submit_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """Submit human feedback for a prediction"""
    
    try:
        db_feedback = Feedback(
            prediction_id=uuid.UUID(feedback_data.prediction_id),
            reviewer_id=uuid.UUID(feedback_data.reviewer_id),
            corrected_queue=feedback_data.corrected_queue,
            feedback_notes=feedback_data.feedback_notes,
            keywords_highlighted=feedback_data.keywords_highlighted,
            difficulty_score=feedback_data.difficulty_score,
            is_correct=feedback_data.is_correct
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        return {
            "feedback_id": str(db_feedback.id),
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

# Keywords analysis endpoint
@app.get("/api/keywords/{queue_name}")
async def get_queue_keywords(
    queue_name: str,
    db: Session = Depends(get_db)
):
    """Get keywords analysis for a specific queue"""
    
    keywords = db.query(CategoryKeyword).filter(
        CategoryKeyword.queue_name == queue_name
    ).order_by(CategoryKeyword.frequency.desc()).all()
    
    return {
        "queue_name": queue_name,
        "keywords": [
            {
                "keyword": kw.keyword,
                "frequency": kw.frequency,
                "importance_score": float(kw.importance_score) if kw.importance_score else 0.0
            }
            for kw in keywords
        ]
    }

# Time Series endpoints disabled for now due to dependency conflicts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
