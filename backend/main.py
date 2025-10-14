"""
FastAPI Backend for CERT-EU Ticket Classification System
"""

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, case
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Enhanced Reporting Endpoints
@app.get("/api/reports/confidence-analysis")
async def get_confidence_analysis(
    days_back: int = 30,
    db: Session = Depends(get_db)
):
    """Get confidence analysis report"""
    
    try:
        # Calculate confidence distribution
        confidence_ranges = [
            (0.9, 1.0, "High Confidence"),
            (0.7, 0.9, "Medium Confidence"), 
            (0.5, 0.7, "Low Confidence"),
            (0.0, 0.5, "Very Low Confidence")
        ]
        
        confidence_dist = []
        for min_conf, max_conf, label in confidence_ranges:
            count = db.query(Prediction).filter(
                and_(
                    Prediction.confidence_score >= min_conf,
                    Prediction.confidence_score < max_conf
                )
            ).count()
            confidence_dist.append({
                "range": f"{min_conf}-{max_conf}",
                "label": label,
                "count": count,
                "percentage": round((count / db.query(Prediction).count() * 100), 2) if db.query(Prediction).count() > 0 else 0
            })
        
        # Routing decision breakdown
        routing_stats = db.query(
            Prediction.routing_decision,
            func.count(Prediction.id).label('count'),
            func.avg(Prediction.confidence_score).label('avg_confidence')
        ).group_by(Prediction.routing_decision).all()
        
        routing_breakdown = [
            {
                "decision": stat.routing_decision,
                "count": stat.count,
                "avg_confidence": float(stat.avg_confidence) if stat.avg_confidence else 0.0
            }
            for stat in routing_stats
        ]
        
        # Queue-wise confidence analysis
        queue_confidence = db.query(
            Prediction.predicted_queue,
            func.avg(Prediction.confidence_score).label('avg_confidence'),
            func.min(Prediction.confidence_score).label('min_confidence'),
            func.max(Prediction.confidence_score).label('max_confidence'),
            func.count(Prediction.id).label('total_predictions')
        ).group_by(Prediction.predicted_queue).all()
        
        queue_analysis = [
            {
                "queue": q.predicted_queue,
                "avg_confidence": float(q.avg_confidence) if q.avg_confidence else 0.0,
                "min_confidence": float(q.min_confidence) if q.min_confidence else 0.0,
                "max_confidence": float(q.max_confidence) if q.max_confidence else 0.0,
                "total_predictions": q.total_predictions
            }
            for q in queue_confidence
        ]
        
        return {
            "confidence_distribution": confidence_dist,
            "routing_breakdown": routing_breakdown,
            "queue_analysis": queue_analysis,
            "total_predictions": db.query(Prediction).count(),
            "analysis_period": f"Last {days_back} days"
        }
        
    except Exception as e:
        logger.error(f"Error generating confidence analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate confidence analysis: {str(e)}"
        )

@app.get("/api/reports/feedback-analysis")
async def get_feedback_analysis(
    days_back: int = 30,
    db: Session = Depends(get_db)
):
    """Get feedback analysis report"""
    
    try:
        # Overall feedback stats
        total_feedback = db.query(Feedback).count()
        correct_feedback = db.query(Feedback).filter(Feedback.is_correct == True).count()
        incorrect_feedback = db.query(Feedback).filter(Feedback.is_correct == False).count()
        
        # Feedback by queue
        feedback_by_queue = db.query(
            Prediction.predicted_queue,
            func.count(Feedback.id).label('total_feedback'),
            func.sum(case((Feedback.is_correct == True, 1), else_=0)).label('correct_count'),
            func.avg(Feedback.difficulty_score).label('avg_difficulty')
        ).join(Prediction, Feedback.prediction_id == Prediction.id).group_by(
            Prediction.predicted_queue
        ).all()
        
        queue_feedback = [
            {
                "queue": f.predicted_queue,
                "total_feedback": f.total_feedback,
                "correct_count": f.correct_count,
                "accuracy": round((f.correct_count / f.total_feedback * 100), 2) if f.total_feedback > 0 else 0,
                "avg_difficulty": float(f.avg_difficulty) if f.avg_difficulty else 0.0
            }
            for f in feedback_by_queue
        ]
        
        # Difficulty distribution
        difficulty_ranges = [
            (1, 2, "Easy"),
            (3, 4, "Medium"),
            (5, 6, "Hard"),
            (7, 8, "Very Hard"),
            (9, 10, "Extremely Hard")
        ]
        
        difficulty_dist = []
        for min_diff, max_diff, label in difficulty_ranges:
            count = db.query(Feedback).filter(
                and_(
                    Feedback.difficulty_score >= min_diff,
                    Feedback.difficulty_score <= max_diff
                )
            ).count()
            difficulty_dist.append({
                "range": f"{min_diff}-{max_diff}",
                "label": label,
                "count": count
            })
        
        return {
            "overall_stats": {
                "total_feedback": total_feedback,
                "correct_feedback": correct_feedback,
                "incorrect_feedback": incorrect_feedback,
                "accuracy": round((correct_feedback / total_feedback * 100), 2) if total_feedback > 0 else 0
            },
            "queue_feedback": queue_feedback,
            "difficulty_distribution": difficulty_dist,
            "analysis_period": f"Last {days_back} days"
        }
        
    except Exception as e:
        logger.error(f"Error generating feedback analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate feedback analysis: {str(e)}"
        )

@app.get("/api/reports/tickets-detail")
async def get_tickets_detail_report(
    limit: int = 100,
    offset: int = 0,
    queue_filter: Optional[str] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get detailed tickets report with filtering options"""
    
    try:
        query = db.query(Prediction).join(Ticket, Prediction.ticket_id == Ticket.ticket_id)
        
        # Apply filters
        if queue_filter:
            query = query.filter(Prediction.predicted_queue == queue_filter)
        
        if confidence_min is not None:
            query = query.filter(Prediction.confidence_score >= confidence_min)
            
        if confidence_max is not None:
            query = query.filter(Prediction.confidence_score <= confidence_max)
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination and get results
        tickets = query.offset(offset).limit(limit).all()
        
        # Format results
        tickets_data = []
        for ticket in tickets:
            tickets_data.append({
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "predicted_queue": ticket.predicted_queue,
                "confidence_score": float(ticket.confidence_score),
                "routing_decision": ticket.routing_decision,
                "created_at": ticket.created_at.isoformat(),
                "processing_time_ms": ticket.processing_time_ms,
                "has_feedback": db.query(Feedback).filter(Feedback.prediction_id == ticket.id).count() > 0
            })
        
        return {
            "tickets": tickets_data,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filters_applied": {
                "queue_filter": queue_filter,
                "confidence_min": confidence_min,
                "confidence_max": confidence_max
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating tickets detail report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate tickets detail report: {str(e)}"
        )

# Email and Report Filtering Endpoints
@app.post("/api/reports/email")
async def send_email_report(
    email_data: dict,
    db: Session = Depends(get_db)
):
    """Send simple email report with today's date"""
    
    try:
        recipient_email = email_data.get('recipient_email')
        subject = email_data.get('subject', 'CERT-EU Ticket Classification Report')
        message = email_data.get('message', '')
        
        if not recipient_email:
            raise HTTPException(status_code=400, detail="Recipient email is required")
        
        # Get all current predictions (since we don't have created_at field)
        predictions = db.query(Prediction).all()
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No tickets found in the database")
        
        # Generate report data
        total_tickets = len(predictions)
        auto_routed = len([p for p in predictions if p.routing_decision == 'auto_route'])
        human_verify = len([p for p in predictions if p.routing_decision == 'human_verify'])
        manual_triage = len([p for p in predictions if p.routing_decision == 'manual_triage'])
        avg_confidence = sum(p.confidence_score for p in predictions) / total_tickets if total_tickets > 0 else 0
        
        # Queue performance
        queue_stats = {}
        for prediction in predictions:
            queue = prediction.predicted_queue
            if queue not in queue_stats:
                queue_stats[queue] = {
                    'total_predictions': 0,
                    'auto_routed_count': 0,
                    'human_verify_count': 0,
                    'manual_triage_count': 0,
                    'confidence_scores': []
                }
            
            queue_stats[queue]['total_predictions'] += 1
            queue_stats[queue]['confidence_scores'].append(prediction.confidence_score)
            
            if prediction.routing_decision == 'auto_route':
                queue_stats[queue]['auto_routed_count'] += 1
            elif prediction.routing_decision == 'human_verify':
                queue_stats[queue]['human_verify_count'] += 1
            else:
                queue_stats[queue]['manual_triage_count'] += 1
        
        # Calculate average confidence for each queue
        for queue in queue_stats:
            scores = queue_stats[queue]['confidence_scores']
            queue_stats[queue]['avg_confidence'] = sum(scores) / len(scores) if scores else 0
            del queue_stats[queue]['confidence_scores']  # Remove raw scores
        
        # Get today's date
        today = datetime.now().strftime('%d %B %Y')  # e.g., "13 October 2025"
        
        # Create simple email content
        email_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; max-width: 800px; margin: 0 auto;">
            <div style="background-color: #003399; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0; font-size: 24px;">CERT-EU Ticket Classification Report</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px;">European Union Cybersecurity Emergency Response Team</p>
            </div>
            
            <div style="padding: 30px; background-color: #f9f9f9; border-radius: 0 0 8px 8px;">
                <h2 style="color: #003399; margin-top: 0;">Hi! Here is the report of {today}</h2>
                
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #003399; margin-top: 0;">📊 Key Metrics</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li style="margin: 10px 0; padding: 10px; background-color: #f0f8ff; border-left: 4px solid #003399;"><strong>Total Tickets:</strong> {total_tickets}</li>
                        <li style="margin: 10px 0; padding: 10px; background-color: #f0fff0; border-left: 4px solid #10B981;"><strong>Auto-Routed:</strong> {auto_routed} ({((auto_routed/total_tickets)*100):.1f}%)</li>
                        <li style="margin: 10px 0; padding: 10px; background-color: #fffbf0; border-left: 4px solid #F59E0B;"><strong>Human Review:</strong> {human_verify} ({((human_verify/total_tickets)*100):.1f}%)</li>
                        <li style="margin: 10px 0; padding: 10px; background-color: #fff0f0; border-left: 4px solid #EF4444;"><strong>Manual Triage:</strong> {manual_triage} ({((manual_triage/total_tickets)*100):.1f}%)</li>
                        <li style="margin: 10px 0; padding: 10px; background-color: #f0f0ff; border-left: 4px solid #8B5CF6;"><strong>Average Confidence:</strong> {(avg_confidence*100):.1f}%</li>
                    </ul>
                </div>
                
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #003399; margin-top: 0;">📈 Queue Performance</h3>
                    <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
                        <tr style="background-color: #003399; color: white;">
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Queue</th>
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Total</th>
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Avg Confidence</th>
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Auto-Routed</th>
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Human Review</th>
                            <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Manual Triage</th>
                        </tr>
        """
        
        # Add queue data to table
        for queue_name, stats in queue_stats.items():
            email_html += f"""
                        <tr style="background-color: #f9f9f9;">
                            <td style="border: 1px solid #ddd; padding: 12px;"><strong>{queue_name}</strong></td>
                            <td style="border: 1px solid #ddd; padding: 12px;">{stats['total_predictions']}</td>
                            <td style="border: 1px solid #ddd; padding: 12px;">{(stats['avg_confidence']*100):.1f}%</td>
                            <td style="border: 1px solid #ddd; padding: 12px; color: #10B981;">{stats['auto_routed_count']}</td>
                            <td style="border: 1px solid #ddd; padding: 12px; color: #F59E0B;">{stats['human_verify_count']}</td>
                            <td style="border: 1px solid #ddd; padding: 12px; color: #EF4444;">{stats['manual_triage_count']}</td>
                        </tr>
            """
        
        email_html += f"""
                    </table>
                </div>
                
                {f'<div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="color: #003399; margin-top: 0;">💬 Additional Message</h3><p>{message}</p></div>' if message else ''}
                
                <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <p style="color: #666; font-size: 14px; margin: 0;">
                        <strong>Generated:</strong> {datetime.now().strftime('%d %B %Y at %H:%M:%S')}<br>
                        This report was automatically generated by the CERT-EU Ticket Classification System.<br>
                        For questions or support, please contact the CERT-EU team.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Quick SMTP setup for proof of concept
        try:
            # SMTP Configuration (from environment variables)
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_username = os.getenv("SMTP_USERNAME", "mjovanovjr@gmail.com")
            smtp_password = os.getenv("SMTP_PASSWORD", "ttizkzzoregfazqy")
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = smtp_username
            msg['To'] = recipient_email
            
            # Add HTML content
            html_part = MIMEText(email_html, 'html')
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {recipient_email}: {total_tickets} tickets on {today}")
            
            return {
                "success": True,
                "message": f"Email sent successfully to {recipient_email} for {total_tickets} tickets on {today}",
                "period_info": f"Report of {today}",
                "email_preview": email_html,
                "total_tickets": total_tickets,
                "date": today,
                "delivery_status": "Email sent via Gmail SMTP"
            }
            
        except Exception as smtp_error:
            logger.error(f"SMTP error: {smtp_error}")
            # Fallback: return success without sending (for demo purposes)
            return {
                "success": True,
                "message": f"Email content generated for {total_tickets} tickets on {today} (SMTP not configured)",
                "period_info": f"Report of {today}",
                "email_preview": email_html,
                "total_tickets": total_tickets,
                "date": today,
                "delivery_status": "Email content generated - SMTP configuration needed",
                "smtp_error": str(smtp_error)
            }
        
    except Exception as e:
        logger.error(f"Error generating email report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate email report: {str(e)}"
        )

@app.get("/api/reports/filtered")
async def get_filtered_report(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    ticket_limit: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get filtered report data based on date range and ticket limit"""
    
    try:
        query = db.query(Prediction)
        
        # Apply date filters
        if date_from:
            query = query.filter(Prediction.created_at >= datetime.fromisoformat(date_from))
        if date_to:
            query = query.filter(Prediction.created_at <= datetime.fromisoformat(date_to))
        
        # Apply ticket limit
        if ticket_limit:
            query = query.limit(ticket_limit)
        
        filtered_predictions = query.all()
        
        if not filtered_predictions:
            return {
                "message": "No tickets found for the specified criteria",
                "total_tickets": 0,
                "queue_performance": []
            }
        
        # Calculate stats
        total_tickets = len(filtered_predictions)
        auto_routed = len([p for p in filtered_predictions if p.routing_decision == 'auto_route'])
        human_verify = len([p for p in filtered_predictions if p.routing_decision == 'human_verify'])
        manual_triage = len([p for p in filtered_predictions if p.routing_decision == 'manual_triage'])
        avg_confidence = sum(p.confidence_score for p in filtered_predictions) / total_tickets if total_tickets > 0 else 0
        
        return {
            "total_tickets": total_tickets,
            "auto_routed": auto_routed,
            "human_verify": human_verify,
            "manual_triage": manual_triage,
            "avg_confidence": avg_confidence,
            "date_from": date_from,
            "date_to": date_to,
            "ticket_limit": ticket_limit,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting filtered report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get filtered report: {str(e)}"
        )

# Time Series endpoints disabled for now due to dependency conflicts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
