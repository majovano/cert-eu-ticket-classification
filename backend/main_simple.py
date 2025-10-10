"""
Simple FastAPI Backend for CERT-EU Ticket Classification System (Demo Mode)
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from database import (
    get_db, create_tables, Ticket, Prediction, Feedback, User, 
    TicketCreate, PredictionCreate, FeedbackCreate,
    TicketResponse, PredictionResponse, DashboardStats, QueuePerformance
)

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo class names
class_names = ['CTI', 'DFIR::incidents', 'DFIR::phishing', 'OFFSEC::CVD', 'OFFSEC::Pentesting', 'SMS', 'Trash']

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    create_tables()
    print("✅ Database initialized")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": False,
        "mode": "demo",
        "class_names": class_names
    }

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    
    total_tickets = db.query(Ticket).count()
    total_predictions = db.query(Prediction).count()
    auto_routed = db.query(Prediction).filter(Prediction.routing_decision == "auto_route").count()
    human_verify = db.query(Prediction).filter(Prediction.routing_decision == "human_verify").count()
    manual_triage = db.query(Prediction).filter(Prediction.routing_decision == "manual_triage").count()
    
    avg_confidence_result = db.query(Prediction).with_entities(
        db.func.avg(Prediction.confidence_score)
    ).scalar()
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
    
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
    
    predictions = db.query(Prediction).all()
    feedbacks = db.query(Feedback).all()
    
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
    
    for feedback in feedbacks:
        pred = db.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
        if pred:
            queue = pred.predicted_queue
            if queue in queue_stats:
                queue_stats[queue]['feedback_count'] += 1
                if not feedback.is_correct:
                    queue_stats[queue]['corrections_count'] += 1
    
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

@app.post("/api/predict")
async def predict_ticket(
    ticket_data: TicketCreate,
    db: Session = Depends(get_db)
):
    """Predict queue for a single ticket (demo mode)"""
    
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
    
    # Demo prediction logic
    text_content = (ticket_data.title + " " + ticket_data.content).lower()
    
    # Simple keyword-based classification
    if any(word in text_content for word in ['phishing', 'email', 'suspicious']):
        predicted_queue = 'DFIR::phishing'
        confidence = 0.85
    elif any(word in text_content for word in ['vulnerability', 'cve', 'exploit']):
        predicted_queue = 'OFFSEC::CVD'
        confidence = 0.82
    elif any(word in text_content for word in ['incident', 'breach', 'attack']):
        predicted_queue = 'DFIR::incidents'
        confidence = 0.88
    elif any(word in text_content for word in ['threat', 'intelligence', 'ioc']):
        predicted_queue = 'CTI'
        confidence = 0.75
    elif any(word in text_content for word in ['pentest', 'security test', 'assessment']):
        predicted_queue = 'OFFSEC::Pentesting'
        confidence = 0.80
    elif any(word in text_content for word in ['admin', 'management', 'policy']):
        predicted_queue = 'SMS'
        confidence = 0.70
    else:
        predicted_queue = 'Trash'
        confidence = 0.60
    
    # Generate all probabilities
    all_probabilities = {}
    for queue in class_names:
        if queue == predicted_queue:
            all_probabilities[queue] = confidence
        else:
            all_probabilities[queue] = (1 - confidence) / (len(class_names) - 1)
    
    # Determine routing decision
    if confidence >= 0.85:
        routing_decision = "auto_route"
    elif confidence >= 0.65:
        routing_decision = "human_verify"
    else:
        routing_decision = "manual_triage"
    
    # Store prediction
    db_prediction = Prediction(
        ticket_id=db_ticket.id,
        predicted_queue=predicted_queue,
        confidence_score=confidence,
        all_probabilities=all_probabilities,
        model_version="demo_v1",
        processing_time_ms=50,
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
        "confidence_score": confidence,
        "all_probabilities": all_probabilities,
        "routing_decision": routing_decision,
        "processing_time_ms": 50,
        "prediction_id": str(db_prediction.id)
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
