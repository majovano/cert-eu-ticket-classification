"""
Database configuration and models for CERT-EU Ticket Classification System
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, DECIMAL, ForeignKey, JSON
from sqlalchemy.sql import func
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any

# Database URL - will be set from environment variables
import os
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cert_user:cert_password@postgres:5432/cert_eu_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    department = Column(String(255))
    role = Column(String(50), default="reviewer")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id = Column(String(50), unique=True, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_date = Column(DateTime(timezone=True))
    email_address = Column(String(255))
    raw_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id = Column(UUID(as_uuid=True), ForeignKey("tickets.id", ondelete="CASCADE"))
    predicted_queue = Column(String(50), nullable=False)
    confidence_score = Column(DECIMAL(5, 4), nullable=False)
    all_probabilities = Column(JSON)
    model_version = Column(String(50))
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_ms = Column(Integer)
    features_used = Column(JSON)
    routing_decision = Column(String(20), nullable=False)

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"))
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    corrected_queue = Column(String(50))
    feedback_notes = Column(Text)
    keywords_highlighted = Column(JSON)
    difficulty_score = Column(Integer)
    feedback_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_correct = Column(Boolean)

class AmbiguousBatch(Base):
    __tablename__ = "ambiguous_batches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_name = Column(String(255), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    status = Column(String(20), default="pending")
    total_tickets = Column(Integer, nullable=False)
    reviewed_tickets = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    email_sent_at = Column(DateTime(timezone=True))

class AmbiguousTicket(Base):
    __tablename__ = "ambiguous_tickets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), ForeignKey("ambiguous_batches.id", ondelete="CASCADE"))
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"))
    assigned_reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    review_link = Column(String(500))
    status = Column(String(20), default="pending")
    reviewed_at = Column(DateTime(timezone=True))

class ModelMetric(Base):
    __tablename__ = "model_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(DECIMAL(10, 6), nullable=False)
    metric_type = Column(String(50))
    queue_name = Column(String(50))
    calculation_date = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String(50))
    cv_fold = Column(Integer)
    notes = Column(Text)

class CategoryKeyword(Base):
    __tablename__ = "category_keywords"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    queue_name = Column(String(50), nullable=False)
    keyword = Column(String(100), nullable=False)
    frequency = Column(Integer, default=1)
    importance_score = Column(DECIMAL(5, 4))
    last_updated = Column(DateTime(timezone=True), server_default=func.now())

class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(Text, nullable=False)
    description = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

# Pydantic models for API
class TicketCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    ticket_id: str
    title: str
    content: str
    created_date: Optional[datetime] = None
    email_address: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

class PredictionCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    ticket_id: str
    predicted_queue: str
    confidence_score: float
    all_probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None
    features_used: Optional[Dict[str, Any]] = None
    routing_decision: str

class FeedbackCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction_id: str
    reviewer_id: str
    corrected_queue: Optional[str] = None
    feedback_notes: Optional[str] = None
    keywords_highlighted: Optional[List[str]] = None
    difficulty_score: Optional[int] = None
    is_correct: Optional[bool] = None

class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: str
    ticket_id: str
    title: str
    content: str
    created_date: Optional[datetime]
    email_address: Optional[str]
    created_at: datetime

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: str
    ticket_id: str
    predicted_queue: str
    confidence_score: float
    all_probabilities: Optional[Dict[str, float]]
    model_version: Optional[str]
    prediction_timestamp: datetime
    processing_time_ms: Optional[int]
    routing_decision: str

class DashboardStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    total_tickets: int
    total_predictions: int
    auto_routed: int
    human_verify: int
    manual_triage: int
    avg_confidence: float
    total_feedback: int
    corrections_needed: int

class QueuePerformance(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_queue: str
    total_predictions: int
    avg_confidence: float
    auto_routed_count: int
    human_verify_count: int
    manual_triage_count: int
    total_feedback: int
    corrections_count: int
    error_rate_percent: float

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)