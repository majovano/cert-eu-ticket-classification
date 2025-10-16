"""
Test suite for database operations and models.
Tests CRUD operations, relationships, and data integrity.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import uuid

from database import Base, get_database_url
from models import Ticket, Prediction, Feedback, User

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_ticket():
    """Create a sample ticket for testing."""
    return Ticket(
        id=str(uuid.uuid4()),
        title="Test Security Incident",
        description="Malware detected in user workstation",
        priority="high",
        category="security",
        status="open",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

@pytest.fixture
def sample_prediction(sample_ticket):
    """Create a sample prediction for testing."""
    return Prediction(
        id=str(uuid.uuid4()),
        ticket_id=sample_ticket.id,
        predicted_queue="DFIR::incidents",
        confidence_score=0.85,
        routing_decision="auto_route",
        reasoning="Security incident detected",
        prediction_timestamp=datetime.now()
    )

class TestTicketModel:
    """Test Ticket model operations."""
    
    def test_create_ticket(self, db_session, sample_ticket):
        """Test creating a ticket."""
        db_session.add(sample_ticket)
        db_session.commit()
        
        # Verify ticket was created
        ticket = db_session.query(Ticket).filter(Ticket.id == sample_ticket.id).first()
        assert ticket is not None
        assert ticket.title == "Test Security Incident"
        assert ticket.description == "Malware detected in user workstation"
        assert ticket.priority == "high"
        assert ticket.category == "security"
        assert ticket.status == "open"
    
    def test_ticket_validation(self, db_session):
        """Test ticket validation rules."""
        # Test required fields
        ticket = Ticket()
        db_session.add(ticket)
        
        with pytest.raises(Exception):
            db_session.commit()
    
    def test_ticket_timestamps(self, db_session):
        """Test automatic timestamp handling."""
        ticket = Ticket(
            id=str(uuid.uuid4()),
            title="Test Ticket",
            description="Test Description",
            priority="medium",
            category="test"
        )
        
        db_session.add(ticket)
        db_session.commit()
        
        assert ticket.created_at is not None
        assert ticket.updated_at is not None
        assert ticket.created_at <= datetime.now()
        assert ticket.updated_at <= datetime.now()
    
    def test_ticket_update(self, db_session, sample_ticket):
        """Test updating a ticket."""
        db_session.add(sample_ticket)
        db_session.commit()
        
        # Update ticket
        sample_ticket.status = "closed"
        sample_ticket.updated_at = datetime.now()
        db_session.commit()
        
        # Verify update
        updated_ticket = db_session.query(Ticket).filter(Ticket.id == sample_ticket.id).first()
        assert updated_ticket.status == "closed"
    
    def test_ticket_deletion(self, db_session, sample_ticket):
        """Test deleting a ticket."""
        db_session.add(sample_ticket)
        db_session.commit()
        
        # Delete ticket
        db_session.delete(sample_ticket)
        db_session.commit()
        
        # Verify deletion
        ticket = db_session.query(Ticket).filter(Ticket.id == sample_ticket.id).first()
        assert ticket is None
    
    def test_ticket_relationships(self, db_session, sample_ticket, sample_prediction):
        """Test ticket relationships with predictions."""
        db_session.add(sample_ticket)
        db_session.add(sample_prediction)
        db_session.commit()
        
        # Test relationship
        ticket = db_session.query(Ticket).filter(Ticket.id == sample_ticket.id).first()
        predictions = db_session.query(Prediction).filter(Prediction.ticket_id == ticket.id).all()
        
        assert len(predictions) == 1
        assert predictions[0].predicted_queue == "DFIR::incidents"

class TestPredictionModel:
    """Test Prediction model operations."""
    
    def test_create_prediction(self, db_session, sample_ticket, sample_prediction):
        """Test creating a prediction."""
        db_session.add(sample_ticket)
        db_session.add(sample_prediction)
        db_session.commit()
        
        # Verify prediction was created
        prediction = db_session.query(Prediction).filter(Prediction.id == sample_prediction.id).first()
        assert prediction is not None
        assert prediction.ticket_id == sample_ticket.id
        assert prediction.predicted_queue == "DFIR::incidents"
        assert prediction.confidence_score == 0.85
        assert prediction.routing_decision == "auto_route"
    
    def test_prediction_validation(self, db_session):
        """Test prediction validation rules."""
        # Test required fields
        prediction = Prediction()
        db_session.add(prediction)
        
        with pytest.raises(Exception):
            db_session.commit()
    
    def test_prediction_confidence_range(self, db_session, sample_ticket):
        """Test confidence score validation."""
        # Valid confidence scores
        valid_confidences = [0.0, 0.5, 0.85, 1.0]
        
        for confidence in valid_confidences:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue="CTI",
                confidence_score=confidence,
                routing_decision="auto_route",
                reasoning="Test",
                prediction_timestamp=datetime.now()
            )
            
            db_session.add(prediction)
            db_session.commit()
            
            # Verify prediction was created
            pred = db_session.query(Prediction).filter(Prediction.id == prediction.id).first()
            assert pred.confidence_score == confidence
    
    def test_prediction_routing_decisions(self, db_session, sample_ticket):
        """Test routing decision logic."""
        test_cases = [
            (0.9, "auto_route"),
            (0.7, "human_verify"),
            (0.4, "manual_triage")
        ]
        
        for confidence, expected_decision in test_cases:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue="CTI",
                confidence_score=confidence,
                routing_decision=expected_decision,
                reasoning="Test",
                prediction_timestamp=datetime.now()
            )
            
            db_session.add(prediction)
            db_session.commit()
            
            # Verify routing decision
            pred = db_session.query(Prediction).filter(Prediction.id == prediction.id).first()
            assert pred.routing_decision == expected_decision
    
    def test_prediction_timestamps(self, db_session, sample_ticket):
        """Test prediction timestamp handling."""
        prediction = Prediction(
            id=str(uuid.uuid4()),
            ticket_id=sample_ticket.id,
            predicted_queue="CTI",
            confidence_score=0.8,
            routing_decision="auto_route",
            reasoning="Test",
            prediction_timestamp=datetime.now()
        )
        
        db_session.add(prediction)
        db_session.commit()
        
        assert prediction.prediction_timestamp is not None
        assert prediction.prediction_timestamp <= datetime.now()

class TestFeedbackModel:
    """Test Feedback model operations."""
    
    def test_create_feedback(self, db_session, sample_prediction):
        """Test creating feedback."""
        feedback = Feedback(
            id=str(uuid.uuid4()),
            prediction_id=sample_prediction.id,
            correct_queue="CTI",
            feedback_type="correction",
            comments="Model was incorrect",
            created_at=datetime.now()
        )
        
        db_session.add(feedback)
        db_session.commit()
        
        # Verify feedback was created
        fb = db_session.query(Feedback).filter(Feedback.id == feedback.id).first()
        assert fb is not None
        assert fb.prediction_id == sample_prediction.id
        assert fb.correct_queue == "CTI"
        assert fb.feedback_type == "correction"
    
    def test_feedback_types(self, db_session, sample_prediction):
        """Test different feedback types."""
        feedback_types = ["correction", "confirmation", "improvement"]
        
        for fb_type in feedback_types:
            feedback = Feedback(
                id=str(uuid.uuid4()),
                prediction_id=sample_prediction.id,
                correct_queue="CTI",
                feedback_type=fb_type,
                comments=f"Test {fb_type}",
                created_at=datetime.now()
            )
            
            db_session.add(feedback)
            db_session.commit()
            
            # Verify feedback type
            fb = db_session.query(Feedback).filter(Feedback.id == feedback.id).first()
            assert fb.feedback_type == fb_type

class TestUserModel:
    """Test User model operations."""
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            id=str(uuid.uuid4()),
            username="test_user",
            email="test@example.com",
            role="analyst",
            is_active=True,
            created_at=datetime.now()
        )
        
        db_session.add(user)
        db_session.commit()
        
        # Verify user was created
        u = db_session.query(User).filter(User.id == user.id).first()
        assert u is not None
        assert u.username == "test_user"
        assert u.email == "test@example.com"
        assert u.role == "analyst"
        assert u.is_active is True
    
    def test_user_roles(self, db_session):
        """Test different user roles."""
        roles = ["admin", "analyst", "reviewer", "viewer"]
        
        for role in roles:
            user = User(
                id=str(uuid.uuid4()),
                username=f"user_{role}",
                email=f"{role}@example.com",
                role=role,
                is_active=True,
                created_at=datetime.now()
            )
            
            db_session.add(user)
            db_session.commit()
            
            # Verify role
            u = db_session.query(User).filter(User.id == user.id).first()
            assert u.role == role

class TestDatabaseQueries:
    """Test complex database queries."""
    
    def test_get_tickets_by_priority(self, db_session):
        """Test querying tickets by priority."""
        priorities = ["low", "medium", "high", "critical"]
        
        for priority in priorities:
            ticket = Ticket(
                id=str(uuid.uuid4()),
                title=f"Test {priority} ticket",
                description="Test description",
                priority=priority,
                category="test",
                created_at=datetime.now()
            )
            db_session.add(ticket)
        
        db_session.commit()
        
        # Query high priority tickets
        high_priority_tickets = db_session.query(Ticket).filter(Ticket.priority == "high").all()
        assert len(high_priority_tickets) == 1
        assert high_priority_tickets[0].priority == "high"
    
    def test_get_predictions_by_confidence(self, db_session, sample_ticket):
        """Test querying predictions by confidence threshold."""
        confidences = [0.3, 0.5, 0.7, 0.9]
        
        for confidence in confidences:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue="CTI",
                confidence_score=confidence,
                routing_decision="auto_route",
                reasoning="Test",
                prediction_timestamp=datetime.now()
            )
            db_session.add(prediction)
        
        db_session.commit()
        
        # Query high confidence predictions
        high_confidence = db_session.query(Prediction).filter(Prediction.confidence_score >= 0.8).all()
        assert len(high_confidence) == 1
        assert high_confidence[0].confidence_score == 0.9
    
    def test_get_predictions_by_queue(self, db_session, sample_ticket):
        """Test querying predictions by queue."""
        queues = ["CTI", "DFIR::incidents", "DFIR::phishing", "OFFSEC::CVD"]
        
        for queue in queues:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue=queue,
                confidence_score=0.8,
                routing_decision="auto_route",
                reasoning="Test",
                prediction_timestamp=datetime.now()
            )
            db_session.add(prediction)
        
        db_session.commit()
        
        # Query CTI predictions
        cti_predictions = db_session.query(Prediction).filter(Prediction.predicted_queue == "CTI").all()
        assert len(cti_predictions) == 1
        assert cti_predictions[0].predicted_queue == "CTI"
    
    def test_get_predictions_by_date_range(self, db_session, sample_ticket):
        """Test querying predictions by date range."""
        now = datetime.now()
        dates = [
            now - timedelta(days=5),
            now - timedelta(days=3),
            now - timedelta(days=1),
            now
        ]
        
        for date in dates:
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue="CTI",
                confidence_score=0.8,
                routing_decision="auto_route",
                reasoning="Test",
                prediction_timestamp=date
            )
            db_session.add(prediction)
        
        db_session.commit()
        
        # Query predictions from last 2 days
        recent_predictions = db_session.query(Prediction).filter(
            Prediction.prediction_timestamp >= now - timedelta(days=2)
        ).all()
        assert len(recent_predictions) == 2
    
    def test_aggregate_queries(self, db_session, sample_ticket):
        """Test aggregate queries for statistics."""
        # Create multiple predictions
        for i in range(10):
            prediction = Prediction(
                id=str(uuid.uuid4()),
                ticket_id=sample_ticket.id,
                predicted_queue="CTI",
                confidence_score=0.5 + (i * 0.05),
                routing_decision="auto_route",
                reasoning="Test",
                prediction_timestamp=datetime.now()
            )
            db_session.add(prediction)
        
        db_session.commit()
        
        # Test count query
        total_predictions = db_session.query(Prediction).count()
        assert total_predictions == 10
        
        # Test average confidence
        from sqlalchemy import func
        avg_confidence = db_session.query(func.avg(Prediction.confidence_score)).scalar()
        assert avg_confidence is not None
        assert 0.0 <= avg_confidence <= 1.0
        
        # Test group by query
        queue_counts = db_session.query(
            Prediction.predicted_queue,
            func.count(Prediction.id)
        ).group_by(Prediction.predicted_queue).all()
        
        assert len(queue_counts) == 1
        assert queue_counts[0][0] == "CTI"
        assert queue_counts[0][1] == 10

class TestDatabaseConstraints:
    """Test database constraints and relationships."""
    
    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraints."""
        # Try to create prediction without valid ticket
        prediction = Prediction(
            id=str(uuid.uuid4()),
            ticket_id="non_existent_ticket",
            predicted_queue="CTI",
            confidence_score=0.8,
            routing_decision="auto_route",
            reasoning="Test",
            prediction_timestamp=datetime.now()
        )
        
        db_session.add(prediction)
        
        # Should raise foreign key constraint error
        with pytest.raises(Exception):
            db_session.commit()
    
    def test_unique_constraints(self, db_session):
        """Test unique constraints."""
        # Create user with unique username
        user1 = User(
            id=str(uuid.uuid4()),
            username="unique_user",
            email="user1@example.com",
            role="analyst",
            is_active=True,
            created_at=datetime.now()
        )
        
        db_session.add(user1)
        db_session.commit()
        
        # Try to create another user with same username
        user2 = User(
            id=str(uuid.uuid4()),
            username="unique_user",  # Same username
            email="user2@example.com",
            role="analyst",
            is_active=True,
            created_at=datetime.now()
        )
        
        db_session.add(user2)
        
        # Should raise unique constraint error
        with pytest.raises(Exception):
            db_session.commit()
    
    def test_cascade_deletions(self, db_session, sample_ticket, sample_prediction):
        """Test cascade deletion behavior."""
        db_session.add(sample_ticket)
        db_session.add(sample_prediction)
        db_session.commit()
        
        # Create feedback for prediction
        feedback = Feedback(
            id=str(uuid.uuid4()),
            prediction_id=sample_prediction.id,
            correct_queue="CTI",
            feedback_type="correction",
            comments="Test",
            created_at=datetime.now()
        )
        db_session.add(feedback)
        db_session.commit()
        
        # Delete ticket (should cascade to predictions and feedback)
        db_session.delete(sample_ticket)
        db_session.commit()
        
        # Verify cascade deletion
        ticket = db_session.query(Ticket).filter(Ticket.id == sample_ticket.id).first()
        prediction = db_session.query(Prediction).filter(Prediction.id == sample_prediction.id).first()
        feedback_record = db_session.query(Feedback).filter(Feedback.id == feedback.id).first()
        
        assert ticket is None
        assert prediction is None
        assert feedback_record is None

class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    def test_bulk_insert_performance(self, db_session):
        """Test bulk insert performance."""
        import time
        
        # Create many tickets
        tickets = []
        for i in range(1000):
            ticket = Ticket(
                id=str(uuid.uuid4()),
                title=f"Test Ticket {i}",
                description=f"Test Description {i}",
                priority="medium",
                category="test",
                created_at=datetime.now()
            )
            tickets.append(ticket)
        
        start_time = time.time()
        db_session.add_all(tickets)
        db_session.commit()
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0  # Less than 5 seconds
        
        # Verify all tickets were inserted
        count = db_session.query(Ticket).count()
        assert count == 1000
    
    def test_query_performance(self, db_session):
        """Test query performance with indexes."""
        # Create test data
        tickets = []
        for i in range(100):
            ticket = Ticket(
                id=str(uuid.uuid4()),
                title=f"Test Ticket {i}",
                description=f"Test Description {i}",
                priority="high" if i % 10 == 0 else "medium",
                category="test",
                created_at=datetime.now()
            )
            tickets.append(ticket)
        
        db_session.add_all(tickets)
        db_session.commit()
        
        # Test query performance
        import time
        
        start_time = time.time()
        high_priority_tickets = db_session.query(Ticket).filter(Ticket.priority == "high").all()
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second
        assert len(high_priority_tickets) == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
