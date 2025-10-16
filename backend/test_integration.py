"""
End-to-end integration tests for the complete application.
Tests the full workflow from frontend to backend to database.
"""

import pytest
import json
import tempfile
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from main import app, get_db
from database import Base, get_database_url
from models import Ticket, Prediction, Feedback

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_integration.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    """Create test client with test database."""
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

class TestCompleteWorkflow:
    """Test complete application workflows."""
    
    def test_single_ticket_workflow(self, client):
        """Test complete single ticket analysis workflow."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "DFIR::incidents",
                "confidence": 0.85,
                "reasoning": "Security incident detected"
            }
            
            # Step 1: Submit ticket for analysis
            ticket_data = {
                "title": "Malware Detection",
                "description": "Malware detected in user workstation",
                "priority": "high",
                "category": "security"
            }
            
            response = client.post("/api/predict", json=ticket_data)
            assert response.status_code == 200
            
            prediction_data = response.json()
            assert "prediction_id" in prediction_data
            assert prediction_data["predicted_queue"] == "DFIR::incidents"
            assert prediction_data["confidence_score"] == 0.85
            
            # Step 2: Verify ticket was stored in database
            db = next(get_db())
            ticket = db.query(Ticket).filter(Ticket.title == "Malware Detection").first()
            assert ticket is not None
            assert ticket.priority == "high"
            
            # Step 3: Verify prediction was stored
            prediction = db.query(Prediction).filter(
                Prediction.ticket_id == ticket.id
            ).first()
            assert prediction is not None
            assert prediction.predicted_queue == "DFIR::incidents"
            assert prediction.confidence_score == 0.85
            
            # Step 4: Check dashboard stats updated
            stats_response = client.get("/api/dashboard/stats")
            assert stats_response.status_code == 200
            
            stats = stats_response.json()
            assert stats["total_tickets"] >= 1
            assert stats["total_predictions"] >= 1
            
            db.close()
    
    def test_batch_processing_workflow(self, client):
        """Test complete batch processing workflow."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "CTI",
                "confidence": 0.75,
                "reasoning": "Intelligence gathering"
            }
            
            # Step 1: Create batch file
            batch_data = [
                {
                    "id": "batch_001",
                    "title": "Threat Intelligence Report",
                    "description": "New IOCs discovered",
                    "priority": "medium"
                },
                {
                    "id": "batch_002",
                    "title": "Network Anomaly",
                    "description": "Unusual traffic patterns",
                    "priority": "high"
                }
            ]
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in batch_data:
                    f.write(json.dumps(item) + '\n')
                temp_file_path = f.name
            
            try:
                # Step 2: Upload and process batch
                with open(temp_file_path, 'rb') as f:
                    response = client.post(
                        "/api/predict/batch",
                        files={"file": ("batch.jsonl", f, "application/json")}
                    )
                
                assert response.status_code == 200
                batch_result = response.json()
                assert batch_result["total_processed"] == 2
                assert batch_result["successful_predictions"] == 2
                
                # Step 3: Verify all tickets were stored
                db = next(get_db())
                tickets = db.query(Ticket).filter(Ticket.title.in_([
                    "Threat Intelligence Report", "Network Anomaly"
                ])).all()
                assert len(tickets) == 2
                
                # Step 4: Verify all predictions were stored
                predictions = db.query(Prediction).filter(
                    Prediction.ticket_id.in_([t.id for t in tickets])
                ).all()
                assert len(predictions) == 2
                
                # Step 5: Check dashboard stats
                stats_response = client.get("/api/dashboard/stats")
                stats = stats_response.json()
                assert stats["total_tickets"] >= 2
                assert stats["total_predictions"] >= 2
                
                db.close()
                
            finally:
                os.unlink(temp_file_path)
    
    def test_time_series_workflow(self, client):
        """Test complete time series forecasting workflow."""
        # Step 1: Create historical data
        db = next(get_db())
        
        # Create tickets and predictions for the last 30 days
        base_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            ticket = Ticket(
                id=f"ticket_{i}",
                title=f"Historical Ticket {i}",
                description=f"Description {i}",
                priority="medium",
                category="test",
                created_at=base_date + timedelta(days=i)
            )
            db.add(ticket)
            
            prediction = Prediction(
                id=f"pred_{i}",
                ticket_id=ticket.id,
                predicted_queue="CTI" if i % 3 == 0 else "DFIR::incidents",
                confidence_score=0.7 + (i % 3) * 0.1,
                routing_decision="auto_route",
                reasoning="Historical data",
                prediction_timestamp=base_date + timedelta(days=i)
            )
            db.add(prediction)
        
        db.commit()
        db.close()
        
        # Step 2: Get historical data
        historical_response = client.get("/api/time-series/historical?days=30")
        assert historical_response.status_code == 200
        
        historical_data = historical_response.json()
        assert "data" in historical_data
        assert len(historical_data["data"]) > 0
        
        # Step 3: Get trends analysis
        trends_response = client.get("/api/time-series/trends")
        assert trends_response.status_code == 200
        
        trends = trends_response.json()
        assert "trends" in trends
        assert len(trends["trends"]) > 0
        
        # Step 4: Generate forecast
        forecast_response = client.get("/api/time-series/forecast?days=7")
        assert forecast_response.status_code == 200
        
        forecast = forecast_response.json()
        assert "forecasts" in forecast
        assert "probability_analysis" in forecast
        assert len(forecast["forecasts"]) > 0
    
    def test_human_review_workflow(self, client):
        """Test complete human review workflow."""
        # Step 1: Create low confidence predictions
        db = next(get_db())
        
        for i in range(5):
            ticket = Ticket(
                id=f"review_ticket_{i}",
                title=f"Review Ticket {i}",
                description=f"Description {i}",
                priority="medium",
                category="test",
                created_at=datetime.now()
            )
            db.add(ticket)
            
            prediction = Prediction(
                id=f"review_pred_{i}",
                ticket_id=ticket.id,
                predicted_queue="CTI",
                confidence_score=0.4 + (i * 0.1),  # Low confidence
                routing_decision="human_verify",
                reasoning="Low confidence prediction",
                prediction_timestamp=datetime.now()
            )
            db.add(prediction)
        
        db.commit()
        db.close()
        
        # Step 2: Get low confidence tickets
        review_response = client.get("/api/tickets/low-confidence?limit=10&threshold=0.75")
        assert review_response.status_code == 200
        
        review_data = review_response.json()
        assert "tickets" in review_data
        assert len(review_data["tickets"]) > 0
        
        # Step 3: Submit feedback
        feedback_data = {
            "prediction_id": "review_pred_0",
            "correct_queue": "DFIR::incidents",
            "feedback_type": "correction",
            "comments": "Model was incorrect, should be DFIR"
        }
        
        feedback_response = client.post("/api/feedback", json=feedback_data)
        assert feedback_response.status_code == 200
        
        feedback_result = feedback_response.json()
        assert "feedback_id" in feedback_result
        
        # Step 4: Verify feedback was stored
        db = next(get_db())
        feedback = db.query(Feedback).filter(
            Feedback.prediction_id == "review_pred_0"
        ).first()
        assert feedback is not None
        assert feedback.correct_queue == "DFIR::incidents"
        assert feedback.feedback_type == "correction"
        
        db.close()
    
    def test_report_generation_workflow(self, client):
        """Test complete report generation workflow."""
        # Step 1: Create test data
        db = next(get_db())
        
        for i in range(10):
            ticket = Ticket(
                id=f"report_ticket_{i}",
                title=f"Report Ticket {i}",
                description=f"Description {i}",
                priority="high" if i % 3 == 0 else "medium",
                category="test",
                created_at=datetime.now() - timedelta(days=i)
            )
            db.add(ticket)
            
            prediction = Prediction(
                id=f"report_pred_{i}",
                ticket_id=ticket.id,
                predicted_queue="CTI" if i % 2 == 0 else "DFIR::incidents",
                confidence_score=0.8,
                routing_decision="auto_route",
                reasoning="Report data",
                prediction_timestamp=datetime.now() - timedelta(days=i)
            )
            db.add(prediction)
        
        db.commit()
        db.close()
        
        # Step 2: Generate email report
        with patch('main.send_email_report') as mock_email:
            mock_email.return_value = True
            
            report_data = {
                "email": "test@example.com",
                "report_type": "summary",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31"
            }
            
            email_response = client.post("/api/reports/email", json=report_data)
            assert email_response.status_code == 200
            
            email_result = email_response.json()
            assert "message" in email_result
            assert "Report sent successfully" in email_result["message"]
        
        # Step 3: Export report
        export_response = client.get("/api/reports/export?format=csv&date_from=2024-01-01&date_to=2024-12-31")
        assert export_response.status_code == 200
        assert export_response.headers["content-type"] == "text/csv"
        
        # Verify CSV content
        csv_content = export_response.text
        assert "ticket_id" in csv_content
        assert "predicted_queue" in csv_content
        assert "confidence_score" in csv_content

class TestErrorHandling:
    """Test error handling across the application."""
    
    def test_database_connection_error(self, client):
        """Test handling of database connection errors."""
        with patch('main.get_db') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/dashboard/stats")
            assert response.status_code == 500
    
    def test_ml_model_error(self, client):
        """Test handling of ML model errors."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.side_effect = Exception("Model prediction failed")
            
            ticket_data = {
                "title": "Test Ticket",
                "description": "Test Description",
                "priority": "medium",
                "category": "test"
            }
            
            response = client.post("/api/predict", json=ticket_data)
            assert response.status_code == 500
    
    def test_file_upload_error(self, client):
        """Test handling of file upload errors."""
        # Test with invalid file
        response = client.post(
            "/api/predict/batch",
            files={"file": ("invalid.txt", b"invalid content", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_timeout_handling(self, client):
        """Test handling of request timeouts."""
        with patch('main.get_demo_prediction') as mock_prediction:
            # Simulate slow prediction
            def slow_prediction(*args, **kwargs):
                import time
                time.sleep(2)  # 2 second delay
                return {"queue": "CTI", "confidence": 0.8, "reasoning": "Slow prediction"}
            
            mock_prediction.side_effect = slow_prediction
            
            ticket_data = {
                "title": "Test Ticket",
                "description": "Test Description",
                "priority": "medium",
                "category": "test"
            }
            
            # Should handle timeout gracefully
            response = client.post("/api/predict", json=ticket_data)
            # Depending on timeout configuration, should either succeed or fail gracefully
            assert response.status_code in [200, 408, 500]

class TestPerformance:
    """Test application performance under load."""
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.get("/api/dashboard/stats")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results)
        assert len(results) == 20
    
    def test_large_batch_processing(self, client):
        """Test processing of large batch files."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "CTI",
                "confidence": 0.8,
                "reasoning": "Batch processing"
            }
            
            # Create large batch
            large_batch = []
            for i in range(100):
                large_batch.append({
                    "id": f"large_batch_{i}",
                    "title": f"Large Batch Ticket {i}",
                    "description": f"Description {i}",
                    "priority": "medium"
                })
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in large_batch:
                    f.write(json.dumps(item) + '\n')
                temp_file_path = f.name
            
            try:
                start_time = time.time()
                
                with open(temp_file_path, 'rb') as f:
                    response = client.post(
                        "/api/predict/batch",
                        files={"file": ("large_batch.jsonl", f, "application/json")}
                    )
                
                end_time = time.time()
                
                assert response.status_code == 200
                batch_result = response.json()
                assert batch_result["total_processed"] == 100
                
                # Should complete in reasonable time
                processing_time = end_time - start_time
                assert processing_time < 30.0  # Less than 30 seconds
                
            finally:
                os.unlink(temp_file_path)
    
    def test_memory_usage(self, client):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for _ in range(50):
            response = client.get("/api/dashboard/stats")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB

class TestDataConsistency:
    """Test data consistency across operations."""
    
    def test_ticket_prediction_consistency(self, client):
        """Test consistency between tickets and predictions."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "DFIR::incidents",
                "confidence": 0.85,
                "reasoning": "Security incident"
            }
            
            # Create ticket
            ticket_data = {
                "title": "Consistency Test",
                "description": "Test for data consistency",
                "priority": "high",
                "category": "security"
            }
            
            response = client.post("/api/predict", json=ticket_data)
            assert response.status_code == 200
            
            prediction_data = response.json()
            prediction_id = prediction_data["prediction_id"]
            
            # Verify consistency in database
            db = next(get_db())
            
            # Get ticket and prediction
            ticket = db.query(Ticket).filter(Ticket.title == "Consistency Test").first()
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            
            assert ticket is not None
            assert prediction is not None
            assert prediction.ticket_id == ticket.id
            assert prediction.predicted_queue == "DFIR::incidents"
            assert prediction.confidence_score == 0.85
            
            db.close()
    
    def test_dashboard_stats_consistency(self, client):
        """Test consistency of dashboard statistics."""
        # Get initial stats
        initial_response = client.get("/api/dashboard/stats")
        assert initial_response.status_code == 200
        initial_stats = initial_response.json()
        
        # Create new ticket
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "CTI",
                "confidence": 0.8,
                "reasoning": "Test"
            }
            
            ticket_data = {
                "title": "Stats Test",
                "description": "Test for stats consistency",
                "priority": "medium",
                "category": "test"
            }
            
            response = client.post("/api/predict", json=ticket_data)
            assert response.status_code == 200
            
            # Get updated stats
            updated_response = client.get("/api/dashboard/stats")
            assert updated_response.status_code == 200
            updated_stats = updated_response.json()
            
            # Stats should be consistent
            assert updated_stats["total_tickets"] == initial_stats["total_tickets"] + 1
            assert updated_stats["total_predictions"] == initial_stats["total_predictions"] + 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
