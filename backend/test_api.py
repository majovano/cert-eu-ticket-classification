"""
Comprehensive test suite for FastAPI backend endpoints.
Tests cover all major functionality including predictions, time series, and data management.
"""

import pytest
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock
import tempfile
import os

from main import app, get_db
from database import Base, get_database_url
from models import Ticket, Prediction, Feedback

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
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

@pytest.fixture
def sample_ticket():
    """Sample ticket data for testing."""
    return {
        "title": "Test Security Incident",
        "description": "Potential malware detected in user workstation",
        "priority": "high",
        "category": "security"
    }

@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return [
        {
            "id": "ticket_001",
            "title": "Phishing Email Report",
            "description": "Suspicious email with malicious attachment",
            "priority": "medium"
        },
        {
            "id": "ticket_002", 
            "title": "Network Anomaly",
            "description": "Unusual network traffic patterns detected",
            "priority": "high"
        }
    ]

class TestHealthEndpoints:
    """Test health check and basic endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

class TestPredictionEndpoints:
    """Test prediction-related endpoints."""
    
    def test_single_prediction_success(self, client, sample_ticket):
        """Test successful single ticket prediction."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "DFIR::incidents",
                "confidence": 0.85,
                "reasoning": "Security incident detected"
            }
            
            response = client.post("/api/predict", json=sample_ticket)
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction_id" in data
            assert data["predicted_queue"] == "DFIR::incidents"
            assert data["confidence_score"] == 0.85
            assert data["routing_decision"] in ["auto_route", "human_verify", "manual_triage"]
    
    def test_single_prediction_validation(self, client):
        """Test prediction with invalid data."""
        invalid_ticket = {
            "title": "",  # Empty title should fail validation
            "description": "Test description"
        }
        
        response = client.post("/api/predict", json=invalid_ticket)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_success(self, client, sample_batch_data):
        """Test successful batch prediction."""
        with patch('main.get_demo_prediction') as mock_prediction:
            mock_prediction.return_value = {
                "queue": "CTI",
                "confidence": 0.75,
                "reasoning": "Intelligence gathering"
            }
            
            # Create temporary JSONL file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for item in sample_batch_data:
                    f.write(json.dumps(item) + '\n')
                temp_file_path = f.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    response = client.post(
                        "/api/predict/batch",
                        files={"file": ("test.jsonl", f, "application/json")}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_processed"] == 2
                assert data["successful_predictions"] == 2
                assert data["failed_predictions"] == 0
                
            finally:
                os.unlink(temp_file_path)
    
    def test_batch_prediction_file_too_large(self, client):
        """Test batch prediction with file size limit."""
        # Create a large file (simulate)
        large_data = {"description": "x" * 1000000}  # 1MB of data
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for _ in range(100):  # Create multiple large entries
                f.write(json.dumps(large_data) + '\n')
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/predict/batch",
                    files={"file": ("large.jsonl", f, "application/json")}
                )
            
            # Should return 413 (Payload Too Large) or handle gracefully
            assert response.status_code in [200, 413]
            
        finally:
            os.unlink(temp_file_path)

class TestDashboardEndpoints:
    """Test dashboard and statistics endpoints."""
    
    def test_dashboard_stats(self, client):
        """Test dashboard statistics endpoint."""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_tickets" in data
        assert "total_predictions" in data
        assert "auto_routed" in data
        assert "human_verify" in data
        assert "manual_triage" in data
        assert "avg_confidence" in data
    
    def test_queue_performance(self, client):
        """Test queue performance endpoint."""
        response = client.get("/api/dashboard/queue-performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "queue_performance" in data
        assert isinstance(data["queue_performance"], list)
    
    def test_dashboard_with_filters(self, client):
        """Test dashboard with date and limit filters."""
        response = client.get("/api/dashboard/stats?date_from=2024-01-01&date_to=2024-12-31&ticket_limit=100")
        assert response.status_code == 200
        
        data = response.json()
        assert "date_from" in data
        assert "date_to" in data
        assert "ticket_limit" in data

class TestTimeSeriesEndpoints:
    """Test time series forecasting endpoints."""
    
    def test_forecast_endpoint(self, client):
        """Test time series forecast endpoint."""
        response = client.get("/api/time-series/forecast?days=7")
        assert response.status_code == 200
        
        data = response.json()
        assert "forecast_days" in data
        assert "forecasts" in data
        assert "probability_analysis" in data
        assert "method" in data
        assert "confidence" in data
    
    def test_historical_data(self, client):
        """Test historical data endpoint."""
        response = client.get("/api/time-series/historical?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert "summary" in data
        assert isinstance(data["data"], list)
    
    def test_trends_endpoint(self, client):
        """Test trends analysis endpoint."""
        response = client.get("/api/time-series/trends")
        assert response.status_code == 200
        
        data = response.json()
        assert "trends" in data
        assert isinstance(data["trends"], dict)
    
    def test_forecast_with_different_periods(self, client):
        """Test forecast with different time periods."""
        for days in [7, 30, 90]:
            response = client.get(f"/api/time-series/forecast?days={days}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["forecast_days"] == days

class TestHumanReviewEndpoints:
    """Test human review and feedback endpoints."""
    
    def test_low_confidence_tickets(self, client):
        """Test low confidence tickets endpoint."""
        response = client.get("/api/tickets/low-confidence?limit=10&threshold=0.75")
        assert response.status_code == 200
        
        data = response.json()
        assert "tickets" in data
        assert "total_count" in data
        assert isinstance(data["tickets"], list)
    
    def test_feedback_submission(self, client):
        """Test feedback submission endpoint."""
        feedback_data = {
            "prediction_id": "test_prediction_123",
            "correct_queue": "CTI",
            "feedback_type": "correction",
            "comments": "Model was incorrect, should be CTI not DFIR"
        }
        
        response = client.post("/api/feedback", json=feedback_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Feedback recorded successfully"
        assert "feedback_id" in data

class TestReportEndpoints:
    """Test reporting and export endpoints."""
    
    def test_email_report(self, client):
        """Test email report generation."""
        with patch('main.send_email_report') as mock_email:
            mock_email.return_value = True
            
            report_data = {
                "email": "test@example.com",
                "report_type": "summary",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31"
            }
            
            response = client.post("/api/reports/email", json=report_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["message"] == "Report sent successfully"
    
    def test_export_report(self, client):
        """Test report export endpoint."""
        response = client.get("/api/reports/export?format=csv&date_from=2024-01-01&date_to=2024-12-31")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv"

class TestDataImport:
    """Test data import functionality."""
    
    def test_import_sample_data(self, client):
        """Test sample data import endpoint."""
        response = client.post("/api/admin/import-sample-data")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "imported_tickets" in data
        assert "imported_predictions" in data

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/api/invalid-endpoint")
        assert response.status_code == 404
    
    def test_malformed_json(self, client):
        """Test malformed JSON returns 422."""
        response = client.post("/api/predict", data="invalid json")
        assert response.status_code == 422
    
    def test_database_connection_error(self, client):
        """Test database connection error handling."""
        with patch('main.get_db') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/dashboard/stats")
            assert response.status_code == 500

class TestPerformance:
    """Test performance and load handling."""
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/api/dashboard/stats")
            results.append(response.status_code)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    def test_large_batch_processing(self, client):
        """Test processing of large batch files."""
        # Create a batch with many tickets
        large_batch = []
        for i in range(100):
            large_batch.append({
                "id": f"ticket_{i:03d}",
                "title": f"Test Ticket {i}",
                "description": f"Test description for ticket {i}",
                "priority": "medium"
            })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in large_batch:
                f.write(json.dumps(item) + '\n')
            temp_file_path = f.name
        
        try:
            with patch('main.get_demo_prediction') as mock_prediction:
                mock_prediction.return_value = {
                    "queue": "CTI",
                    "confidence": 0.75,
                    "reasoning": "Test prediction"
                }
                
                with open(temp_file_path, 'rb') as f:
                    response = client.post(
                        "/api/predict/batch",
                        files={"file": ("large_batch.jsonl", f, "application/json")}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_processed"] == 100
                
        finally:
            os.unlink(temp_file_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
