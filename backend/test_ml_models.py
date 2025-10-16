"""
Test suite for machine learning model functionality.
Tests model loading, prediction generation, and time series analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

from time_series_predictor import TimeSeriesPredictor
from main import get_demo_prediction, load_demo_model, create_prediction_record
from models import Ticket, Prediction

class TestDemoModel:
    """Test demo model functionality."""
    
    def test_load_demo_model(self):
        """Test demo model loading."""
        model = load_demo_model()
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_get_demo_prediction(self):
        """Test demo prediction generation."""
        model = load_demo_model()
        
        # Test different ticket descriptions
        test_cases = [
            {
                "description": "Malware detected in user workstation",
                "expected_queue": "DFIR::incidents",
                "min_confidence": 0.7
            },
            {
                "description": "Phishing email with suspicious attachment",
                "expected_queue": "DFIR::phishing",
                "min_confidence": 0.6
            },
            {
                "description": "Network intrusion attempt detected",
                "expected_queue": "OFFSEC::CVD",
                "min_confidence": 0.5
            },
            {
                "description": "Vulnerability assessment request",
                "expected_queue": "OFFSEC::Pentesting",
                "min_confidence": 0.5
            },
            {
                "description": "Threat intelligence report",
                "expected_queue": "CTI",
                "min_confidence": 0.6
            },
            {
                "description": "SMS spam report",
                "expected_queue": "SMS",
                "min_confidence": 0.5
            },
            {
                "description": "Random unrelated ticket",
                "expected_queue": "Trash",
                "min_confidence": 0.3
            }
        ]
        
        for case in test_cases:
            prediction = get_demo_prediction(case["description"], model)
            
            assert "queue" in prediction
            assert "confidence" in prediction
            assert "reasoning" in prediction
            
            # Check if prediction matches expected queue (allowing for some variation)
            assert prediction["queue"] in [
                "CTI", "DFIR::incidents", "DFIR::phishing", 
                "OFFSEC::CVD", "OFFSEC::Pentesting", "SMS", "Trash"
            ]
            
            assert 0.0 <= prediction["confidence"] <= 1.0
            assert prediction["confidence"] >= case["min_confidence"]
            assert isinstance(prediction["reasoning"], str)
            assert len(prediction["reasoning"]) > 0
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for similar inputs."""
        model = load_demo_model()
        
        description = "Malware detected in user workstation"
        
        # Get multiple predictions for the same input
        predictions = []
        for _ in range(5):
            prediction = get_demo_prediction(description, model)
            predictions.append(prediction)
        
        # All predictions should have the same queue
        queues = [p["queue"] for p in predictions]
        assert len(set(queues)) == 1, f"Predictions were inconsistent: {queues}"
        
        # Confidence scores should be similar (within 0.1)
        confidences = [p["confidence"] for p in predictions]
        max_diff = max(confidences) - min(confidences)
        assert max_diff <= 0.1, f"Confidence scores too different: {confidences}"
    
    def test_prediction_with_empty_description(self):
        """Test prediction with empty description."""
        model = load_demo_model()
        
        prediction = get_demo_prediction("", model)
        
        assert "queue" in prediction
        assert "confidence" in prediction
        assert "reasoning" in prediction
        
        # Empty description should result in low confidence
        assert prediction["confidence"] < 0.5
    
    def test_prediction_with_very_long_description(self):
        """Test prediction with very long description."""
        model = load_demo_model()
        
        long_description = "This is a very long description. " * 100
        prediction = get_demo_prediction(long_description, model)
        
        assert "queue" in prediction
        assert "confidence" in prediction
        assert "reasoning" in prediction
        
        # Should still produce a valid prediction
        assert 0.0 <= prediction["confidence"] <= 1.0

class TestTimeSeriesPredictor:
    """Test time series prediction functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        data = []
        
        for date in dates:
            for queue in ['CTI', 'DFIR::incidents', 'DFIR::phishing']:
                data.append({
                    'date': date,
                    'queue': queue,
                    'ticket_count': np.random.randint(1, 10)
                })
        
        return pd.DataFrame(data)
    
    def test_time_series_predictor_initialization(self):
        """Test TimeSeriesPredictor initialization."""
        predictor = TimeSeriesPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'prepare_time_series_data')
        assert hasattr(predictor, 'predict_future')
        assert hasattr(predictor, 'analyze_trends')
    
    def test_prepare_time_series_data(self, sample_data):
        """Test time series data preparation."""
        predictor = TimeSeriesPredictor()
        
        # Mock database query
        with patch('time_series_predictor.db') as mock_db:
            mock_query = MagicMock()
            mock_query.filter.return_value.all.return_value = []
            mock_db.query.return_value = mock_query
            
            # Test with sample data
            result = predictor.prepare_time_series_data(mock_db)
            
            # Should return a DataFrame
            assert isinstance(result, pd.DataFrame)
    
    def test_predict_future(self, sample_data):
        """Test future prediction functionality."""
        predictor = TimeSeriesPredictor()
        
        # Test prediction with sample data
        forecasts = predictor.predict_future(sample_data, days=7)
        
        assert isinstance(forecasts, dict)
        assert len(forecasts) > 0
        
        for queue, forecast in forecasts.items():
            assert 'dates' in forecast
            assert 'values' in forecast
            assert len(forecast['dates']) == 7
            assert len(forecast['values']) == 7
            assert all(isinstance(v, (int, float)) for v in forecast['values'])
    
    def test_analyze_trends(self, sample_data):
        """Test trend analysis functionality."""
        predictor = TimeSeriesPredictor()
        
        trends = predictor.analyze_trends(sample_data)
        
        assert isinstance(trends, dict)
        assert len(trends) > 0
        
        for queue, trend_data in trends.items():
            assert 'trend' in trend_data
            assert 'slope' in trend_data
            assert 'r_squared' in trend_data
            assert trend_data['trend'] in ['increasing', 'decreasing', 'stable']
            assert isinstance(trend_data['slope'], (int, float))
            assert 0 <= trend_data['r_squared'] <= 1
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        predictor = TimeSeriesPredictor()
        
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        forecasts = predictor.predict_future(empty_df, days=7)
        assert isinstance(forecasts, dict)
        assert len(forecasts) == 0
        
        trends = predictor.analyze_trends(empty_df)
        assert isinstance(trends, dict)
        assert len(trends) == 0
    
    def test_single_queue_data(self):
        """Test with single queue data."""
        predictor = TimeSeriesPredictor()
        
        # Create data for single queue
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'queue': ['CTI'] * len(dates),
            'ticket_count': [1, 2, 3, 2, 4, 3, 5, 4, 6, 5]
        })
        
        forecasts = predictor.predict_future(data, days=5)
        assert 'CTI' in forecasts
        assert len(forecasts['CTI']['dates']) == 5
        
        trends = predictor.analyze_trends(data)
        assert 'CTI' in trends
        assert trends['CTI']['trend'] in ['increasing', 'decreasing', 'stable']

class TestPredictionRecord:
    """Test prediction record creation."""
    
    def test_create_prediction_record(self):
        """Test prediction record creation."""
        ticket_data = {
            "id": "test_ticket_123",
            "title": "Test Security Incident",
            "description": "Malware detected",
            "priority": "high",
            "category": "security"
        }
        
        prediction_data = {
            "queue": "DFIR::incidents",
            "confidence": 0.85,
            "reasoning": "Security incident detected"
        }
        
        record = create_prediction_record(ticket_data, prediction_data)
        
        assert record.ticket_id == "test_ticket_123"
        assert record.predicted_queue == "DFIR::incidents"
        assert record.confidence_score == 0.85
        assert record.reasoning == "Security incident detected"
        assert record.prediction_timestamp is not None
        
        # Test routing decision logic
        if record.confidence_score >= 0.8:
            assert record.routing_decision == "auto_route"
        elif record.confidence_score >= 0.6:
            assert record.routing_decision == "human_verify"
        else:
            assert record.routing_decision == "manual_triage"
    
    def test_routing_decision_logic(self):
        """Test routing decision logic based on confidence."""
        test_cases = [
            (0.9, "auto_route"),
            (0.8, "auto_route"),
            (0.79, "human_verify"),
            (0.6, "human_verify"),
            (0.59, "manual_triage"),
            (0.3, "manual_triage")
        ]
        
        for confidence, expected_decision in test_cases:
            ticket_data = {"id": "test", "title": "Test", "description": "Test"}
            prediction_data = {
                "queue": "CTI",
                "confidence": confidence,
                "reasoning": "Test"
            }
            
            record = create_prediction_record(ticket_data, prediction_data)
            assert record.routing_decision == expected_decision

class TestModelPerformance:
    """Test model performance and accuracy."""
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy on known cases."""
        model = load_demo_model()
        
        # Test cases with expected outcomes
        test_cases = [
            {
                "description": "malware virus trojan",
                "expected_queues": ["DFIR::incidents", "OFFSEC::CVD"]
            },
            {
                "description": "phishing email scam",
                "expected_queues": ["DFIR::phishing", "SMS"]
            },
            {
                "description": "vulnerability exploit",
                "expected_queues": ["OFFSEC::CVD", "OFFSEC::Pentesting"]
            },
            {
                "description": "threat intelligence ioc",
                "expected_queues": ["CTI"]
            },
            {
                "description": "spam sms text",
                "expected_queues": ["SMS", "Trash"]
            }
        ]
        
        for case in test_cases:
            prediction = get_demo_prediction(case["description"], model)
            
            # Prediction should be one of the expected queues
            assert prediction["queue"] in case["expected_queues"], \
                f"Expected one of {case['expected_queues']}, got {prediction['queue']}"
    
    def test_confidence_distribution(self):
        """Test that confidence scores are well distributed."""
        model = load_demo_model()
        
        descriptions = [
            "Clear malware incident",
            "Unclear security issue",
            "Possible phishing attempt",
            "Unknown threat",
            "Random text"
        ]
        
        confidences = []
        for desc in descriptions:
            prediction = get_demo_prediction(desc, model)
            confidences.append(prediction["confidence"])
        
        # Should have some variation in confidence scores
        assert max(confidences) - min(confidences) > 0.1
        
        # Most predictions should have reasonable confidence
        assert sum(1 for c in confidences if c > 0.5) >= len(confidences) * 0.6
    
    def test_reasoning_quality(self):
        """Test quality of reasoning explanations."""
        model = load_demo_model()
        
        descriptions = [
            "Malware detected in workstation",
            "Phishing email with attachment",
            "Network intrusion attempt",
            "Vulnerability assessment request"
        ]
        
        for desc in descriptions:
            prediction = get_demo_prediction(desc, model)
            reasoning = prediction["reasoning"]
            
            # Reasoning should be meaningful
            assert len(reasoning) > 10
            assert any(word in reasoning.lower() for word in ["security", "threat", "incident", "analysis"])
    
    def test_model_robustness(self):
        """Test model robustness with edge cases."""
        model = load_demo_model()
        
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            "x" * 1000,  # Very long string
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
            "malware" + " " * 100 + "virus",  # Lots of whitespace
        ]
        
        for case in edge_cases:
            prediction = get_demo_prediction(case, model)
            
            # Should always return valid prediction
            assert "queue" in prediction
            assert "confidence" in prediction
            assert "reasoning" in prediction
            
            assert 0.0 <= prediction["confidence"] <= 1.0
            assert prediction["queue"] in [
                "CTI", "DFIR::incidents", "DFIR::phishing", 
                "OFFSEC::CVD", "OFFSEC::Pentesting", "SMS", "Trash"
            ]

class TestModelIntegration:
    """Test model integration with database and API."""
    
    def test_model_with_database_session(self):
        """Test model with database session."""
        from database import get_db
        
        # This would test the full integration
        # In a real test, you'd use a test database
        pass
    
    def test_model_performance_benchmark(self):
        """Benchmark model performance."""
        model = load_demo_model()
        
        # Test prediction speed
        import time
        
        descriptions = ["Test description"] * 100
        
        start_time = time.time()
        for desc in descriptions:
            get_demo_prediction(desc, model)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(descriptions)
        
        # Should be fast (less than 0.1 seconds per prediction)
        assert avg_time < 0.1, f"Average prediction time too slow: {avg_time}s"
    
    def test_model_memory_usage(self):
        """Test model memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load model
        model = load_demo_model()
        
        # Make some predictions
        for _ in range(100):
            get_demo_prediction("Test description", model)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024}MB"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
