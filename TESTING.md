# Testing Documentation

## 🧪 Comprehensive Test Suite

This document outlines the complete testing strategy for the CERT-EU Ticket Classification System, demonstrating quality assurance practices and technical expertise.

## 📋 Test Coverage Overview

### **Backend Tests**
- **API Tests** (`test_api.py`) - FastAPI endpoint testing
- **ML Model Tests** (`test_ml_models.py`) - Machine learning functionality
- **Database Tests** (`test_database.py`) - Data persistence and relationships
- **Integration Tests** (`test_integration.py`) - End-to-end workflows

### **Frontend Tests**
- **Component Tests** - React component testing with Jest and React Testing Library
- **Integration Tests** - User interaction and API integration
- **Accessibility Tests** - WCAG compliance and keyboard navigation
- **Performance Tests** - Rendering and memory usage

## 🚀 Running Tests

### **Backend Testing**
```bash
# Install test dependencies
cd backend
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only
pytest -m api          # API tests only
pytest -m database     # Database tests only
pytest -m ml           # ML model tests only

# Run with verbose output
pytest -v

# Run specific test file
pytest test_api.py -v

# Run specific test function
pytest test_api.py::TestPredictionEndpoints::test_single_prediction_success -v
```

### **Frontend Testing**
```bash
# Install test dependencies
cd frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom jest

# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch

# Run specific test file
npm test TicketAnalyzer.test.js

# Run with verbose output
npm test -- --verbose
```

## 🔧 Test Configuration

### **Backend Configuration**
- **pytest.ini** - Test discovery and execution configuration
- **Coverage thresholds** - Minimum 80% coverage required
- **Test markers** - Categorize tests by type and performance
- **Parallel execution** - Run tests concurrently for speed

### **Frontend Configuration**
- **Jest** - JavaScript testing framework
- **React Testing Library** - Component testing utilities
- **jsdom** - DOM simulation for browser testing
- **Coverage reporting** - HTML and console coverage reports

## 📊 Test Categories

### **1. Unit Tests**
Test individual functions and components in isolation.

**Backend Examples:**
```python
def test_get_demo_prediction():
    """Test demo prediction generation."""
    model = load_demo_model()
    prediction = get_demo_prediction("Malware detected", model)
    
    assert "queue" in prediction
    assert "confidence" in prediction
    assert 0.0 <= prediction["confidence"] <= 1.0
```

**Frontend Examples:**
```javascript
test('handles form input changes', () => {
  renderWithRouter(<TicketAnalyzer />);
  
  const titleInput = screen.getByLabelText(/title/i);
  fireEvent.change(titleInput, { target: { value: 'Test Ticket' } });
  
  expect(titleInput.value).toBe('Test Ticket');
});
```

### **2. Integration Tests**
Test component interactions and API communication.

**Backend Examples:**
```python
def test_single_ticket_workflow(client):
    """Test complete single ticket analysis workflow."""
    with patch('main.get_demo_prediction') as mock_prediction:
        mock_prediction.return_value = {
            "queue": "DFIR::incidents",
            "confidence": 0.85,
            "reasoning": "Security incident detected"
        }
        
        response = client.post("/api/predict", json=ticket_data)
        assert response.status_code == 200
        
        # Verify database consistency
        db = next(get_db())
        ticket = db.query(Ticket).filter(Ticket.title == "Test").first()
        assert ticket is not None
```

**Frontend Examples:**
```javascript
test('submits single ticket analysis successfully', async () => {
  const mockResponse = {
    data: {
      prediction_id: 'test-123',
      predicted_queue: 'CTI',
      confidence_score: 0.85
    }
  };
  
  mockedAxios.post.mockResolvedValueOnce(mockResponse);
  
  renderWithRouter(<TicketAnalyzer />);
  
  fireEvent.change(screen.getByLabelText(/title/i), { 
    target: { value: 'Test Ticket' } 
  });
  fireEvent.click(screen.getByRole('button', { name: /analyze ticket/i }));
  
  await waitFor(() => {
    expect(mockedAxios.post).toHaveBeenCalledWith('/api/predict', {
      title: 'Test Ticket',
      description: 'Test description',
      priority: 'high',
      category: 'security'
    });
  });
});
```

### **3. Performance Tests**
Test application performance under load.

**Backend Examples:**
```python
def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import threading
    
    results = []
    
    def make_request():
        response = client.get("/api/dashboard/stats")
        results.append(response.status_code)
    
    threads = []
    for _ in range(20):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert all(status == 200 for status in results)
```

### **4. Database Tests**
Test data persistence and relationships.

**Examples:**
```python
def test_ticket_prediction_consistency(client):
    """Test consistency between tickets and predictions."""
    # Create ticket
    response = client.post("/api/predict", json=ticket_data)
    prediction_data = response.json()
    
    # Verify database consistency
    db = next(get_db())
    ticket = db.query(Ticket).filter(Ticket.title == "Test").first()
    prediction = db.query(Prediction).filter(
        Prediction.ticket_id == ticket.id
    ).first()
    
    assert prediction.ticket_id == ticket.id
    assert prediction.predicted_queue == "DFIR::incidents"
```

### **5. Machine Learning Tests**
Test ML model functionality and accuracy.

**Examples:**
```python
def test_prediction_accuracy():
    """Test prediction accuracy on known cases."""
    model = load_demo_model()
    
    test_cases = [
        {
            "description": "malware virus trojan",
            "expected_queues": ["DFIR::incidents", "OFFSEC::CVD"]
        },
        {
            "description": "phishing email scam",
            "expected_queues": ["DFIR::phishing", "SMS"]
        }
    ]
    
    for case in test_cases:
        prediction = get_demo_prediction(case["description"], model)
        assert prediction["queue"] in case["expected_queues"]
```

## 🎯 Test Quality Metrics

### **Coverage Requirements**
- **Backend**: Minimum 80% code coverage
- **Frontend**: Minimum 70% code coverage
- **Critical paths**: 100% coverage required
- **API endpoints**: 100% coverage required

### **Performance Benchmarks**
- **API response time**: < 200ms for 95% of requests
- **Database queries**: < 100ms for 95% of queries
- **ML predictions**: < 50ms per prediction
- **Frontend rendering**: < 100ms for component updates

### **Error Handling**
- **Graceful degradation** for all error conditions
- **Proper HTTP status codes** for all responses
- **User-friendly error messages** for all failures
- **Logging and monitoring** for all errors

## 🔍 Test Data Management

### **Test Fixtures**
```python
@pytest.fixture
def sample_ticket():
    """Create a sample ticket for testing."""
    return {
        "title": "Test Security Incident",
        "description": "Malware detected in user workstation",
        "priority": "high",
        "category": "security"
    }

@pytest.fixture
def sample_prediction():
    """Create a sample prediction for testing."""
    return {
        "queue": "DFIR::incidents",
        "confidence": 0.85,
        "reasoning": "Security incident detected"
    }
```

### **Mock Data**
```javascript
const mockTicketData = {
  title: 'Test Security Incident',
  description: 'Malware detected in user workstation',
  priority: 'high',
  category: 'security'
};

const mockPredictionResponse = {
  prediction_id: 'test-123',
  predicted_queue: 'DFIR::incidents',
  confidence_score: 0.85,
  routing_decision: 'auto_route'
};
```

## 🚨 Error Testing

### **Backend Error Scenarios**
- **Database connection failures**
- **ML model loading errors**
- **Invalid input validation**
- **File upload errors**
- **Timeout handling**

### **Frontend Error Scenarios**
- **Network connectivity issues**
- **API response errors**
- **Form validation errors**
- **File upload failures**
- **Component rendering errors**

## 📈 Continuous Integration

### **GitHub Actions Workflow**
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run backend tests
        run: pytest --cov=. --cov-report=xml
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: 16
      - name: Install frontend dependencies
        run: cd frontend && npm install
      - name: Run frontend tests
        run: cd frontend && npm test -- --coverage
```

## 🎓 Interview Demonstration

### **Technical Skills Demonstrated**
1. **Testing Strategy** - Comprehensive test coverage
2. **Quality Assurance** - Automated testing and CI/CD
3. **Performance Testing** - Load testing and benchmarking
4. **Error Handling** - Graceful failure management
5. **Code Quality** - Clean, maintainable test code

### **Key Points to Highlight**
- **Test Coverage**: 80%+ coverage across all components
- **Test Categories**: Unit, integration, performance, and E2E tests
- **Mocking Strategy**: Proper isolation of dependencies
- **Data Management**: Test fixtures and mock data
- **CI/CD Integration**: Automated testing pipeline
- **Performance Monitoring**: Benchmarking and optimization

### **Code Examples to Show**
- **Complex test scenarios** with multiple assertions
- **Mock implementations** for external dependencies
- **Performance benchmarks** with timing measurements
- **Error handling tests** with exception scenarios
- **Integration tests** with database and API interactions

This comprehensive testing suite demonstrates professional software development practices and quality assurance expertise that will impress any technical interviewer.
