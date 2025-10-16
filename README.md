# CERT-EU Ticket Classification System

A comprehensive full-stack machine learning application for automated ticket classification and time series forecasting, built with modern technologies and deployed using containerization.

## 🎯 Project Overview

This system automatically classifies cybersecurity tickets into appropriate queues using machine learning models, provides time series forecasting for ticket volumes, and offers a complete web interface for analysis and management.

## 🏗️ Architecture & Technologies

### **Backend Technologies**
- **FastAPI** - Modern, fast web framework for building APIs with automatic OpenAPI documentation
- **Python 3.11** - Core programming language with type hints and async support
- **SQLAlchemy** - Advanced ORM for database operations with relationship mapping
- **Pydantic** - Data validation and serialization using Python type annotations
- **Uvicorn** - ASGI server for high-performance async Python applications
- **Pandas** - Data manipulation and analysis for time series processing
- **NumPy** - Numerical computing for mathematical operations
- **Scikit-learn** - Machine learning library for model training and evaluation

### **Frontend Technologies**
- **React 18** - Modern JavaScript library with hooks and functional components
- **Recharts** - Composable charting library for data visualization
- **Tailwind CSS** - Utility-first CSS framework for responsive design
- **Axios** - Promise-based HTTP client for API communication
- **React Router** - Client-side routing for single-page application navigation

### **Database & Storage**
- **PostgreSQL** - Robust relational database for production environments
- **SQLite** - Lightweight database for development and demo purposes
- **Database Migrations** - Version-controlled schema management

### **Machine Learning Pipeline**
- **RoBERTa** - Transformer-based model for text classification
- **Time Series Forecasting** - ARIMA and seasonal decomposition for trend analysis
- **Feature Engineering** - Text preprocessing, tokenization, and vectorization
- **Model Evaluation** - Cross-validation, confusion matrices, and performance metrics

### **DevOps & Deployment**
- **Docker** - Containerization for consistent environments
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Reverse proxy and static file serving
- **Git** - Version control with GitHub integration
- **Render.com** - Cloud deployment platform
- **Environment Variables** - Secure configuration management

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+
- Docker & Docker Compose
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/majovano/cert-eu-ticket-classification.git
cd cert-eu-ticket-classification
```

### 2. Local Development Setup

#### Option A: Docker Compose (Recommended)
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

#### Option B: Manual Setup
```bash
# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm start
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📊 Features & Capabilities

### **Core Functionality**
- **Single Ticket Classification** - Real-time ML prediction with confidence scores
- **Batch Processing** - Upload JSONL files for bulk ticket analysis
- **Time Series Forecasting** - Predict future ticket volumes with probability analysis
- **Dashboard Analytics** - Real-time statistics and performance metrics
- **Human Review Interface** - Manual verification and correction system
- **Queue Analysis** - Detailed breakdown by ticket categories
- **Report Generation** - Automated email reports with data export

### **Machine Learning Features**
- **Multi-class Classification** - 7 distinct queue categories (CTI, DFIR, OFFSEC, etc.)
- **Confidence Scoring** - Probability-based routing decisions
- **Model Training Pipeline** - Automated retraining with new data
- **Performance Monitoring** - Accuracy, precision, recall tracking
- **A/B Testing** - Model comparison and validation

### **Time Series Analysis**
- **Historical Data Processing** - Trend analysis and pattern recognition
- **Forecast Generation** - 7-365 day predictions with confidence intervals
- **Probability Analysis** - Likelihood of volume increases/decreases
- **Interactive Charts** - Visual representation of trends and forecasts
- **Data Export** - CSV download for external analysis

## 🏛️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React SPA     │    │   FastAPI       │    │   PostgreSQL   │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Database)    │
│   Port 3000     │    │   Port 8000     │    │   Port 5432     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   ML Models     │    │   Data Import   │
│   (Reverse      │    │   (RoBERTa)     │    │   (JSONL)       │
│    Proxy)       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration & Environment

### **Environment Variables**
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cert_eu_db

# Security
SECRET_KEY=your-secret-key-here

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Application
ENVIRONMENT=production
DEBUG=false
```

### **Database Schema**
- **Tickets** - Core ticket information and metadata
- **Predictions** - ML model outputs with confidence scores
- **Feedback** - Human corrections for model improvement
- **Time Series** - Historical data for forecasting
- **Users** - Authentication and access control

## 📈 Performance & Scalability

### **Optimization Strategies**
- **Async/Await** - Non-blocking I/O operations
- **Database Indexing** - Optimized queries for large datasets
- **Caching** - Redis for session and model caching
- **Load Balancing** - Horizontal scaling capabilities
- **CDN Integration** - Static asset optimization

### **Monitoring & Logging**
- **Application Logs** - Structured logging with levels
- **Performance Metrics** - Response time and throughput
- **Error Tracking** - Exception handling and reporting
- **Health Checks** - Service availability monitoring

## 🚀 Deployment Options

### **Option 1: Render.com (Recommended)**
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy to production"
git push origin main

# 2. Connect to Render
# - Go to https://render.com
# - Connect GitHub repository
# - Configure environment variables
# - Deploy automatically
```

### **Option 2: Docker Production**
```bash
# Build production image
docker build -t cert-eu-app .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e SECRET_KEY=... \
  cert-eu-app
```

### **Option 3: Local Development**
```bash
# Start with Docker Compose
docker compose up -d

# Or manual setup
cd backend && uvicorn main:app --reload
cd frontend && npm start
```

## 🧪 Testing & Quality Assurance

### **Testing Strategy**
- **Unit Tests** - Individual component testing
- **Integration Tests** - API endpoint validation
- **End-to-End Tests** - Full user workflow testing
- **Performance Tests** - Load and stress testing
- **Security Tests** - Vulnerability assessment

### **Code Quality**
- **Type Hints** - Python type annotations throughout
- **Linting** - ESLint and Pylint for code quality
- **Formatting** - Black and Prettier for consistent style
- **Documentation** - Comprehensive docstrings and comments

## 📚 API Documentation

### **Core Endpoints**
```python
# Ticket Classification
POST /api/predict
POST /api/predict/batch

# Dashboard Data
GET /api/dashboard/stats
GET /api/dashboard/queue-performance

# Time Series
GET /api/time-series/forecast
GET /api/time-series/historical
GET /api/time-series/trends

# Human Review
GET /api/tickets/low-confidence
POST /api/feedback

# Reports
POST /api/reports/email
GET /api/reports/export
```

### **Data Models**
```python
class Ticket(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    created_at: datetime

class Prediction(BaseModel):
    ticket_id: str
    predicted_queue: str
    confidence_score: float
    prediction_timestamp: datetime
```

## 🔒 Security Considerations

### **Data Protection**
- **Input Validation** - Pydantic models for data sanitization
- **SQL Injection Prevention** - Parameterized queries
- **CORS Configuration** - Cross-origin request security
- **Rate Limiting** - API abuse prevention
- **Authentication** - JWT token-based security

### **Privacy & Compliance**
- **Data Encryption** - Sensitive information protection
- **Audit Logging** - User action tracking
- **GDPR Compliance** - Data retention policies
- **Access Control** - Role-based permissions

## 🎓 Technical Interview Points

### **Full-Stack Expertise**
- **Backend Architecture** - RESTful API design with FastAPI
- **Frontend Development** - Modern React with hooks and context
- **Database Design** - Relational modeling with SQLAlchemy
- **Machine Learning** - End-to-end ML pipeline implementation
- **DevOps** - Containerization and cloud deployment

### **Advanced Concepts**
- **Async Programming** - Python async/await patterns
- **State Management** - React hooks and context API
- **Data Processing** - Pandas for time series analysis
- **Model Deployment** - Production ML model serving
- **Monitoring** - Application performance tracking

### **Problem-Solving Skills**
- **Error Handling** - Comprehensive exception management
- **Performance Optimization** - Database queries and caching
- **Scalability** - Horizontal scaling strategies
- **Security** - Input validation and authentication
- **Testing** - Quality assurance methodologies

## 📞 Support & Contact

For technical questions or deployment assistance:
- **GitHub Issues**: [Repository Issues](https://github.com/majovano/cert-eu-ticket-classification/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: mjovanovjr@gmail.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for CERT-EU cybersecurity operations**