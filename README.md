# CERT-EU Ticket Classification System

A comprehensive AI-powered ticket classification system for the European Union's Cybersecurity Emergency Response Team (CERT-EU). This system automatically routes security tickets to appropriate queues using a hybrid transformer model and provides a professional web interface for human review and analysis.

## 🚀 Features

### Core Functionality
- **Hybrid ML Model**: RoBERTa + numerical features with attention-based fusion
- **7 Queue Classification**: CTI, DFIR::incidents, DFIR::phishing, OFFSEC::CVD, OFFSEC::Pentesting, SMS, Trash
- **Confidence-based Routing**: Auto-route, human verification, and manual triage
- **Cross-validation Support**: Robust model evaluation with stratified k-fold CV
- **Real-time Predictions**: FastAPI backend with batch processing capabilities

### Web Interface
- **Modern Dashboard**: EU-branded React frontend with comprehensive analytics
- **Ticket Analyzer**: Single ticket and batch file analysis
- **Queue Analysis**: Detailed performance metrics and trends
- **Human Review Interface**: Low-confidence ticket review with feedback system
- **Reporting System**: Exportable reports in JSON and CSV formats

### Database & Analytics
- **PostgreSQL Database**: Comprehensive schema for tickets, predictions, and feedback
- **Performance Monitoring**: Queue-specific metrics and confidence analysis
- **Feedback Loop**: Human corrections for continuous model improvement
- **Keyword Analysis**: Security keyword extraction and importance scoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  PostgreSQL DB  │
│                 │    │                 │    │                 │
│ • Dashboard     │◄──►│ • Prediction    │◄──►│ • Tickets       │
│ • Ticket Analyzer│    │ • Analytics     │    │ • Predictions   │
│ • Human Review  │    │ • Feedback      │    │ • Feedback      │
│ • Reports       │    │ • Model Mgmt    │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │  ML Model       │
                    │                 │
                    │ • RoBERTa       │
                    │ • Feature Fusion│
                    │ • Confidence    │
                    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Primary database
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **Scikit-learn**: ML utilities and metrics

### Frontend
- **React 18**: Modern UI framework
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Data visualization
- **Lucide React**: Icon library
- **React Router**: Client-side routing

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Docker & Docker Compose (optional)

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd react-cert-eu-ml-challenge
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

### Option 2: Manual Setup

1. **Setup Database**
   ```bash
   # Start PostgreSQL
   docker run -d --name cert-eu-postgres \
     -e POSTGRES_DB=cert_eu_db \
     -e POSTGRES_USER=cert_user \
     -e POSTGRES_PASSWORD=cert_password \
     -p 5432:5432 postgres:15
   
   # Initialize schema
   psql -h localhost -U cert_user -d cert_eu_db -f database/schema.sql
   ```

2. **Setup Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 📊 Model Training

### Train the Model
```bash
cd code
python main.py --data_path data/train_dataset.jsonl --epochs 5 --batch_size 16
```

### Generate Predictions
```bash
python predict.py --test_data data/test_dataset.jsonl --model_dir ./models --output data/test_predictions.jsonl
```

### Cross-Validation
```bash
python main.py --data_path data/train_dataset.jsonl --cross_validate --cv_folds 5 --epochs 5
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:
```env
DATABASE_URL=postgresql://cert_user:cert_password@localhost:5432/cert_eu_db
SECRET_KEY=your-secret-key-here
MODEL_PATH=./models/hybrid_roberta_model
```

### System Configuration
The system uses configurable thresholds:
- **High Confidence**: 0.85 (auto-route)
- **Low Confidence**: 0.65 (manual triage)
- **Medium Confidence**: 0.65-0.85 (human verification)

## 📈 Usage

### 1. Dashboard
- View overall system statistics
- Monitor queue performance
- Track confidence distributions
- Analyze routing decisions

### 2. Ticket Analyzer
- Analyze individual tickets
- Upload batch files (JSONL format)
- View prediction probabilities
- Monitor processing times

### 3. Queue Analysis
- Detailed queue performance metrics
- Trend analysis over time
- Confidence vs error rate analysis
- Queue-specific insights

### 4. Human Review
- Review low-confidence predictions
- Provide feedback and corrections
- Highlight relevant keywords
- Rate classification difficulty

### 5. Reports
- Generate comprehensive reports
- Export data in JSON/CSV formats
- Customize report types and date ranges
- Track system performance over time

## 🔍 API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `GET /api/dashboard/stats` - Dashboard statistics
- `GET /api/dashboard/queue-performance` - Queue performance metrics

### Prediction Endpoints
- `POST /api/predict` - Single ticket prediction
- `POST /api/predict/batch` - Batch file prediction
- `GET /api/tickets/low-confidence` - Low confidence tickets

### Feedback Endpoints
- `POST /api/feedback` - Submit human feedback
- `GET /api/keywords/{queue_name}` - Queue keywords analysis

## 🎨 EU Branding

The interface uses official EU colors and styling:
- **Primary Blue**: #003399 (EU Blue)
- **Secondary Yellow**: #FFCC02 (EU Yellow)
- **Accent Colors**: Various shades for different queues
- **Typography**: Inter font family
- **Icons**: Lucide React icon set

## 📊 Queue Descriptions

| Queue | Description | Color |
|-------|-------------|-------|
| **CTI** | Cyber Threat Intelligence - Analysis of threat indicators | Blue |
| **DFIR::incidents** | Active security incidents requiring investigation | Red |
| **DFIR::phishing** | Phishing attacks and email-based threats | Orange |
| **OFFSEC::CVD** | Coordinated Vulnerability Disclosure reports | Purple |
| **OFFSEC::Pentesting** | Penetration testing and security assessments | Green |
| **SMS** | Security Management Services - Administrative tasks | Teal |
| **Trash** | Irrelevant or spam tickets | Gray |

## 🔒 Security Considerations

- Input validation and sanitization
- SQL injection prevention
- CORS configuration
- Rate limiting (recommended for production)
- Authentication system (to be implemented)
- HTTPS enforcement (for production)

## 🚀 Deployment

### Production Deployment
1. Set up production PostgreSQL database
2. Configure environment variables
3. Set up reverse proxy (nginx)
4. Enable HTTPS
5. Configure monitoring and logging
6. Set up backup strategies

### Scaling Considerations
- Horizontal scaling with multiple backend instances
- Database read replicas for analytics
- Redis for caching frequently accessed data
- Message queues for batch processing
- Load balancing for high availability

## 📝 Data Import

Import existing predictions:
```bash
cd backend
python data_import.py
```

This will import the `test_predictions.jsonl` file into the database with mock confidence scores and routing decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the EU Public License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the CERT-EU team
- Check the API documentation at `/api/docs`

## 🔮 Future Enhancements

- **Real-time Monitoring**: WebSocket connections for live updates
- **Advanced Analytics**: Machine learning model performance tracking
- **Email Integration**: Direct email processing and routing
- **Mobile App**: React Native mobile application
- **Multi-language Support**: Internationalization
- **Advanced Security**: OAuth2, RBAC, audit logging
- **Model Versioning**: A/B testing and model comparison
- **Automated Retraining**: Continuous learning pipeline

---

**Built with ❤️ for the European Union Cybersecurity Emergency Response Team**
