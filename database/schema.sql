-- CERT-EU Ticket Classification Database Schema
-- PostgreSQL Database for ML Model Predictions and Human Feedback

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table for human reviewers
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(255),
    role VARCHAR(50) DEFAULT 'reviewer', -- reviewer, admin, analyst
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Tickets table - original ticket data
CREATE TABLE tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id VARCHAR(50) UNIQUE NOT NULL, -- Original ticket ID from dataset
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_date TIMESTAMP WITH TIME ZONE,
    email_address VARCHAR(255),
    raw_data JSONB, -- Store original JSON data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID REFERENCES tickets(id) ON DELETE CASCADE,
    predicted_queue VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    all_probabilities JSONB, -- Store all class probabilities
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER, -- Time taken for prediction
    features_used JSONB, -- Store extracted features for analysis
    routing_decision VARCHAR(20) NOT NULL -- 'auto_route', 'human_verify', 'manual_triage'
);

-- Human feedback table for corrections
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    reviewer_id UUID REFERENCES users(id),
    corrected_queue VARCHAR(50),
    feedback_notes TEXT,
    keywords_highlighted JSONB, -- Store highlighted keywords
    difficulty_score INTEGER CHECK (difficulty_score BETWEEN 1 AND 5), -- 1=very easy, 5=very difficult
    feedback_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_correct BOOLEAN -- Whether the original prediction was correct
);

-- Ambiguous tickets batch processing
CREATE TABLE ambiguous_batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_name VARCHAR(255) NOT NULL,
    created_by UUID REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending', -- pending, in_review, completed
    total_tickets INTEGER NOT NULL,
    reviewed_tickets INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    email_sent_at TIMESTAMP WITH TIME ZONE
);

-- Link ambiguous tickets to batches
CREATE TABLE ambiguous_tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID REFERENCES ambiguous_batches(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    assigned_reviewer_id UUID REFERENCES users(id),
    review_link VARCHAR(500), -- Unique link for human review
    status VARCHAR(20) DEFAULT 'pending', -- pending, reviewed, corrected
    reviewed_at TIMESTAMP WITH TIME ZONE
);

-- Model performance metrics
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    metric_type VARCHAR(50), -- accuracy, f1_macro, f1_weighted, etc.
    queue_name VARCHAR(50), -- Specific queue metric
    calculation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    cv_fold INTEGER, -- For cross-validation results
    notes TEXT
);

-- Keywords analysis per category
CREATE TABLE category_keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_name VARCHAR(50) NOT NULL,
    keyword VARCHAR(100) NOT NULL,
    frequency INTEGER DEFAULT 1,
    importance_score DECIMAL(5,4), -- TF-IDF or custom importance
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(queue_name, keyword)
);

-- System configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id)
);

-- Create indexes for performance
CREATE INDEX idx_tickets_ticket_id ON tickets(ticket_id);
CREATE INDEX idx_predictions_ticket_id ON predictions(ticket_id);
CREATE INDEX idx_predictions_routing_decision ON predictions(routing_decision);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score);
CREATE INDEX idx_predictions_predicted_queue ON predictions(predicted_queue);
CREATE INDEX idx_feedback_prediction_id ON feedback(prediction_id);
CREATE INDEX idx_feedback_reviewer_id ON feedback(reviewer_id);
CREATE INDEX idx_ambiguous_tickets_batch_id ON ambiguous_tickets(batch_id);
CREATE INDEX idx_ambiguous_tickets_status ON ambiguous_tickets(status);
CREATE INDEX idx_model_metrics_calculation_date ON model_metrics(calculation_date);
CREATE INDEX idx_category_keywords_queue ON category_keywords(queue_name);

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('confidence_threshold_high', '0.85', 'High confidence threshold for auto-routing'),
('confidence_threshold_low', '0.65', 'Low confidence threshold for manual triage'),
('batch_size_ambiguous', '100', 'Number of ambiguous tickets per batch'),
('model_version', 'hybrid_roberta_v1', 'Current model version'),
('email_notification_enabled', 'true', 'Enable email notifications for ambiguous batches');

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_tickets_updated_at BEFORE UPDATE ON tickets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create view for dashboard statistics
CREATE VIEW dashboard_stats AS
SELECT 
    COUNT(DISTINCT t.id) as total_tickets,
    COUNT(DISTINCT p.id) as total_predictions,
    COUNT(DISTINCT CASE WHEN p.routing_decision = 'auto_route' THEN p.id END) as auto_routed,
    COUNT(DISTINCT CASE WHEN p.routing_decision = 'human_verify' THEN p.id END) as human_verify,
    COUNT(DISTINCT CASE WHEN p.routing_decision = 'manual_triage' THEN p.id END) as manual_triage,
    AVG(p.confidence_score) as avg_confidence,
    COUNT(DISTINCT f.id) as total_feedback,
    COUNT(DISTINCT CASE WHEN f.is_correct = false THEN f.id END) as corrections_needed
FROM tickets t
LEFT JOIN predictions p ON t.id = p.ticket_id
LEFT JOIN feedback f ON p.id = f.prediction_id;

-- Create view for queue performance
CREATE VIEW queue_performance AS
SELECT 
    p.predicted_queue,
    COUNT(*) as total_predictions,
    AVG(p.confidence_score) as avg_confidence,
    COUNT(CASE WHEN p.routing_decision = 'auto_route' THEN 1 END) as auto_routed_count,
    COUNT(CASE WHEN p.routing_decision = 'human_verify' THEN 1 END) as human_verify_count,
    COUNT(CASE WHEN p.routing_decision = 'manual_triage' THEN 1 END) as manual_triage_count,
    COUNT(f.id) as total_feedback,
    COUNT(CASE WHEN f.is_correct = false THEN 1 END) as corrections_count,
    ROUND(COUNT(CASE WHEN f.is_correct = false THEN 1 END)::decimal / NULLIF(COUNT(f.id), 0) * 100, 2) as error_rate_percent
FROM predictions p
LEFT JOIN feedback f ON p.id = f.prediction_id
GROUP BY p.predicted_queue
ORDER BY total_predictions DESC;
