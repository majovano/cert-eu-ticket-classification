"""
Script to import test_predictions.jsonl into the database
"""

import json
import sys
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from database import SessionLocal, Ticket, Prediction, create_tables
import pandas as pd
import numpy as np

def import_test_predictions():
    """Import test predictions from JSONL file"""
    
    # Initialize database
    create_tables()
    db = SessionLocal()
    
    try:
        # Path to test predictions file
        predictions_file = Path("/app/test_predictions.jsonl")
        
        if not predictions_file.exists():
            print(f"❌ File not found: {predictions_file}")
            return
        
        print(f"📁 Loading predictions from: {predictions_file}")
        
        # Read predictions
        predictions_data = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    predictions_data.append(json.loads(line.strip()))
        
        print(f"📊 Found {len(predictions_data)} predictions to import")
        
        # Import tickets and predictions
        imported_count = 0
        for i, pred_data in enumerate(predictions_data):
            try:
                # Create or get ticket
                ticket_id = pred_data.get('ticket_id', f'imported_{i}')
                
                # Check if ticket already exists
                existing_ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
                
                if existing_ticket:
                    db_ticket = existing_ticket
                    print(f"⚠️ Ticket {ticket_id} already exists, skipping...")
                    continue
                else:
                    # Create new ticket
                    db_ticket = Ticket(
                        ticket_id=ticket_id,
                        title=pred_data.get('title', ''),
                        content=pred_data.get('content', ''),
                        created_date=datetime.fromisoformat(pred_data.get('created_date', datetime.now().isoformat())) if pred_data.get('created_date') else None,
                        email_address=pred_data.get('email_address'),
                        raw_data=pred_data
                    )
                    db.add(db_ticket)
                    db.flush()  # Get the ID
                
                # Create prediction
                predicted_queue = pred_data.get('assigned_queue', 'Unknown')
                
                # Generate mock confidence score and probabilities for demo
                # In real scenario, these would come from the model
                confidence_score = np.random.uniform(0.6, 0.95)  # Random confidence for demo
                
                # Mock probabilities for all queues
                queues = ['CTI', 'DFIR::incidents', 'DFIR::phishing', 'OFFSEC::CVD', 'OFFSEC::Pentesting', 'SMS', 'Trash']
                all_probabilities = {}
                for queue in queues:
                    if queue == predicted_queue:
                        all_probabilities[queue] = confidence_score
                    else:
                        remaining_prob = (1 - confidence_score) / (len(queues) - 1)
                        all_probabilities[queue] = remaining_prob
                
                # Determine routing decision based on confidence
                if confidence_score >= 0.85:
                    routing_decision = "auto_route"
                elif confidence_score >= 0.65:
                    routing_decision = "human_verify"
                else:
                    routing_decision = "manual_triage"
                
                db_prediction = Prediction(
                    ticket_id=db_ticket.id,
                    predicted_queue=predicted_queue,
                    confidence_score=confidence_score,
                    all_probabilities=all_probabilities,
                    model_version="hybrid_roberta_v1",
                    processing_time_ms=np.random.randint(100, 500),  # Mock processing time
                    features_used={
                        'text_length': len(pred_data.get('title', '') + " " + pred_data.get('content', '')),
                        'has_urls': 'http' in pred_data.get('content', '').lower(),
                        'has_attachments': any(ext in pred_data.get('content', '').lower() for ext in ['.pdf', '.doc', '.xlsx'])
                    },
                    routing_decision=routing_decision
                )
                db.add(db_prediction)
                
                imported_count += 1
                
                if imported_count % 100 == 0:
                    print(f"📈 Imported {imported_count} predictions...")
                    db.commit()
                
            except Exception as e:
                print(f"❌ Error importing prediction {i}: {e}")
                continue
        
        # Final commit
        db.commit()
        print(f"✅ Successfully imported {imported_count} predictions")
        
        # Print summary statistics
        total_tickets = db.query(Ticket).count()
        total_predictions = db.query(Prediction).count()
        
        print(f"\n📊 Database Summary:")
        print(f"   Total tickets: {total_tickets}")
        print(f"   Total predictions: {total_predictions}")
        
        # Queue distribution
        from sqlalchemy import func
        queue_stats = db.query(
            Prediction.predicted_queue,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.predicted_queue).all()
        
        print(f"\n🏷️ Queue Distribution:")
        for queue, count in queue_stats:
            print(f"   {queue}: {count}")
        
        # Routing decision distribution
        routing_stats = db.query(
            Prediction.routing_decision,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.routing_decision).all()
        
        print(f"\n🔄 Routing Decision Distribution:")
        for routing, count in routing_stats:
            print(f"   {routing}: {count}")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_test_predictions()