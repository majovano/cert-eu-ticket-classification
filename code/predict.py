#!/usr/bin/env python3
"""
Generate predictions for CERT-EU test dataset.

Usage: 
    python m_predict.py --test_data data/test_dataset.jsonl --model_dir ./models --output data/test_predictions.jsonl
"""

import argparse
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

# Import your classes
from data_processor import DataProcessor
from model_trainer import HybridTransformerModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate predictions for CERT-EU test dataset')
    
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset (JSONL file)')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--output', type=str, default='predictions.jsonl',
                       help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for prediction')
    parser.add_argument('--conf_threshold', type=float, default=0.8,
                       help='Threshold for low-confidence outputs')
    
    return parser.parse_args()

def load_test_data(test_data_path):
    """Load test dataset."""
    print(f"Loading test data from {test_data_path}...")
    
    test_records = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line.strip()))
    
    print(f"Loaded {len(test_records)} test samples")
    return test_records, pd.DataFrame(test_records)

def load_model_and_processor(model_dir):
    """Load trained model and data processor with error handling."""
    print(f"Loading model from {model_dir}...")
    
    # Check for required files
    required_files = {
        'processor': 'data_processor.pkl',
        'class_names': 'class_names.txt',
        'model': ['hybrid_roberta_model', 'basic_model']  # Try both names
    }
    
    # Load processor
    processor_path = os.path.join(model_dir, required_files['processor'])
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Data processor not found: {processor_path}")
    
    processor = DataProcessor()
    processor.load_processor(processor_path)
    
    # Load class names
    class_names_path = os.path.join(model_dir, required_files['class_names'])
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Try to find model directory (support both naming conventions)
    model_path = None
    for model_name in required_files['model']:
        potential_path = os.path.join(model_dir, model_name)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model not found. Tried: {required_files['model']}")
    
    # Initialize model
    print(f"Initializing model with {len(class_names)} classes...")
    model = HybridTransformerModel(
        num_labels=len(class_names),
        num_numerical_features=17,  # Your standard feature count
        use_gpu=True
    )
    
    try:
        model.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print(f"Classes: {class_names}")
    return model, processor, class_names

def generate_predictions(model, processor, test_df, class_names, batch_size=16):
    """Generate predictions with error handling."""
    print("Processing test data...")
    
    try:
        # Extract features
        test_features = processor.extract_features(test_df)
        
        # Prepare features (no labels for test data)
        text_features, numerical_features, _ = processor.prepare_features(test_features, is_training=False)
        
        print(f"Feature shapes: Text={len(text_features)}, Numerical={numerical_features.shape}")
        
        # Generate predictions
        print("Generating predictions...")
        predictions = model.predict(text_features, numerical_features, batch_size=batch_size)
        probabilities = model.predict_proba(text_features, numerical_features, batch_size=batch_size)
        
        # Convert to readable format
        predicted_queues = [class_names[pred] for pred in predictions]
        confidence_scores = np.max(probabilities, axis=1)
        
        print(f"Generated predictions for {len(predictions)} samples")
        
        # Show quick stats
        print(f"\nTop predicted queues:")
        unique, counts = np.unique(predictions, return_counts=True)
        for idx, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {class_names[idx]}: {count} ({count/len(predictions)*100:.1f}%)")
        
        print(f"\nConfidence: mean={np.mean(confidence_scores):.3f}, "
              f"min={np.min(confidence_scores):.3f}, max={np.max(confidence_scores):.3f}")
        
        return predicted_queues, confidence_scores
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def save_predictions(original_records, predicted_queues, confidence_scores, output_path):
    """Save predictions in JSONL format."""
    print(f"Saving {len(predicted_queues)} predictions to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record, predicted_queue, confidence in zip(original_records, predicted_queues, confidence_scores):
            prediction_record = record.copy()
            prediction_record['assigned_queue'] = predicted_queue
            f.write(json.dumps(prediction_record) + '\n')
    
    print(f"Predictions saved to: {output_path}")
    
    # Show first example
    with open(output_path, 'r') as f:
        first_record = json.loads(f.readline())
        print(f"Example: ticket_id='{first_record.get('ticket_id', 'N/A')}' → "
              f"assigned_queue='{first_record.get('assigned_queue', 'N/A')}'")

def save_low_confidence(
    original_records, predicted_queues, confidence_scores, output_path, threshold=0.8
):
    """Save only low-confidence predictions in a separate JSONL file."""
    low_conf_records = []
    for record, pred_queue, conf in zip(original_records, predicted_queues, confidence_scores):
        if conf < threshold:
            low_conf_records.append({
                'ticket_id': record.get('ticket_id', None),
                'title': record.get('title', None),
                'content': record.get('content', None),
                'assigned_queue': pred_queue,
                'confidence': float(conf)
            })
    print(f"Found {len(low_conf_records)} low-confidence predictions (conf < {threshold})")
    if len(low_conf_records) > 0:
        low_conf_path = output_path.replace('.jsonl', f'_low_conf.jsonl')
        with open(low_conf_path, 'w', encoding='utf-8') as f:
            for rec in low_conf_records:
                f.write(json.dumps(rec) + '\n')
        print(f"Low-confidence predictions saved to: {low_conf_path}")
    else:
        print("No low-confidence predictions found.")
    
    if low_conf_records:
        class_counts = Counter([rec['assigned_queue'] for rec in low_conf_records])
        print("\nLow-confidence distribution by queue:")
        for queue, count in class_counts.most_common():
            print(f"  {queue}: {count}")

def main():
    """Main prediction function."""
    args = parse_arguments()
    
    print("="*50)
    print("CERT-EU PREDICTION")
    print("="*50)
    print(f"Test data: {args.test_data}")
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output}")
    print("="*50)
    
    try:
        # Load everything
        original_records, test_df = load_test_data(args.test_data)
        model, processor, class_names = load_model_and_processor(args.model_dir)
        
        # Generate and save predictions
        predicted_queues, confidence_scores = generate_predictions(
            model, processor, test_df, class_names, args.batch_size
        )
        save_predictions(original_records, predicted_queues, confidence_scores, args.output)
        save_low_confidence(original_records, predicted_queues, confidence_scores, args.output, threshold=args.conf_threshold)
        
        print("\n" + "="*50)
        print("PREDICTION COMPLETED!")
        print("="*50)
        
        return 0
        
    except Exception as e:
        print(f"\nPREDICTION FAILED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
