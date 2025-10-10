#!/usr/bin/env python3
"""
Main training script for CERT-EU ticket classification with Cross-Validation support.
Usage: 
    # Regular training with preprocessing
    python m_main.py --data_path data/train.jsonl --epochs 3 --batch_size 16

    # Regular training without preprocessing (data already processed)
    python m_main.py --data_path data/train.jsonl --skip_preprocessing --epochs 3 --batch_size 16

    # Cross-validation mode with preprocessing
    python m_main.py --data_path data/train.jsonl --cross_validate --cv_folds 5

    # Cross-validation with preprocessed data
    python train_model.py \
        --data_path data/preprocessed_train.jsonl \
        --skip_preprocessing \
        --cross_validate \
        --cv_folds 5 \
        --epochs 3 \
        --batch_size 16 \
        --use_attention_fusion \
        --use_class_weights \
        --confidence_high 0.95 \
        --confidence_low 0.7
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import json
from datetime import datetime
import pickle
import torch

# Import your classes (adjust paths as needed)
from data_processor import DataProcessor  # Your preprocessing class
from model_trainer import HybridTransformerModel, ModelEvaluator  # Enhanced model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CERT-EU ticket classifier')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (JSONL file)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained model')
    
    # NEW: Preprocessing control
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing step (assumes data is already preprocessed)')
    parser.add_argument('--preprocessed_features_path', type=str, default=None,
                       help='Path to preprocessed features file (optional, auto-generated if not provided)')
    parser.add_argument('--save_preprocessed', action='store_true', default=True,
                       help='Save preprocessed features for future use (default: True)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--use_attention_fusion', action='store_true', default=True,
                       help='Use attention-based fusion (default: True)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (ignored in CV mode)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Class balancing
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data (default: True)')
    
    # Hardware arguments
    parser.add_argument('--no_gpu', action='store_true',
                       help='Force CPU usage even if GPU available')
    
    # Confidence routing
    parser.add_argument('--confidence_high', type=float, default=0.9,
                       help='High confidence threshold for auto-routing')
    parser.add_argument('--confidence_low', type=float, default=0.6,
                       help='Low confidence threshold for manual triage')
    
    # Cross-validation arguments
    parser.add_argument('--cross_validate', action='store_true',
                       help='Use cross-validation instead of train/test split')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--cv_stratified', action='store_true', default=True,
                       help='Use stratified cross-validation (recommended for imbalanced data)')
    parser.add_argument('--save_cv_models', action='store_true',
                       help='Save models from each CV fold (requires more storage)')
    
    # early-stopping arguments
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Enable early stopping (default: True)')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of epochs to wait for improvement before stopping (default: 2)')
    parser.add_argument('--monitor_metric', type=str, default='val_loss', choices=['val_loss', 'val_f1', 'val_acc'],
                        help='Metric to monitor for early stopping (val_loss, val_f1, val_acc)')

    
    return parser.parse_args()

def load_preprocessed_data(preprocessed_path):
    """Load preprocessed features from file."""
    print(f"Loading preprocessed data from {preprocessed_path}...")
    
    with open(preprocessed_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    text_features = preprocessed_data['text_features']
    numerical_features = preprocessed_data['numerical_features']
    labels = preprocessed_data['labels']
    class_names = preprocessed_data['class_names']
    processor = preprocessed_data['processor']
    
    print(f"Loaded preprocessed data:")
    print(f"  Text features: {text_features.shape}")
    print(f"  Numerical features: {numerical_features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {len(class_names)}")
    
    return text_features, numerical_features, labels, class_names, processor

def save_preprocessed_data(text_features, numerical_features, labels, class_names, processor, save_path):
    """Save preprocessed features to file."""
    print(f"Saving preprocessed data to {save_path}...")
    
    preprocessed_data = {
        'text_features': text_features,
        'numerical_features': numerical_features,
        'labels': labels,
        'class_names': class_names,
        'processor': processor,
        'timestamp': datetime.now().isoformat(),
        'feature_shapes': {
            'text': text_features.shape,
            'numerical': numerical_features.shape,
            'labels': labels.shape
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"Preprocessed data saved successfully!")

def load_and_preprocess_data(data_path, processor, skip_preprocessing=False, preprocessed_path=None):
    """Load and optionally preprocess the data."""
    
    if skip_preprocessing:
        # Load preprocessed data
        if preprocessed_path and os.path.exists(preprocessed_path):
            return load_preprocessed_data(preprocessed_path)
        else:
            # Try to find preprocessed file based on data_path
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            auto_preprocessed_path = os.path.join(os.path.dirname(data_path), f"{base_name}_preprocessed.pkl")
            
            if os.path.exists(auto_preprocessed_path):
                print(f"Found preprocessed file: {auto_preprocessed_path}")
                return load_preprocessed_data(auto_preprocessed_path)
            else:
                print(f"⚠️  Warning: --skip_preprocessing specified but no preprocessed file found!")
                print(f"    Looked for: {auto_preprocessed_path}")
                print(f"    Falling back to preprocessing raw data...")
                skip_preprocessing = False
    
    if not skip_preprocessing:
        # Perform preprocessing
        print(f"Loading raw data from {data_path}...")
        
        # Load data
        df = processor.load_data(data_path)
        print(f"Loaded {len(df)} samples")
        
        # Show class distribution
        print("\nClass distribution:")
        class_counts = df['assigned_queue'].value_counts()
        total_samples = len(df)
        
        print("-" * 60)
        print(f"{'Queue':<25} {'Count':<8} {'Percentage':<12} {'Imbalance'}")
        print("-" * 60)
        
        max_count = class_counts.max()
        for queue, count in class_counts.items():
            percentage = count/total_samples*100
            imbalance_ratio = max_count/count
            print(f"{queue:<25} {count:<8} {percentage:<12.1f}% {imbalance_ratio:.1f}x")
        
        # Check for severe imbalance
        imbalance_ratio = max_count / class_counts.min()
        if imbalance_ratio > 10:
            print(f"\n⚠️  HIGH IMBALANCE DETECTED: {imbalance_ratio:.1f}:1 ratio")
            print("   → Class weights will be crucial for good performance")
            print("   → Cross-validation highly recommended for robust evaluation")
        elif imbalance_ratio > 3:
            print(f"\n⚠️  MODERATE IMBALANCE: {imbalance_ratio:.1f}:1 ratio")
            print("   → Class weights recommended")
            print("   → Cross-validation recommended")
        else:
            print(f"\n✅ BALANCED DATASET: {imbalance_ratio:.1f}:1 ratio")
        
        print("-" * 60)
        
        # Extract features
        print("\nExtracting features...")
        df_features = processor.extract_features(df)
        
        # Prepare features for model
        text_features, numerical_features, labels = processor.prepare_features(df_features, is_training=True)
        class_names = list(processor.label_encoder.classes_)
        
        return text_features, numerical_features, labels, class_names, processor

def run_cross_validation(args, text_features, numerical_features, labels, class_names, class_weights):
    """Run cross-validation training and evaluation."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION MODE")
    print("="*60)
    print(f"Folds: {args.cv_folds}")
    print(f"Stratified: {args.cv_stratified}")
    print(f"Save models: {args.save_cv_models}")
    print("="*60)
    
    # Initialize model for CV
    model = HybridTransformerModel(
        model_name=args.model_name,
        num_labels=len(class_names),
        num_numerical_features=numerical_features.shape[1],
        use_gpu=not args.no_gpu,
        use_attention_fusion=args.use_attention_fusion
    )
    
    # Run cross-validation
    cv_results = model.cross_validate(
        texts=text_features,
        numerical_features=numerical_features,
        labels=labels,
        cv_folds=args.cv_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        stratified=args.cv_stratified,
        random_state=args.random_state
    )
    
    # Analyze CV results
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    analysis = model.analyze_cv_results(cv_results, class_names)
    
    # Confidence-based routing analysis across all folds
    print("\n" + "-"*60)
    print("CONFIDENCE-BASED ROUTING ANALYSIS (ALL FOLDS)")
    print("-"*60)
    
    # Combine all probabilities from CV
    all_probabilities = np.concatenate(cv_results['fold_probabilities'])
    all_true_labels = np.concatenate(cv_results['fold_true_labels'])
    
    routing_results = model.confidence_based_routing(
        all_probabilities,
        threshold_high=args.confidence_high,
        threshold_low=args.confidence_low
    )
    
    # Show routing efficiency by queue type
    print("\nRouting efficiency by queue type (cross-validation):")
    for i, queue_name in enumerate(class_names):
        queue_mask = all_true_labels == i
        if np.sum(queue_mask) > 0:
            queue_routing = routing_results['routing_decisions']
            
            auto_route_count = np.sum(queue_routing['auto_route'][queue_mask])
            human_verify_count = np.sum(queue_routing['human_verify'][queue_mask])
            manual_triage_count = np.sum(queue_routing['manual_triage'][queue_mask])
            total_queue = np.sum(queue_mask)
            
            print(f"\n{queue_name}:")
            print(f"  Auto-route: {auto_route_count}/{total_queue} ({auto_route_count/total_queue*100:.1f}%)")
            print(f"  Human verify: {human_verify_count}/{total_queue} ({human_verify_count/total_queue*100:.1f}%)")
            print(f"  Manual triage: {manual_triage_count}/{total_queue} ({manual_triage_count/total_queue*100:.1f}%)")
            
            # Calculate accuracy for auto-routed tickets of this queue
            all_predictions = np.concatenate(cv_results['fold_predictions'])
            auto_mask = queue_mask & queue_routing['auto_route']
            if np.sum(auto_mask) > 0:
                auto_accuracy = np.mean(all_predictions[auto_mask] == all_true_labels[auto_mask])
                print(f"  Auto-route accuracy: {auto_accuracy:.3f}")
    
    # Save CV results
    cv_output_dir = os.path.join(args.output_dir, f'cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # Save detailed CV results
    cv_results_path = os.path.join(cv_output_dir, 'cv_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    cv_results_serializable = {
        'fold_scores': cv_results['fold_scores'],
        'fold_f1_macro': cv_results['fold_f1_macro'],
        'fold_f1_weighted': cv_results['fold_f1_weighted'],
        'mean_accuracy': cv_results['mean_accuracy'],
        'std_accuracy': cv_results['std_accuracy'],
        'mean_f1_macro': cv_results['mean_f1_macro'],
        'std_f1_macro': cv_results['std_f1_macro'],
        'mean_f1_weighted': cv_results['mean_f1_weighted'],
        'std_f1_weighted': cv_results['std_f1_weighted'],
        'class_names': class_names,
        'training_args': vars(args),
        'routing_statistics': routing_results['statistics']
    }
    
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results_serializable, f, indent=2)
    
    print(f"\nCV results saved to: {cv_results_path}")
    
    # Save CV analysis report
    report_path = os.path.join(cv_output_dir, 'cv_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("CERT-EU Cross-Validation Analysis Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Preprocessing: {'Skipped' if args.skip_preprocessing else 'Performed'}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"CV Folds: {args.cv_folds}\n")
        f.write(f"Epochs per fold: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n\n")
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}\n")
        f.write(f"F1-Macro: {cv_results['mean_f1_macro']:.4f} ± {cv_results['std_f1_macro']:.4f}\n")
        f.write(f"F1-Weighted: {cv_results['mean_f1_weighted']:.4f} ± {cv_results['std_f1_weighted']:.4f}\n\n")
        
        f.write("CONFIDENCE ROUTING\n")
        f.write("-" * 30 + "\n")
        f.write(f"High threshold: {args.confidence_high}\n")
        f.write(f"Low threshold: {args.confidence_low}\n")
        f.write(f"Auto-route: {routing_results['statistics']['auto_route_pct']:.1f}%\n")
        f.write(f"Human verify: {routing_results['statistics']['human_verify_pct']:.1f}%\n")
        f.write(f"Manual triage: {routing_results['statistics']['manual_triage_pct']:.1f}%\n\n")
        
        f.write("DEPLOYMENT RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        # Generate recommendations based on results
        if cv_results['std_accuracy'] < 0.02:
            f.write("✅ Model shows good stability across folds\n")
        else:
            f.write("⚠️  High variance between folds - consider more data or regularization\n")
        
        if cv_results['mean_f1_macro'] > 0.8:
            f.write("✅ Excellent performance for deployment\n")
        elif cv_results['mean_f1_macro'] > 0.7:
            f.write("✅ Good performance, ready for deployment with monitoring\n")
        else:
            f.write("⚠️  Consider more training or data before deployment\n")
        
        if routing_results['statistics']['auto_route_pct'] > 50:
            f.write("✅ Good automation potential\n")
        else:
            f.write("⚠️  Low automation rate - consider adjusting confidence thresholds\n")
    
    print(f"CV analysis report saved to: {report_path}")

    cms = cv_results.get('fold_confusion_matrices', [])
    if cms:
        # Pad matrices if needed (for rare cases of missing classes in a fold)
        max_shape = np.array([cm.shape for cm in cms]).max(axis=0)
        padded_cms = []
        for cm in cms:
            padded = np.zeros(max_shape, dtype=float)
            padded[:cm.shape[0], :cm.shape[1]] = cm
            padded_cms.append(padded)
        cms = np.stack(padded_cms, axis=0)
        mean_cm = np.mean(cms, axis=0)
        mean_cm_rounded = np.round(mean_cm, 2)
        mean_cm_csv_path = os.path.join(cv_output_dir, 'mean_confusion_matrix.csv')
        mean_cm_df = pd.DataFrame(mean_cm_rounded, index=class_names, columns=class_names)
        mean_cm_df.to_csv(mean_cm_csv_path)
        print(f"\nMean confusion matrix saved to: {mean_cm_csv_path}")
        print("\nMean Confusion Matrix (rounded):")
        print(mean_cm_df)
    
    # Optionally save models from each fold
    if args.save_cv_models:
        print("\nNote: Individual fold models were not saved in this run.")
        print("To save fold models, implement model saving within the cross_validate method.")
    
    return cv_results, analysis, routing_results

def run_regular_training(args, text_features, numerical_features, labels, class_names, class_weights):
    """Run regular train/test split training."""
    print("\n" + "="*60)
    print("REGULAR TRAINING MODE")
    print("="*60)
    
    # Split data
    print(f"Splitting data (test_size={args.test_size})...")
    (X_text_train, X_text_val, 
     X_num_train, X_num_val, 
     y_train, y_val) = train_test_split(
        text_features, numerical_features, labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels
    )
    
    print(f"Train set: {len(X_text_train)} samples")
    print(f"Validation set: {len(X_text_val)} samples")
    
    # Initialize model
    print(f"\nInitializing hybrid model...")
    model = HybridTransformerModel(
        model_name=args.model_name,
        num_labels=len(class_names),
        num_numerical_features=numerical_features.shape[1],
        use_gpu=not args.no_gpu,
        use_attention_fusion=args.use_attention_fusion
    )
    
    # Train model
    print(f"\nStarting training...")
    model.train(
        train_texts=X_text_train,
        train_numerical=X_num_train,
        train_labels=y_train,
        val_texts=X_text_val,
        val_numerical=X_num_val,
        val_labels=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        early_stopping= args.early_stopping,
        patience=args.patience,
        monitor_metric = args.monitor_metric
    )
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    predictions = model.predict(X_text_val, X_num_val)
    probabilities = model.predict_proba(X_text_val, X_num_val)
    
    # Standard evaluation
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_val, predictions, class_names)
    
    # CERT-specific evaluation for critical queues
    print("\n" + "-"*60)
    evaluator.cert_specific_metrics(y_val, predictions, class_names)
    
    # Confidence analysis
    print("\n" + "-"*60)
    evaluator.analyze_predictions(y_val, predictions, probabilities, class_names)
    
    # Confidence-based routing demonstration
    print("\n" + "-"*60)
    print("CONFIDENCE-BASED ROUTING ANALYSIS")
    print("-"*60)
    routing_results = model.confidence_based_routing(
        probabilities, 
        threshold_high=args.confidence_high, 
        threshold_low=args.confidence_low
    )
    
    # Show routing efficiency for different queue types
    print("\nRouting efficiency by queue type:")
    for i, queue_name in enumerate(class_names):
        queue_mask = y_val == i
        if np.sum(queue_mask) > 0:
            queue_routing = routing_results['routing_decisions']
            
            auto_route_count = np.sum(queue_routing['auto_route'][queue_mask])
            human_verify_count = np.sum(queue_routing['human_verify'][queue_mask])
            manual_triage_count = np.sum(queue_routing['manual_triage'][queue_mask])
            total_queue = np.sum(queue_mask)
            
            print(f"\n{queue_name}:")
            print(f"  Auto-route: {auto_route_count}/{total_queue} ({auto_route_count/total_queue*100:.1f}%)")
            print(f"  Human verify: {human_verify_count}/{total_queue} ({human_verify_count/total_queue*100:.1f}%)")
            print(f"  Manual triage: {manual_triage_count}/{total_queue} ({manual_triage_count/total_queue*100:.1f}%)")
            
            # Calculate accuracy for auto-routed tickets of this queue
            auto_mask = queue_mask & queue_routing['auto_route']
            if np.sum(auto_mask) > 0:
                auto_accuracy = np.mean(predictions[auto_mask] == y_val[auto_mask])
                print(f"  Auto-route accuracy: {auto_accuracy:.3f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'hybrid_roberta_model')
    print(f"\n" + "="*60)
    print(f"SAVING MODEL")
    print("="*60)
    print(f"Model path: {model_path}")
    model.save_model(model_path)
    
    # Save routing results for analysis
    routing_path = os.path.join(args.output_dir, 'routing_analysis.txt')
    with open(routing_path, 'w') as f:
        f.write("CERT-EU Confidence-Based Routing Analysis\n")
        f.write("="*50 + "\n\n")
        f.write(f"Thresholds: High={args.confidence_high}, Low={args.confidence_low}\n")
        f.write(f"Total validation samples: {len(y_val)}\n\n")
        
        stats = routing_results['statistics']
        f.write(f"Auto-route: {stats['auto_route_pct']:.1f}%\n")
        f.write(f"Human verify: {stats['human_verify_pct']:.1f}%\n")
        f.write(f"Manual triage: {stats['manual_triage_pct']:.1f}%\n\n")
        
        f.write("This suggests that:\n")
        f.write(f"- {stats['auto_route_pct']:.1f}% of tickets can be automatically routed\n")
        f.write(f"- {stats['human_verify_pct']:.1f}% need human verification\n")
        f.write(f"- {stats['manual_triage_pct']:.1f}% require full manual analysis\n")
    
    print(f"Routing analysis: {routing_path}")
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Final validation F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Model ready for deployment with confidence-based routing!")
    print("="*60)
    
    return model, metrics, routing_results

def main():
    """Main training function with cross-validation support."""
    args = parse_arguments()
    
    print("="*60)
    print("CERT-EU TICKET CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Preprocessing: {'SKIP' if args.skip_preprocessing else 'PERFORM'}")
    if args.skip_preprocessing and args.preprocessed_features_path:
        print(f"Preprocessed features: {args.preprocessed_features_path}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Cross-Validation' if args.cross_validate else 'Regular Training'}")
    if args.cross_validate:
        print(f"CV folds: {args.cv_folds}")
        print(f"Stratified: {args.cv_stratified}")
    print(f"Attention fusion: {args.use_attention_fusion}")
    print(f"Class weights: {args.use_class_weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence thresholds: High={args.confidence_high}, Low={args.confidence_low}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = DataProcessor()
    
    # Load and optionally preprocess data
    text_features, numerical_features, labels, class_names, processor = load_and_preprocess_data(
        args.data_path, processor, args.skip_preprocessing, args.preprocessed_features_path
    )
    
    # Save preprocessed data for future use (if preprocessing was performed)
    if not args.skip_preprocessing and args.save_preprocessed:
        base_name = os.path.splitext(os.path.basename(args.data_path))[0]
        preprocessed_path = os.path.join(args.output_dir, f"{base_name}_preprocessed.pkl")
        save_preprocessed_data(text_features, numerical_features, labels, class_names, processor, preprocessed_path)
    
    print(f"\nFeature shapes:")
    print(f"Text features: {text_features.shape}")
    print(f"Numerical features: {numerical_features.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Classes: {list(class_names)}")
    
    # Calculate class weights
    class_weights = None
    if args.use_class_weights:
        print(f"\nCalculating class weights for imbalanced data...")
        class_weights = processor.get_class_weights(labels)
        print("Class weights:")
        for i, (class_name, weight) in enumerate(zip(class_names, class_weights.values())):
            print(f"  {class_name:<25}: {weight:.3f}")
    
     # Learning Rate Finder (optional)
    if not args.cross_validate:  # Only for regular training
        print(f"\n" + "="*60)
        print("LEARNING RATE FINDER")
        print("="*60)
        
        # Initialize model for LR finding
        lr_finder_model = HybridTransformerModel(
            model_name=args.model_name,
            num_labels=len(class_names),
            num_numerical_features=numerical_features.shape[1],
            use_gpu=not args.no_gpu,
            use_attention_fusion=args.use_attention_fusion
        )
        
        # Run LR finder
        lrs, losses = lr_finder_model.lr_find(
            text_features, numerical_features, labels,
            batch_size=args.batch_size
        )
        
        # Find optimal LR (steepest decline)
        gradients = np.gradient(losses)
        optimal_idx = np.argmin(gradients)
        suggested_lr = lrs[optimal_idx]
        
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        print(f"Original learning rate: {args.learning_rate:.2e}")
        
        # Optionally update the learning rate
        response = input("Use suggested LR? (y/n): ")
        if response.lower() == 'y':
            args.learning_rate = suggested_lr
            print(f"Updated learning rate to: {args.learning_rate:.2e}")
    
        del lr_finder_model  # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Choose training mode
    if args.cross_validate:
        # Run cross-validation
        cv_results, analysis, routing_results = run_cross_validation(
            args, text_features, numerical_features, labels, class_names, class_weights
        )
        
        print(f"\n" + "="*60)
        print("CROSS-VALIDATION COMPLETED!")
        print("="*60)
        print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Mean F1 (macro): {cv_results['mean_f1_macro']:.4f} ± {cv_results['std_f1_macro']:.4f}")
        print("Results saved to output directory.")
        print("Ready for production deployment!")
        print("="*60)
        
    else:
        # Run regular training
        model, metrics, routing_results = run_regular_training(
            args, text_features, numerical_features, labels, class_names, class_weights
        )
    
    # Save processor and class names (for both modes)
    processor_path = os.path.join(args.output_dir, 'data_processor.pkl')
    print(f"Processor path: {processor_path}")
    processor.save_processor(processor_path)
    
    class_names_path = os.path.join(args.output_dir, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names: {class_names_path}")

if __name__ == "__main__":
    main()