"""
Time Series Prediction Module for CERT-EU Ticket Forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sqlalchemy.orm import Session
from database import Prediction, Ticket
from sqlalchemy import func, and_
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    """
    Time series predictor for forecasting ticket volumes by queue
    """
    
    def __init__(self, model_dir: str = "/app/models/time_series"):
        self.model_dir = Path(model_dir)
        # Create parent directory if it doesn't exist
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.feature_scalers = {}
        
    def prepare_time_series_data(self, db: Session, days_back: int = 30) -> pd.DataFrame:
        """
        Prepare time series data from database predictions
        
        Args:
            db: Database session
            days_back: Number of days to look back for training data
            
        Returns:
            DataFrame with daily ticket counts by queue
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Query predictions grouped by date and queue
            query = db.query(
                func.date(Prediction.created_at).label('date'),
                Prediction.predicted_queue.label('queue'),
                func.count(Prediction.id).label('ticket_count')
            ).filter(
                and_(
                    Prediction.created_at >= start_date,
                    Prediction.created_at <= end_date
                )
            ).group_by(
                func.date(Prediction.created_at),
                Prediction.predicted_queue
            ).order_by('date', 'queue')
            
            results = query.all()
            
            if not results:
                logger.warning("No prediction data found for time series analysis")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in results:
                data.append({
                    'date': row.date,
                    'queue': row.queue,
                    'ticket_count': row.ticket_count
                })
            
            df = pd.DataFrame(data)
            
            # Create complete time series with all dates and queues
            all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
            all_queues = df['queue'].unique()
            
            # Create complete grid
            complete_data = []
            for date in all_dates:
                for queue in all_queues:
                    existing = df[(df['date'] == date.date()) & (df['queue'] == queue)]
                    count = existing['ticket_count'].iloc[0] if not existing.empty else 0
                    complete_data.append({
                        'date': date.date(),
                        'queue': queue,
                        'ticket_count': count
                    })
            
            return pd.DataFrame(complete_data)
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.DataFrame()
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for better prediction
        
        Args:
            df: DataFrame with date and ticket_count columns
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Day of month
        df['day_of_month'] = df['date'].dt.day
        
        # Week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Month
        df['month'] = df['date'].dt.month
        
        # Lag features (previous days)
        df['lag_1'] = df.groupby('queue')['ticket_count'].shift(1)
        df['lag_2'] = df.groupby('queue')['ticket_count'].shift(2)
        df['lag_3'] = df.groupby('queue')['ticket_count'].shift(3)
        df['lag_7'] = df.groupby('queue')['ticket_count'].shift(7)  # Weekly pattern
        
        # Rolling averages
        df['ma_3'] = df.groupby('queue')['ticket_count'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        df['ma_7'] = df.groupby('queue')['ticket_count'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        
        # Trend (difference from moving average)
        df['trend'] = df['ticket_count'] - df['ma_7']
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Train time series models for each queue
        
        Args:
            df: Prepared time series DataFrame
            
        Returns:
            Dictionary of trained models by queue
        """
        models = {}
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import warnings
            warnings.filterwarnings('ignore')
            
            # Feature columns (exclude target and date)
            feature_cols = ['day_of_week', 'is_weekend', 'day_of_month', 'week_of_year', 'month',
                          'lag_1', 'lag_2', 'lag_3', 'lag_7', 'ma_3', 'ma_7', 'trend']
            
            for queue in df['queue'].unique():
                queue_data = df[df['queue'] == queue].copy()
                
                if len(queue_data) < 10:  # Need minimum data points
                    logger.warning(f"Insufficient data for queue {queue}")
                    continue
                
                # Prepare features and target
                queue_data = queue_data.dropna()  # Remove rows with NaN from lags
                
                if len(queue_data) < 7:
                    continue
                
                X = queue_data[feature_cols]
                y = queue_data['ticket_count']
                
                # Split data (use last 20% for validation)
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                if len(X_train) < 5:
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val_scaled)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                logger.info(f"Queue {queue} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                
                models[queue] = {
                    'model': model,
                    'scaler': scaler,
                    'mae': mae,
                    'rmse': rmse,
                    'feature_cols': feature_cols
                }
                
                # Save models
                self.save_model(queue, models[queue])
            
            self.models = models
            return models
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def predict_future(self, days_ahead: int = 7, db: Session = None) -> Dict[str, List[Dict]]:
        """
        Predict future ticket volumes for each queue
        
        Args:
            days_ahead: Number of days to predict ahead
            db: Database session for getting recent data
            
        Returns:
            Dictionary with predictions by queue
        """
        predictions = {}
        
        try:
            # Get recent data for feature calculation
            if db:
                recent_data = self.prepare_time_series_data(db, days_back=30)
                if recent_data.empty:
                    return predictions
                
                recent_data = self.add_time_features(recent_data)
            else:
                return predictions
            
            # Load models if not already loaded
            if not self.models:
                self.load_models()
            
            # Predict for each queue
            for queue, model_info in self.models.items():
                if queue not in recent_data['queue'].values:
                    continue
                
                queue_predictions = []
                queue_data = recent_data[recent_data['queue'] == queue].copy()
                
                if queue_data.empty:
                    continue
                
                # Get last known values for features
                last_row = queue_data.iloc[-1].copy()
                
                # Predict each future day
                for day in range(1, days_ahead + 1):
                    future_date = datetime.now().date() + timedelta(days=day)
                    
                    # Prepare features for prediction
                    features = pd.DataFrame([{
                        'day_of_week': future_date.weekday(),
                        'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                        'day_of_month': future_date.day,
                        'week_of_year': future_date.isocalendar().week,
                        'month': future_date.month,
                        'lag_1': last_row['ticket_count'],
                        'lag_2': queue_data.iloc[-2]['ticket_count'] if len(queue_data) > 1 else last_row['ticket_count'],
                        'lag_3': queue_data.iloc[-3]['ticket_count'] if len(queue_data) > 2 else last_row['ticket_count'],
                        'lag_7': queue_data.iloc[-7]['ticket_count'] if len(queue_data) > 6 else last_row['ticket_count'],
                        'ma_3': last_row['ma_3'],
                        'ma_7': last_row['ma_7'],
                        'trend': last_row['trend']
                    }])
                    
                    # Scale features
                    X_scaled = model_info['scaler'].transform(features)
                    
                    # Predict
                    predicted_count = model_info['model'].predict(X_scaled)[0]
                    predicted_count = max(0, int(round(predicted_count)))  # Ensure non-negative integer
                    
                    queue_predictions.append({
                        'date': future_date.isoformat(),
                        'predicted_tickets': predicted_count,
                        'confidence': self._calculate_confidence(model_info)
                    })
                    
                    # Update last_row for next prediction (simple approach)
                    last_row['ticket_count'] = predicted_count
                
                predictions[queue] = queue_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
            return {}
    
    def _calculate_confidence(self, model_info: Dict) -> float:
        """
        Calculate confidence score based on model performance
        
        Args:
            model_info: Model information including MAE and RMSE
            
        Returns:
            Confidence score between 0 and 1
        """
        mae = model_info.get('mae', 1.0)
        rmse = model_info.get('rmse', 1.0)
        
        # Convert error metrics to confidence (lower error = higher confidence)
        avg_error = (mae + rmse) / 2
        confidence = max(0.1, min(0.9, 1.0 - (avg_error / 10.0)))  # Scale based on expected range
        
        return round(confidence, 2)
    
    def save_model(self, queue: str, model_info: Dict):
        """Save model to disk"""
        try:
            model_path = self.model_dir / f"{queue}_model.pkl"
            joblib.dump(model_info, model_path)
            logger.info(f"Saved model for queue {queue}")
        except Exception as e:
            logger.error(f"Error saving model for {queue}: {e}")
    
    def load_models(self):
        """Load all saved models"""
        try:
            for model_file in self.model_dir.glob("*_model.pkl"):
                queue = model_file.stem.replace("_model", "")
                model_info = joblib.load(model_file)
                self.models[queue] = model_info
                logger.info(f"Loaded model for queue {queue}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_prediction_summary(self, predictions: Dict[str, List[Dict]]) -> Dict:
        """
        Generate summary statistics for predictions
        
        Args:
            predictions: Predictions dictionary from predict_future
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_predicted_tickets': 0,
            'queue_summary': {},
            'peak_days': {},
            'trend_analysis': {}
        }
        
        for queue, queue_predictions in predictions.items():
            total_tickets = sum(pred['predicted_tickets'] for pred in queue_predictions)
            avg_tickets = total_tickets / len(queue_predictions) if queue_predictions else 0
            avg_confidence = sum(pred['confidence'] for pred in queue_predictions) / len(queue_predictions) if queue_predictions else 0
            
            summary['queue_summary'][queue] = {
                'total_predicted': total_tickets,
                'average_daily': round(avg_tickets, 1),
                'confidence': round(avg_confidence, 2)
            }
            
            summary['total_predicted_tickets'] += total_tickets
            
            # Find peak day
            if queue_predictions:
                peak_day = max(queue_predictions, key=lambda x: x['predicted_tickets'])
                summary['peak_days'][queue] = {
                    'date': peak_day['date'],
                    'tickets': peak_day['predicted_tickets']
                }
        
        return summary
