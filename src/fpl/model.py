"""XGBoost model training."""
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import joblib

class XGBoostTrainer:
    """XGBoost model trainer with position-specific models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_names: Optional[List[str]] = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series,
              positions_train: pd.Series, positions_val: pd.Series) -> Dict[str, Any]:
        """Train position-specific XGBoost models."""
        
        self.feature_names = X_train.columns.tolist()
        results = {}
        
        if self.config.get('position_specific', True):
            # Train separate model for each position
            for position in self.config['positions']:
                logger.info(f"Training model for position: {position}")
                
                # Filter data for this position
                train_mask = positions_train == position
                val_mask = positions_val == position
                
                if train_mask.sum() == 0:
                    logger.warning(f"No training data for position {position}")
                    continue
                    
                X_pos_train = X_train[train_mask]
                y_pos_train = y_train[train_mask]
                X_pos_val = X_val[val_mask] if val_mask.sum() > 0 else None
                y_pos_val = y_val[val_mask] if val_mask.sum() > 0 else None
                
                # Create and train model
                model = xgb.XGBRegressor(**self.config['model'])
                
                eval_set = [(X_pos_train, y_pos_train)]
                if X_pos_val is not None:
                    eval_set.append((X_pos_val, y_pos_val))
                    
                model.fit(
                    X_pos_train, y_pos_train,
                    eval_set=eval_set,
                    early_stopping_rounds=self.config.get('early_stopping_rounds', 50),
                    verbose=False
                )
                
                self.models[position] = model
                
                # Evaluate
                if X_pos_val is not None:
                    y_pred = model.predict(X_pos_val)
                    metrics = self._calculate_metrics(y_pos_val, y_pred)
                    results[f"{position}_metrics"] = metrics
                    logger.info(f"{position} validation RMSE: {metrics['rmse']:.4f}")
                    
        return results
        
    def predict(self, X: pd.DataFrame, positions: pd.Series) -> np.ndarray:
        """Make predictions using position-specific models."""
        predictions = np.zeros(len(X))
        
        for position in positions.unique():
            if position in self.models:
                mask = positions == position
                predictions[mask] = self.models[position].predict(X[mask])
            else:
                logger.warning(f"No model found for position {position}")
                
        return predictions
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
    def save_models(self, path: str):
        """Save trained models."""
        joblib.dump({
            'models': self.models,
            'feature_names': self.feature_names,
            'config': self.config
        }, path)
        logger.info(f"Models saved to {path}")
        
    @classmethod
    def load_models(cls, path: str) -> 'XGBoostTrainer':
        """Load trained models."""
        data = joblib.load(path)
        trainer = cls(data['config'])
        trainer.models = data['models']
        trainer.feature_names = data['feature_names']
        return trainer
