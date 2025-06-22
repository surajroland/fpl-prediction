"""Test XGBoost model training."""
import pytest
import pandas as pd
import numpy as np
from fpl.model import XGBoostTrainer

def test_xgboost_trainer():
    """Test XGBoost trainer."""
    config = {
        'model': {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        },
        'position_specific': True,
        'positions': ['FWD', 'MID']
    }
    
    trainer = XGBoostTrainer(config)
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    X_train = pd.DataFrame(np.random.randn(n_samples, 5), 
                          columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(np.random.randn(n_samples))
    positions_train = pd.Series(np.random.choice(['FWD', 'MID'], n_samples))
    
    X_val = pd.DataFrame(np.random.randn(50, 5),
                        columns=[f'feature_{i}' for i in range(5)])
    y_val = pd.Series(np.random.randn(50))
    positions_val = pd.Series(np.random.choice(['FWD', 'MID'], 50))
    
    # Train model
    results = trainer.train(X_train, y_train, X_val, y_val, 
                           positions_train, positions_val)
    
    # Test predictions
    predictions = trainer.predict(X_val, positions_val)
    
    assert len(predictions) == len(X_val)
    assert len(trainer.models) > 0
