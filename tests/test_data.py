"""Test data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from fpl.data import FPLDataLoader, FPLDataConfig

def test_fpl_data_loader():
    """Test FPL data loader."""
    config = FPLDataConfig(
        features=['goals_scored', 'assists'],
        rolling_window=3
    )
    loader = FPLDataLoader(config)
    
    # Create test data
    data = pd.DataFrame({
        'player_id': [1, 1, 1, 2, 2, 2],
        'gameweek': [1, 2, 3, 1, 2, 3],
        'goals_scored': [1, 0, 2, 0, 1, 0],
        'assists': [0, 1, 1, 2, 0, 1],
        'position': ['FWD', 'FWD', 'FWD', 'MID', 'MID', 'MID'],
        'selected_by_percent': [10, 10, 10, 20, 20, 20]
    })
    
    # Test rolling features
    result_df, feature_cols = loader.create_rolling_features(data)
    
    assert 'goals_scored_rolling_3' in result_df.columns
    assert 'assists_rolling_3' in result_df.columns
    assert len(feature_cols) == 2
