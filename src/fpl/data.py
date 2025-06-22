"""FPL data loading and preprocessing."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class FPLDataConfig:
    features: List[str]
    rolling_window: int = 5
    min_games: int = 3
    top_percent: int = 20

class FPLDataLoader:
    """Load and preprocess FPL data."""
    
    def __init__(self, config: FPLDataConfig):
        self.config = config
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load FPL data from file."""
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
        
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features."""
        logger.info(f"Creating {self.config.rolling_window}-game rolling features")
        
        features = []
        for feature in self.config.features:
            rolling_col = f"{feature}_rolling_{self.config.rolling_window}"
            df[rolling_col] = (df.groupby('player_id')[feature]
                              .rolling(self.config.rolling_window, min_periods=1)
                              .mean().reset_index(0, drop=True))
            features.append(rolling_col)
            
        return df, features
        
    def filter_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to top players by selection percentage."""
        logger.info(f"Filtering to top {self.config.top_percent}% of players")
        
        # Group by position and filter
        filtered_dfs = []
        for position in df['position'].unique():
            pos_df = df[df['position'] == position]
            threshold = pos_df['selected_by_percent'].quantile(1 - self.config.top_percent/100)
            filtered_dfs.append(pos_df[pos_df['selected_by_percent'] >= threshold])
            
        return pd.concat(filtered_dfs, ignore_index=True)
