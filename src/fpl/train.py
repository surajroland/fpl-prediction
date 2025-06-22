"""Training script with Hydra configuration."""
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.xgboost
from loguru import logger
import sys

from fpl.data import FPLDataLoader, FPLDataConfig
from fpl.model import XGBoostTrainer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)
    logger.add(f"{cfg.logs_dir}/training.log", level=cfg.log_level)
    
    logger.info("Starting FPL XGBoost training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup MLflow
    if cfg.training.mlflow.enabled:
        mlflow.set_experiment(cfg.training.mlflow.experiment_name)
        mlflow.start_run()
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    
    try:
        # Load and preprocess data
        data_config = FPLDataConfig(**cfg.data.dataset)
        loader = FPLDataLoader(data_config)
        
        # TODO: Load actual FPL data
        # For now, create dummy data
        logger.info("Creating dummy data for demonstration")
        df = create_dummy_data()
        
        # Preprocess
        df, feature_cols = loader.create_rolling_features(df)
        df = loader.filter_players(df)
        
        # Split data
        train_df, val_df, test_df = split_data(df, cfg.data.splits)
        
        # Prepare features and targets
        X_train = train_df[feature_cols]
        y_train = train_df['points']
        X_val = val_df[feature_cols]
        y_val = val_df['points']
        
        # Train model
        trainer = XGBoostTrainer(OmegaConf.to_container(cfg.model, resolve=True))
        results = trainer.train(
            X_train, y_train, X_val, y_val,
            train_df['position'], val_df['position']
        )
        
        # Log results
        for metric_name, metrics in results.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if cfg.training.mlflow.enabled:
                        mlflow.log_metric(f"{metric_name}_{k}", v)
                    logger.info(f"{metric_name}_{k}: {v}")
        
        # Save model
        model_path = Path(cfg.models_dir) / "xgboost_model.pkl"
        model_path.parent.mkdir(exist_ok=True)
        trainer.save_models(str(model_path))
        
        if cfg.training.mlflow.enabled:
            mlflow.log_artifact(str(model_path))
            
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if cfg.training.mlflow.enabled:
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if cfg.training.mlflow.enabled:
            mlflow.end_run()

def create_dummy_data() -> pd.DataFrame:
    """Create dummy FPL data for testing."""
    np.random.seed(42)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    players = []
    
    for i in range(200):  # 200 players
        for gw in range(1, 39):  # 38 gameweeks
            players.append({
                'player_id': i,
                'gameweek': gw,
                'position': np.random.choice(positions),
                'minutes': np.random.randint(0, 91),
                'goals_scored': np.random.poisson(0.3),
                'assists': np.random.poisson(0.2),
                'bps': np.random.randint(0, 40),
                'influence': np.random.normal(20, 10),
                'creativity': np.random.normal(15, 8),
                'threat': np.random.normal(10, 5),
                'ict_index': np.random.normal(8, 3),
                'expected_goals': np.random.exponential(0.5),
                'expected_assists': np.random.exponential(0.3),
                'selected_by_percent': np.random.uniform(0.1, 50),
                'points': np.random.randint(0, 15)
            })
    
    return pd.DataFrame(players)

def split_data(df: pd.DataFrame, split_config: DictConfig) -> tuple:
    """Split data into train/val/test."""
    # Sort by gameweek for temporal split
    df = df.sort_values('gameweek')
    
    n_total = len(df['gameweek'].unique())
    train_gws = int(n_total * split_config.train_size)
    val_gws = int(n_total * split_config.val_size)
    
    train_df = df[df['gameweek'] <= train_gws]
    val_df = df[(df['gameweek'] > train_gws) & (df['gameweek'] <= train_gws + val_gws)]
    test_df = df[df['gameweek'] > train_gws + val_gws]
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train()
