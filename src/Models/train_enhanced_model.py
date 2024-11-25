import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class EnhancedModelTrainer:
    def __init__(self, stats_db_path, models_dir):
        self.stats_db_path = stats_db_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def load_training_data(self, start_date='2022-10-01', end_date='2023-02-28'):
        """Load training data with enhanced features"""
        from src.Process_Data.enhanced_processor import EnhancedStatsProcessor
        processor = EnhancedStatsProcessor()
        return processor.load_data(start_date, end_date)
        
    def optimize_hyperparameters(self, X_train, y_train, target):
        """Use Optuna to find optimal hyperparameters"""
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Use cross-validation to evaluate parameters
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=1000,
                nfold=5,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Return the best MAE
            return np.min(cv_results['valid_mae-mean'])
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
        
    def train_model(self, X_train, y_train, params):
        """Train a LightGBM model with given parameters"""
        train_data = lgb.Dataset(X_train, label=y_train)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
        
    def train_all_models(self):
        """Train all prediction models with enhanced features"""
        # Load and prepare data
        logging.info("Loading training data...")
        df = self.load_training_data()
        
        # Prepare features
        from src.Process_Data.enhanced_processor import EnhancedStatsProcessor
        processor = EnhancedStatsProcessor()
        X = processor.prepare_features(df)
        
        # Train models for each target
        targets = ['points', 'assists', 'rebounds', 'efficiency']
        
        for target in targets:
            logging.info(f"\nTraining {target} prediction model...")
            
            # Prepare target variable
            y = df[target]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Optimize hyperparameters
            logging.info("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train, target)
            
            # Train final model
            logging.info("Training final model...")
            model = self.train_model(X_train_scaled, y_train, best_params)
            
            # Save model and scaler
            model_path = os.path.join(self.models_dir, f'{target}_prediction_model.joblib')
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'features': list(X.columns),
                'params': best_params,
                'training_date': datetime.now().strftime('%Y-%m-%d')
            }, model_path)
            
            logging.info(f"Saved {target} model to {model_path}")
            
def main():
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    models_dir = os.path.join(project_root, 'Models')
    
    # Initialize trainer
    trainer = EnhancedModelTrainer(stats_db_path, models_dir)
    
    # Train all models
    trainer.train_all_models()
    
if __name__ == "__main__":
    main()
