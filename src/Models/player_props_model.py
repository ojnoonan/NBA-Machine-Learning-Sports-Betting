import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple
import sqlite3

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class PlayerPropsPredictor:
    def __init__(self):
        self.models = {
            'points': self._create_model(),
            'rebounds': self._create_model(),
            'assists': self._create_model(),
            'threes': self._create_model(),
            'pra': self._create_model()  # Points + Rebounds + Assists
        }
        
    def _create_model(self):
        """Create a regression model for player props"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])
    
    def prepare_player_features(self, df: pd.DataFrame, stat_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for player prop prediction"""
        # Basic rolling averages
        windows = [5, 10, 15, 20]
        feature_columns = []
        
        # Create rolling averages for the target stat
        for window in windows:
            df[f'{stat_type}_last_{window}g'] = df.groupby('PLAYER_ID')[stat_type].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            feature_columns.append(f'{stat_type}_last_{window}g')
            
            # Add standard deviation
            df[f'{stat_type}_std_{window}g'] = df.groupby('PLAYER_ID')[stat_type].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            feature_columns.append(f'{stat_type}_std_{window}g')
        
        # Minutes played features
        for window in windows:
            df[f'minutes_last_{window}g'] = df.groupby('PLAYER_ID')['MIN'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            feature_columns.append(f'minutes_last_{window}g')
        
        # Matchup features
        df['opp_stat_allowed'] = df.groupby('OPPONENT_TEAM_ID')[stat_type].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        feature_columns.append('opp_stat_allowed')
        
        # Rest days
        df['days_rest'] = df.groupby('PLAYER_ID')['GAME_DATE'].transform(
            lambda x: (x - x.shift(1)).dt.days
        )
        feature_columns.append('days_rest')
        
        # Home/Away
        df['is_home'] = df['is_home'].astype(int)
        feature_columns.append('is_home')
        
        # Season progress
        df['game_number'] = df.groupby('PLAYER_ID').cumcount()
        feature_columns.append('game_number')
        
        # Target variable
        y = df[stat_type]
        X = df[feature_columns]
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, stat_type: str):
        """Train model for specific stat type"""
        self.models[stat_type].fit(X_train, y_train)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, stat_type: str) -> Dict:
        """Evaluate model performance"""
        y_pred = self.models[stat_type].predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Calculate accuracy for over/under predictions
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            correct_predictions = np.sum(
                (np.abs(y_test - y_pred) <= threshold)
            )
            metrics[f'Accuracy_within_{threshold}'] = correct_predictions / len(y_test)
        
        return metrics
    
    def predict_props(self, player_data: pd.DataFrame, stat_type: str) -> np.ndarray:
        """Make predictions for player props"""
        return self.models[stat_type].predict(player_data)
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        for stat_type, model in self.models.items():
            model_path = os.path.join(output_dir, f'player_props_{stat_type}_model.joblib')
            joblib.dump(model, model_path)
    
    def load_models(self, model_dir: str):
        """Load trained models"""
        for stat_type in self.models.keys():
            model_path = os.path.join(model_dir, f'player_props_{stat_type}_model.joblib')
            if os.path.exists(model_path):
                self.models[stat_type] = joblib.load(model_path)

def main():
    # Initialize predictor
    predictor = PlayerPropsPredictor()
    
    # Load data (you'll need to implement this based on your data structure)
    db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Train and evaluate models for each stat type
    stat_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
    
    for stat_type in stat_types:
        print(f"\nTraining model for {stat_type}...")
        
        # Load and prepare data
        query = f"SELECT * FROM player_game_stats ORDER BY GAME_DATE"
        df = pd.read_sql_query(query, conn)
        
        # Prepare features
        X, y = predictor.prepare_player_features(df, stat_type)
        
        # Split data temporally
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        predictor.train_model(X_train, y_train, stat_type)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test, stat_type)
        
        print(f"\nMetrics for {stat_type}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save models
    models_dir = os.path.join(project_root, 'Models', 'player_props')
    predictor.save_models(models_dir)
    print(f"\nSaved models to {models_dir}")
    
    conn.close()

if __name__ == "__main__":
    main()
