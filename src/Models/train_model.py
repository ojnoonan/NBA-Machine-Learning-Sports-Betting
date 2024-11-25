import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class NBABettingModel:
    def __init__(self):
        self.db_path = os.path.join(project_root, 'Data', 'dataset.sqlite')
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset for training"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM dataset_2012_24 ORDER BY Date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        return df
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Select features based on correlation analysis
        feature_columns = [
            'HOME_WIN_PCT', 'HOME_WIN_STREAK',
            'REL_PTS', 'HOME_AVG_FG_PCT', 'HOME_AVG_FG3_PCT',
            'HOME_REST_DAYS', 'AWAY_REST_DAYS',
            'HOME_AVG_PTS', 'AWAY_AVG_PTS',
            'HOME_AVG_AST', 'AWAY_AVG_AST',
            'HOME_AVG_REB', 'AWAY_AVG_REB'
        ]
        
        X = df[feature_columns]
        y = df['Home_Team_Win']
        
        return X, y
        
    def train_model(self, X_train, y_train):
        """Train the model using RandomForestClassifier"""
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
        
    def save_model(self, model_name='nba_betting_model.joblib'):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        model_path = os.path.join(project_root, 'Models', model_name)
        scaler_path = os.path.join(project_root, 'Models', f'scaler_{model_name}')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
def main():
    # Initialize the model
    nba_model = NBABettingModel()
    
    # Load and prepare data
    print("Loading data...")
    df = nba_model.load_data()
    X, y = nba_model.prepare_features(df)
    
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train and evaluate the model using time series cross-validation
    print("\nTraining and evaluating model using time series cross-validation...")
    metrics_list = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        print(f"\nFold {fold}:")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train the model
        nba_model.train_model(X_train, y_train)
        
        # Evaluate the model
        metrics = nba_model.evaluate_model(X_test, y_test)
        metrics_list.append(metrics)
        
        print(f"Metrics for fold {fold}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Calculate and display average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in metrics_list])
        for metric in metrics_list[0].keys()
    }
    
    print("\nAverage metrics across all folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Train final model on all data and save it
    print("\nTraining final model on all data...")
    nba_model.train_model(X, y)
    nba_model.save_model()
    print("Model training complete and saved!")

if __name__ == "__main__":
    main()
