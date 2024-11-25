import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import sqlite3
import joblib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_data():
    """Load and prepare data for training"""
    db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Load all data
    df = pd.read_sql_query("SELECT * FROM player_game_stats", conn)
    conn.close()
    
    # Split into train and test based on dates
    dates = pd.to_datetime(df['GAME_DATE'])
    split_date = dates.max() - pd.Timedelta(days=30)  # Use last 30 days as test set
    
    train_df = df[dates < split_date]
    test_df = df[dates >= split_date]
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df

def prepare_features(df):
    """Prepare features for model training"""
    # Define features to exclude from training
    exclude_cols = ['points', 'rebounds', 'assists', 'threes', 'pra', 
                   'GAME_DATE', 'Season', 'GAME_ID', 'PLAYER_NAME', 
                   'TEAM_ABBREVIATION', 'TEAM_CITY']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Convert all features to float
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any missing values
    X = X.fillna(X.mean())
    
    return X

def train_models(X_train, y_train_dict):
    """Train models for each prop type"""
    models = {}
    
    # Model parameters
    params = {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 5,
        'learning_rate': 0.01,
        'random_state': 42
    }
    
    for prop_type, y_train in y_train_dict.items():
        print(f"\nTraining {prop_type} model...")
        
        # Initialize and train model
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        models[prop_type] = model
        
        # Save model
        save_path = os.path.join(project_root, 'Models', 'player_props', 
                               f'player_props_{prop_type}_model.joblib')
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    
    return models

def evaluate_models(models, X_test, y_test_dict):
    """Basic evaluation of trained models"""
    results = {}
    
    for prop_type, model in models.items():
        y_pred = model.predict(X_test)
        y_true = y_test_dict[prop_type]
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        results[prop_type] = {
            'MAE': mae,
            'RMSE': rmse
        }
    
    return results

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("\nPreparing features...")
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    
    # Prepare target variables
    y_train_dict = {
        'points': train_df['points'],
        'rebounds': train_df['rebounds'],
        'assists': train_df['assists'],
        'threes': train_df['threes'],
        'pra': train_df['points'] + train_df['rebounds'] + train_df['assists']
    }
    
    y_test_dict = {
        'points': test_df['points'],
        'rebounds': test_df['rebounds'],
        'assists': test_df['assists'],
        'threes': test_df['threes'],
        'pra': test_df['points'] + test_df['rebounds'] + test_df['assists']
    }
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(project_root, 'Models', 'player_props', 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train_scaled, y_train_dict)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test_scaled, y_test_dict)
    
    # Print results
    print("\nModel Performance:")
    for prop_type, metrics in results.items():
        print(f"\n{prop_type.upper()}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}")

if __name__ == "__main__":
    main()
