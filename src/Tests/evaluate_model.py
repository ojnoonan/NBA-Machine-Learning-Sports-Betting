import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

def load_data(with_player_features=False):
    """Load and prepare data for model evaluation"""
    db_path = os.path.join(project_root, 'Data', 'dataset.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Load game data
    game_data = pd.read_sql_query("SELECT * FROM dataset_2012_24", conn)
    
    # Pre-game features for prediction
    base_features = [
        # Home team rolling averages
        'HOME_AVG_PTS', 'HOME_AVG_FG_PCT', 'HOME_AVG_FG3_PCT', 'HOME_AVG_FT_PCT',
        'HOME_AVG_REB', 'HOME_AVG_AST', 'HOME_AVG_TOV', 'HOME_AVG_STL', 'HOME_AVG_BLK',
        
        # Home team rolling standard deviations
        'HOME_STD_PTS', 'HOME_STD_FG_PCT', 'HOME_STD_FG3_PCT', 'HOME_STD_FT_PCT',
        'HOME_STD_REB', 'HOME_STD_AST', 'HOME_STD_TOV', 'HOME_STD_STL', 'HOME_STD_BLK',
        
        # Away team rolling averages
        'AWAY_AVG_PTS', 'AWAY_AVG_FG_PCT', 'AWAY_AVG_FG3_PCT', 'AWAY_AVG_FT_PCT',
        'AWAY_AVG_REB', 'AWAY_AVG_AST', 'AWAY_AVG_TOV', 'AWAY_AVG_STL', 'AWAY_AVG_BLK',
        
        # Away team rolling standard deviations
        'AWAY_STD_PTS', 'AWAY_STD_FG_PCT', 'AWAY_STD_FG3_PCT', 'AWAY_STD_FT_PCT',
        'AWAY_STD_REB', 'AWAY_STD_AST', 'AWAY_STD_TOV', 'AWAY_STD_STL', 'AWAY_STD_BLK'
    ]
    
    if with_player_features:
        try:
            # Load and join player data
            player_data = pd.read_sql_query("SELECT * FROM player_stats", conn)
            game_data = pd.merge(
                game_data,
                player_data,
                how='left',
                left_on=['Date', 'Home_Team_ID'],
                right_on=['game_date', 'team_id']
            )
            
            # Add player features to the feature list
            player_features = [col for col in player_data.columns 
                             if col not in ['game_date', 'team_id']]
            base_features.extend(player_features)
            
        except Exception as e:
            print(f"Warning: Could not load player features: {e}")
            print("Proceeding with base features only...")
    
    conn.close()
    
    # Prepare features and target
    X = game_data[base_features]
    y = game_data['Home_Team_Win']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y

def train_and_evaluate_model(X, y, model_name="Base Model"):
    """Train and evaluate the model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC-ROC: {auc_roc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False
    )
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title(f'Top 10 Feature Importance - {model_name}')
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(project_root, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    return model, feature_importance

def main():
    print("Loading data...")
    
    # Evaluate base model
    print("\nEvaluating base model (without player features)...")
    X, y = load_data(with_player_features=False)
    base_model, base_importance = train_and_evaluate_model(X, y, "Base Model")
    
    # Evaluate enhanced model with player features
    print("\nEvaluating enhanced model (with player features)...")
    X_enhanced, y_enhanced = load_data(with_player_features=True)
    enhanced_model, enhanced_importance = train_and_evaluate_model(
        X_enhanced, y_enhanced, "Enhanced Model"
    )
    
    # Compare feature importance between models
    print("\nFeature Importance Comparison:")
    print("\nBase Model vs Enhanced Model")
    print("-----------------------------")
    print("Base Model Top 5:")
    print(base_importance.head())
    print("\nEnhanced Model Top 5:")
    print(enhanced_importance.head())

if __name__ == "__main__":
    main()
