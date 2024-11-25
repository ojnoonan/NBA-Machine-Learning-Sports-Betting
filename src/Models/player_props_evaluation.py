import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sqlite3
import joblib
from scipy.stats import norm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class PlayerPropsEvaluator:
    def __init__(self, models_dir: str):
        """Initialize evaluator with path to saved models"""
        self.models_dir = os.path.join(models_dir, 'player_props')
        self.models = self._load_models()
        self.results_dir = os.path.join(project_root, 'Results', 'player_props')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_models(self) -> Dict:
        """Load all player props models"""
        models = {}
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib') and not f.startswith('scaler')]
        
        for model_file in model_files:
            prop_type = model_file.replace('player_props_', '').replace('_model.joblib', '')
            model_path = os.path.join(self.models_dir, model_file)
            models[prop_type] = joblib.load(model_path)
        
        return models
    
    def evaluate_prediction_accuracy(self, X: pd.DataFrame, y: pd.Series, 
                                  prop_type: str) -> Dict:
        """Evaluate prediction accuracy for a specific prop type"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        metrics = {
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'R2': r2_score(y, y_pred)
        }
        
        # Calculate accuracy for different thresholds
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            within_threshold = np.abs(y - y_pred) <= threshold
            metrics[f'Accuracy_within_{threshold}'] = np.mean(within_threshold)
        
        return metrics
    
    def analyze_over_under_accuracy(self, X: pd.DataFrame, y: pd.Series, 
                                  prop_type: str, lines: pd.Series) -> Dict:
        """Analyze over/under prediction accuracy"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        # Calculate actual over/under results
        actual_overs = y > lines
        
        # Calculate predicted over/under results
        predicted_overs = y_pred > lines
        
        # Calculate metrics
        accuracy = np.mean(actual_overs == predicted_overs)
        
        # Calculate ROI (assuming -110 odds)
        bet_amount = 100
        win_payout = bet_amount * 1.91
        
        correct_bets = np.sum(actual_overs == predicted_overs)
        total_bets = len(y)
        
        roi = ((correct_bets * win_payout) - (total_bets * bet_amount)) / (total_bets * bet_amount)
        
        return {
            'accuracy': accuracy,
            'roi': roi,
            'total_bets': total_bets,
            'correct_bets': correct_bets
        }
    
    def analyze_player_specific_accuracy(self, X: pd.DataFrame, y: pd.Series,
                                      player_ids: pd.Series, prop_type: str) -> pd.DataFrame:
        """Analyze prediction accuracy for specific players"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        results = []
        for player_id in player_ids.unique():
            player_mask = player_ids == player_id
            if sum(player_mask) < 10:  # Minimum games threshold
                continue
            
            y_player = y[player_mask]
            y_pred_player = y_pred[player_mask]
            
            metrics = {
                'player_id': player_id,
                'n_games': sum(player_mask),
                'mae': mean_absolute_error(y_player, y_pred_player),
                'rmse': np.sqrt(mean_squared_error(y_player, y_pred_player))
            }
            
            # Calculate accuracy within different thresholds
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                within_threshold = np.abs(y_player - y_pred_player) <= threshold
                metrics[f'accuracy_within_{threshold}'] = np.mean(within_threshold)
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def analyze_matchup_impact(self, X: pd.DataFrame, y: pd.Series,
                             opponent_ids: pd.Series, prop_type: str) -> pd.DataFrame:
        """Analyze prediction accuracy based on opponents"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        results = []
        for opp_id in opponent_ids.unique():
            opp_mask = opponent_ids == opp_id
            if sum(opp_mask) < 10:  # Minimum games threshold
                continue
            
            y_opp = y[opp_mask]
            y_pred_opp = y_pred[opp_mask]
            
            metrics = {
                'opponent_id': opp_id,
                'n_games': sum(opp_mask),
                'mae': mean_absolute_error(y_opp, y_pred_opp),
                'rmse': np.sqrt(mean_squared_error(y_opp, y_pred_opp))
            }
            
            # Calculate accuracy within different thresholds
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                within_threshold = np.abs(y_opp - y_pred_opp) <= threshold
                metrics[f'accuracy_within_{threshold}'] = np.mean(within_threshold)
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def analyze_rest_impact(self, X: pd.DataFrame, y: pd.Series,
                          rest_days: pd.Series, prop_type: str) -> pd.DataFrame:
        """Analyze prediction accuracy based on rest days"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        results = []
        for days in rest_days.unique():
            rest_mask = rest_days == days
            if sum(rest_mask) < 10:  # Minimum games threshold
                continue
            
            y_rest = y[rest_mask]
            y_pred_rest = y_pred[rest_mask]
            
            metrics = {
                'rest_days': days,
                'n_games': sum(rest_mask),
                'mae': mean_absolute_error(y_rest, y_pred_rest),
                'rmse': np.sqrt(mean_squared_error(y_rest, y_pred_rest))
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_prediction_distribution(self, X: pd.DataFrame, y: pd.Series,
                                  prop_type: str) -> None:
        """Plot distribution of predictions vs actuals"""
        model = self.models[prop_type]
        y_pred = model.predict(X)
        
        plt.figure(figsize=(12, 6))
        
        # Plot distributions
        sns.kdeplot(y, label='Actual', color='blue')
        sns.kdeplot(y_pred, label='Predicted', color='red')
        
        plt.title(f'{prop_type.capitalize()} Distribution - Actual vs Predicted')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, f'{prop_type}_distribution.png'))
        plt.close()
    
    def generate_evaluation_report(self, X: pd.DataFrame, y: Dict[str, pd.Series],
                                 player_ids: pd.Series, opponent_ids: pd.Series,
                                 rest_days: pd.Series) -> None:
        """Generate comprehensive evaluation report for all prop types"""
        print("Generating player props evaluation report...")
        
        for prop_type in self.models.keys():
            print(f"\nEvaluating {prop_type} predictions...")
            
            # Basic accuracy metrics
            metrics = self.evaluate_prediction_accuracy(X, y[prop_type], prop_type)
            
            # Player-specific analysis
            player_analysis = self.analyze_player_specific_accuracy(
                X, y[prop_type], player_ids, prop_type
            )
            
            # Matchup analysis
            matchup_analysis = self.analyze_matchup_impact(
                X, y[prop_type], opponent_ids, prop_type
            )
            
            # Rest impact analysis
            rest_analysis = self.analyze_rest_impact(
                X, y[prop_type], rest_days, prop_type
            )
            
            # Save results
            results_dir = os.path.join(self.results_dir, prop_type)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save metrics
            pd.DataFrame([metrics]).to_csv(
                os.path.join(results_dir, 'accuracy_metrics.csv')
            )
            
            # Save analyses
            player_analysis.to_csv(os.path.join(results_dir, 'player_analysis.csv'))
            matchup_analysis.to_csv(os.path.join(results_dir, 'matchup_analysis.csv'))
            rest_analysis.to_csv(os.path.join(results_dir, 'rest_analysis.csv'))
            
            # Generate plots
            self.plot_prediction_distribution(X, y[prop_type], prop_type)
        
        print(f"\nEvaluation report generated. Results saved in: {self.results_dir}")

def prepare_features(df):
    """Prepare features for model training"""
    # Define features to exclude
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

def main():
    # Initialize evaluator
    models_dir = os.path.join(project_root, 'Models')
    evaluator = PlayerPropsEvaluator(models_dir)
    
    # Load test data
    db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Get most recent season for testing
    df = pd.read_sql_query("SELECT * FROM player_game_stats", conn)
    conn.close()
    
    # Prepare features
    X = prepare_features(df)
    
    # Load scaler
    scaler_path = os.path.join(project_root, 'Models', 'player_props', 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    
    # Convert back to DataFrame with column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Prepare target variables
    y = {
        'points': df['points'],
        'rebounds': df['rebounds'],
        'assists': df['assists'],
        'threes': df['threes'],
        'pra': df['points'] + df['rebounds'] + df['assists']
    }
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(
        X=X_scaled,
        y=y,
        player_ids=df['PLAYER_ID'],
        opponent_ids=df['OPPONENT_TEAM_ID'],
        rest_days=df['days_rest']
    )
    
    print("\nEvaluation complete. Check the Results directory for detailed analysis.")

if __name__ == "__main__":
    main()
