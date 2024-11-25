import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime, timedelta
import joblib
from sklearn.calibration import calibration_curve
from scipy.stats import norm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class AdvancedModelEvaluator:
    def __init__(self, models_dir: str):
        """Initialize evaluator with path to saved models"""
        self.models_dir = models_dir
        self.models = self._load_models()
        self.results_dir = os.path.join(project_root, 'Results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_models(self) -> Dict:
        """Load all saved models"""
        models = {}
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(self.models_dir, model_file)
            models[model_name] = joblib.load(model_path)
        
        return models
    
    def evaluate_confidence_levels(self, X: pd.DataFrame, y: pd.Series, 
                                 confidence_thresholds: List[float] = None) -> pd.DataFrame:
        """Evaluate model performance at different confidence thresholds"""
        if confidence_thresholds is None:
            confidence_thresholds = np.arange(0.5, 0.95, 0.05)
        
        results = []
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name} at different confidence levels...")
            y_pred_proba = model.predict_proba(X)
            
            for threshold in confidence_thresholds:
                # Get predictions where model is confident
                confident_mask = np.max(y_pred_proba, axis=1) >= threshold
                if not any(confident_mask):
                    continue
                
                y_pred = np.argmax(y_pred_proba[confident_mask], axis=1)
                y_true_confident = y[confident_mask]
                
                # Calculate metrics
                metrics = {
                    'model': model_name,
                    'threshold': threshold,
                    'accuracy': accuracy_score(y_true_confident, y_pred),
                    'precision': precision_score(y_true_confident, y_pred),
                    'recall': recall_score(y_true_confident, y_pred),
                    'f1': f1_score(y_true_confident, y_pred),
                    'coverage': np.mean(confident_mask),
                    'n_predictions': sum(confident_mask)
                }
                
                # Calculate ROI
                bet_amount = 100
                win_payout = bet_amount * 1.91  # Assuming -110 odds
                correct_bets = sum(y_pred == y_true_confident)
                wrong_bets = sum(y_pred != y_true_confident)
                roi = ((correct_bets * win_payout) - (wrong_bets * bet_amount)) / (bet_amount * len(y_pred))
                metrics['roi'] = roi
                
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def analyze_streak_performance(self, df: pd.DataFrame, window_sizes: List[int] = None) -> pd.DataFrame:
        """Analyze model performance during winning/losing streaks"""
        if window_sizes is None:
            window_sizes = [3, 5, 7]
        
        results = []
        for model_name, model in self.models.items():
            print(f"\nAnalyzing streak performance for {model_name}...")
            
            for window in window_sizes:
                # Calculate team streaks
                home_streak = df.groupby('Home_Team_ID')['Home_Team_Win'].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                away_streak = df.groupby('Away_Team_ID')['Home_Team_Win'].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # Analyze performance during hot/cold streaks
                for streak_type, streak_data in [('home', home_streak), ('away', away_streak)]:
                    for streak_threshold in [0.7, 0.3]:  # Hot and cold thresholds
                        streak_mask = streak_data >= streak_threshold if streak_threshold > 0.5 else streak_data <= streak_threshold
                        if not any(streak_mask):
                            continue
                        
                        X_streak = df[streak_mask]
                        y_streak = df.loc[streak_mask, 'Home_Team_Win']
                        
                        y_pred = model.predict(X_streak)
                        accuracy = accuracy_score(y_streak, y_pred)
                        
                        results.append({
                            'model': model_name,
                            'window': window,
                            'streak_type': streak_type,
                            'streak_threshold': streak_threshold,
                            'accuracy': accuracy,
                            'n_games': sum(streak_mask)
                        })
        
        return pd.DataFrame(results)
    
    def evaluate_calibration(self, X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> None:
        """Evaluate and plot model probability calibration"""
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=n_bins)
            
            plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability')
        plt.title('Model Calibration Curves')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.results_dir, 'calibration_curves.png'))
        plt.close()
    
    def analyze_value_bets(self, X: pd.DataFrame, y: pd.Series, market_odds: pd.Series) -> pd.DataFrame:
        """Analyze value betting opportunities"""
        results = []
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing value bets for {model_name}...")
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Convert odds to implied probability
            implied_prob = 1 / market_odds
            
            # Calculate edge
            edge = y_pred_proba - implied_prob
            
            # Analyze different edge thresholds
            for edge_threshold in [0.05, 0.10, 0.15]:
                value_mask = edge >= edge_threshold
                if not any(value_mask):
                    continue
                
                y_pred = y_pred_proba[value_mask] >= 0.5
                y_true = y[value_mask]
                market_odds_filtered = market_odds[value_mask]
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Calculate actual ROI using market odds
                bet_amount = 100
                total_bets = len(y_true)
                winnings = sum((y_true == 1) * (market_odds_filtered * bet_amount))
                investment = total_bets * bet_amount
                roi = (winnings - investment) / investment
                
                results.append({
                    'model': model_name,
                    'edge_threshold': edge_threshold,
                    'accuracy': accuracy,
                    'roi': roi,
                    'n_bets': total_bets,
                    'avg_odds': market_odds_filtered.mean()
                })
        
        return pd.DataFrame(results)
    
    def plot_confidence_distribution(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Plot distribution of model confidence scores"""
        plt.figure(figsize=(15, 10))
        
        for i, (model_name, model) in enumerate(self.models.items(), 1):
            plt.subplot(2, 2, i)
            
            y_pred_proba = model.predict_proba(X)
            confidence_scores = np.max(y_pred_proba, axis=1)
            
            # Plot confidence distribution
            sns.histplot(confidence_scores, bins=50)
            plt.axvline(np.mean(confidence_scores), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(confidence_scores):.3f}')
            
            plt.title(f'{model_name} Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confidence_distributions.png'))
        plt.close()
    
    def generate_evaluation_report(self, X: pd.DataFrame, y: pd.Series, 
                                 market_odds: pd.Series = None) -> None:
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive model evaluation report...")
        
        # 1. Confidence level analysis
        conf_results = self.evaluate_confidence_levels(X, y)
        conf_results.to_csv(os.path.join(self.results_dir, 'confidence_analysis.csv'))
        
        # 2. Streak analysis
        streak_results = self.analyze_streak_performance(X)
        streak_results.to_csv(os.path.join(self.results_dir, 'streak_analysis.csv'))
        
        # 3. Calibration analysis
        self.evaluate_calibration(X, y)
        
        # 4. Value bet analysis (if market odds provided)
        if market_odds is not None:
            value_results = self.analyze_value_bets(X, y, market_odds)
            value_results.to_csv(os.path.join(self.results_dir, 'value_bets_analysis.csv'))
        
        # 5. Confidence distribution plots
        self.plot_confidence_distribution(X, y)
        
        print(f"\nEvaluation report generated. Results saved in: {self.results_dir}")

def main():
    # Initialize evaluator
    models_dir = os.path.join(project_root, 'Models')
    evaluator = AdvancedModelEvaluator(models_dir)
    
    # Load test data
    db_path = os.path.join(project_root, 'Data', 'dataset.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Get most recent season for testing
    query = """
    SELECT * FROM dataset_2012_24_enhanced 
    WHERE Season = (SELECT MAX(Season) FROM dataset_2012_24_enhanced)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Prepare features (use same feature preparation as training)
    feature_cols = [col for col in df.columns if col not in ['Home_Team_Win', 'Date', 'Season']]
    X = df[feature_cols]
    y = df['Home_Team_Win']
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(X, y)
    
    print("\nEvaluation complete. Check the Results directory for detailed analysis.")

if __name__ == "__main__":
    main()
