import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.Models.evaluate_enhanced_model import EnhancedModelEvaluator

def analyze_feature_importance():
    """Analyze feature importance using SHAP values"""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    models_dir = os.path.join(project_root, 'Models')
    
    # Initialize evaluator
    evaluator = EnhancedModelEvaluator(stats_db_path, models_dir)
    
    # Load test data
    df = evaluator.load_test_data()
    if len(df) == 0:
        print("No data available for analysis")
        return
    
    # Prepare features
    X = evaluator.prepare_features(df)
    
    # Create results directory
    results_dir = os.path.join(project_root, 'Results', 'feature_importance')
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze each model
    for target in ['points', 'assists', 'rebounds', 'efficiency']:
        if target in evaluator.models:
            print(f"\nAnalyzing {target} model features...")
            model = evaluator.models[target]['model']
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title(f"Feature Importance for {target.capitalize()} Prediction")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'feature_importance_{target}.png'))
            plt.close()
            
            # Get feature importance rankings
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(0)
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Save feature importance to CSV
            feature_importance.to_csv(
                os.path.join(results_dir, f'feature_importance_{target}.csv'),
                index=False
            )
            
if __name__ == "__main__":
    analyze_feature_importance()
