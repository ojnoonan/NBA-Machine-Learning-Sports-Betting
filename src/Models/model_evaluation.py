import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple
import sqlite3
from pathlib import Path
import sys
from scipy.optimize import minimize
from collections import defaultdict

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))
from Features.advanced_features import AdvancedFeatureGenerator

class ModelEvaluator:
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=1000,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.5,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        self.ensemble_weights = None
        
        self.feature_categories = {
            'Basic Stats': ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_STL', 'AVG_BLK'],
            'Shooting': ['FG_PCT', 'FG3_PCT', 'FT_PCT'],
            'Advanced Metrics': ['PACE', 'OFF_EFF', 'DEF_EFF'],
            'Momentum': ['SCORING_MOMENTUM', 'DEFENSIVE_MOMENTUM', 'OVERALL_MOMENTUM'],
            'H2H': ['H2H_WIN_RATE', 'H2H_AVG_POINT_DIFF'],
            'Form': ['WIN_STREAK', 'WIN_PCT', 'WEIGHTED_STREAK']
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load data from SQLite database"""
        print("Loading data...")
        conn = sqlite3.connect('Data/dataset.sqlite')
        df = pd.read_sql_query("SELECT * FROM dataset_2012_24", conn)
        conn.close()
        
        # Generate advanced features
        feature_generator = AdvancedFeatureGenerator(df)
        df = feature_generator.generate_all_features()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        # Drop non-feature columns
        drop_cols = ['Date', 'Home_Team_ID', 'Away_Team_ID']
        
        # Separate features and target
        X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['Home_Team_Win'])
        y = df['Home_Team_Win']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns), y
    
    def optimize_ensemble_weights(self, X: pd.DataFrame, y: pd.Series, cv_splits: list) -> None:
        """Optimize ensemble weights using validation performance"""
        print("\nOptimizing ensemble weights...")
        
        # Initialize storage for predictions
        val_predictions = {name: np.zeros(len(y)) for name in self.models.keys()}
        
        # Get out-of-fold predictions for each model
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                val_predictions[name][val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Create prediction matrix
        pred_matrix = np.column_stack([val_predictions[name] for name in self.models.keys()])
        
        # Define objective function for weight optimization
        def objective(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_preds = np.dot(pred_matrix, weights)
            return -roc_auc_score(y, ensemble_preds)
        
        # Optimize weights
        bounds = [(0, 1)] * len(self.models)
        result = minimize(objective, x0=[1/len(self.models)]*len(self.models), 
                        bounds=bounds, method='L-BFGS-B')
        
        # Normalize weights
        weights = result.x / np.sum(result.x)
        self.ensemble_weights = dict(zip(self.models.keys(), weights))
        
        print("Optimized ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"{name}: {weight:.3f}")
    
    def ensemble_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using the weighted ensemble"""
        predictions = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 1/len(self.models))
            model_pred = model.predict_proba(X)
            predictions += weight * model_pred
        
        return predictions
    
    def analyze_betting_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  model_name: str) -> None:
        """Analyze betting performance with enhanced metrics"""
        print(f"\nBetting Performance Analysis - {model_name}")
        print("Confidence Threshold | Win Rate | Number of Bets | ROI | Kelly Criterion")
        print("-------------------------------------------------------------------")
        
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        
        for threshold in thresholds:
            # Get predictions above threshold
            confident_bets = y_pred_proba[:, 1] >= threshold
            if not any(confident_bets):
                continue
                
            n_bets = np.sum(confident_bets)
            wins = np.sum(y_true[confident_bets] == 1)
            win_rate = wins / n_bets
            
            # Calculate ROI (assuming -110 odds)
            stake = 100  # Standard bet size
            total_cost = n_bets * stake
            total_return = wins * (stake * 1.91)  # 1.91 is the return on -110 odds
            roi = (total_return - total_cost) / total_cost * 100
            
            # Calculate Kelly Criterion
            prob_win = win_rate
            prob_loss = 1 - prob_win
            decimal_odds = 1.91
            kelly = (prob_win * (decimal_odds - 1) - prob_loss) / (decimal_odds - 1)
            kelly = max(0, kelly)  # Only consider positive Kelly values
            
            print(f"      {threshold:.2f}         |  {win_rate:.3f}  |     {n_bets:4d}     | {roi:5.2f}% | {kelly:.3f}")
    
    def analyze_feature_importance(self, model, feature_cols, model_name: str):
        """Generate detailed feature importance analysis and plots"""
        print(f"\nAnalyzing feature importance for {model_name}...")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print(f"{model_name} does not support direct feature importance analysis")
            return
        
        # Create DataFrame of features and their importance scores
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        feature_importance.to_csv(f'feature_importance_{model_name}.csv', index=False)
        print(f"Feature importance saved to feature_importance_{model_name}.csv")
        
        # Plot overall feature importance
        plt.figure(figsize=(15, 10))
        plt.title(f'Top 20 Most Important Features - {model_name}')
        sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}_top20.png')
        plt.close()
        
        # Plot feature importance by category
        self.plot_feature_importance_by_category(feature_importance, model_name)
        
        # Print top 5 features by category
        self.print_top_features_by_category(feature_importance)
    
    def plot_feature_importance_by_category(self, feature_importance: pd.DataFrame, model_name: str):
        """Plot feature importance grouped by category"""
        plt.figure(figsize=(20, 12))
        plt.suptitle(f'Feature Importance by Category - {model_name}', y=1.02, fontsize=16)
        
        num_categories = len(self.feature_categories)
        rows = (num_categories + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(20, 6*rows))
        axes = axes.flatten()
        
        for idx, (category, prefixes) in enumerate(self.feature_categories.items()):
            ax = axes[idx]
            
            # Filter features by category
            mask = feature_importance['feature'].apply(
                lambda x: any(prefix in x for prefix in prefixes)
            )
            category_features = feature_importance[mask]
            
            if not category_features.empty:
                sns.barplot(data=category_features, x='importance', y='feature', ax=ax)
                ax.set_title(f'{category} Importance')
                ax.set_xlabel('Importance Score')
                ax.set_ylabel('Feature')
        
        # Remove empty subplots
        for idx in range(len(self.feature_categories), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}_by_category.png')
        plt.close()
    
    def print_top_features_by_category(self, feature_importance: pd.DataFrame):
        """Print top 5 features within each category"""
        print("\nTop 5 features by category:")
        for category, prefixes in self.feature_categories.items():
            print(f"\n{category}:")
            mask = feature_importance['feature'].apply(
                lambda x: any(prefix in x for prefix in prefixes)
            )
            category_features = feature_importance[mask]
            
            if not category_features.empty:
                for _, row in category_features.head().iterrows():
                    print(f"{row['feature']}: {row['importance']:.4f}")
    
    def run_evaluation(self) -> None:
        """Run the enhanced model evaluation"""
        df = self.load_data()
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['Home_Team_Win', 'Date']]
        X = df[feature_cols]
        y = df['Home_Team_Win']
        
        # Create time-based cross-validation splits
        cv_splits = []
        unique_dates = sorted(df['Date'].unique())
        n_splits = 5
        split_size = len(unique_dates) // n_splits
        
        for i in range(n_splits):
            if i < n_splits - 1:
                split_date = unique_dates[(i + 1) * split_size]
            else:
                split_date = unique_dates[-1]
            
            train_idx = df['Date'] < split_date
            val_idx = df['Date'] >= split_date
            cv_splits.append((train_idx, val_idx))
        
        # Optimize ensemble weights
        self.optimize_ensemble_weights(X, y, cv_splits)
        
        # Evaluate each model and ensemble
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            fold_metrics = defaultdict(list)
            fold_predictions = np.zeros(len(y))
            
            for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
                fold_predictions[val_idx] = y_pred_proba[:, 1]
                
                # Calculate metrics
                fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_metrics['precision'].append(precision_score(y_val, y_pred))
                fold_metrics['recall'].append(recall_score(y_val, y_pred))
                fold_metrics['f1'].append(f1_score(y_val, y_pred))
                fold_metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba[:, 1]))
                
                print(f"Fold {fold} Accuracy: {fold_metrics['accuracy'][-1]:.4f}")
            
            # Analyze feature importance
            self.analyze_feature_importance(model, feature_cols, name)
            
            # Print average metrics with confidence intervals
            print(f"\n{name} Performance Metrics:")
            for metric, values in fold_metrics.items():
                mean = np.mean(values)
                std = np.std(values)
                print(f"{metric}: {mean:.4f} (±{std:.4f})")
            
            # Analyze betting performance
            self.analyze_betting_performance(y, fold_predictions, name)
        
        # Evaluate ensemble
        print("\nEvaluating Weighted Ensemble...")
        ensemble_predictions = self.ensemble_predict_proba(X)
        self.analyze_betting_performance(y, ensemble_predictions, "Weighted Ensemble")

def main():
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
