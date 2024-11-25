import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class ModelTrainer:
    def __init__(self):
        # Base models
        self.base_models = {
            'RandomForest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42
                ))
            ]),
            'GradientBoosting': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ))
            ]),
            'XGBoost': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'LightGBM': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=42
                ))
            ])
        }
        
        # Meta-learner
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        
    def prepare_data(self, df):
        """Prepare data with proper temporal ordering"""
        # Sort by date to ensure temporal order
        df = df.sort_values('Date')
        
        # Create season column
        month = df['Date'].dt.month
        year = df['Date'].dt.year
        df['Season'] = year.where(month >= 8, year - 1)
        
        # Define feature groups
        basic_stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK']
        advanced_features = [
            'HOME_REST_DAYS', 'AWAY_REST_DAYS',
            'HOME_WIN_STREAK', 'AWAY_WIN_STREAK',
            'HOME_WIN_PCT', 'AWAY_WIN_PCT',
            'PACE', 'FG3A_RATE', 'PAINT_PCT',
            'STYLE_MISMATCH_SCORE',
            'PACE_ADVANTAGE', 'SHOOTING_ADVANTAGE',
            'PAINT_ADVANTAGE', 'BALL_CONTROL_ADVANTAGE'
        ]
        
        # Create feature list
        feature_columns = []
        
        # Add basic stats (already have HOME_AVG and AWAY_AVG prefixes in data)
        for stat in basic_stats:
            feature_columns.extend([f'HOME_AVG_{stat}', f'AWAY_AVG_{stat}'])
            feature_columns.extend([f'HOME_STD_{stat}', f'AWAY_STD_{stat}'])
        
        # Add advanced features
        feature_columns.extend(advanced_features)
        
        # Add rolling and weighted features
        rolling_windows = [5, 10, 15]
        for window in rolling_windows:
            for stat in basic_stats:
                feature_columns.extend([
                    f'ROLLING_HOME_AVG_{stat}_{window}G',
                    f'STD_HOME_AVG_{stat}_{window}G',
                    f'WEIGHTED_HOME_AVG_{stat}_{window}G'
                ])
        
        # Add relative features
        feature_columns.extend(['REL_' + stat for stat in basic_stats])
        
        # Add H2H features
        for window in rolling_windows:
            feature_columns.extend([
                f'H2H_WIN_RATE_{window}G',
                f'H2H_POINT_DIFF_{window}G',
                f'H2H_SCORING_DIFF_{window}G',
                f'H2H_FG_PCT_DIFF_{window}G',
                f'H2H_FG3_PCT_DIFF_{window}G'
            ])
        
        # Add style features
        style_features = [
            'STYLE_DIFF_PACE', 'STYLE_RATIO_PACE',
            'STYLE_DIFF_FG3_RATE', 'STYLE_RATIO_FG3_RATE',
            'STYLE_DIFF_PAINT_RATE', 'STYLE_RATIO_PAINT_RATE',
            'STYLE_DIFF_AST_RATE', 'STYLE_RATIO_AST_RATE',
            'STYLE_DIFF_TOV_RATE', 'STYLE_RATIO_TOV_RATE',
            'STYLE_DIFF_REB_RATE', 'STYLE_RATIO_REB_RATE',
            'STYLE_DIFF_EFFICIENCY', 'STYLE_RATIO_EFFICIENCY'
        ]
        feature_columns.extend(style_features)
        
        # Add interaction features
        df['PACE_SHOOTING_INTERACTION'] = df['PACE'] * df['SHOOTING_ADVANTAGE']
        df['REST_PACE_ADVANTAGE'] = df['HOME_REST_DAYS'] * df['PACE_ADVANTAGE']
        df['MOMENTUM_SCORE'] = df['HOME_WIN_STREAK'] * df['HOME_WIN_PCT']
        df['STYLE_EFFICIENCY_IMPACT'] = df['STYLE_MISMATCH_SCORE'] * df['STYLE_DIFF_EFFICIENCY']
        
        feature_columns.extend([
            'PACE_SHOOTING_INTERACTION',
            'REST_PACE_ADVANTAGE',
            'MOMENTUM_SCORE',
            'STYLE_EFFICIENCY_IMPACT'
        ])
        
        # Add time-based features
        df['DAYS_SINCE_SEASON_START'] = (df['Date'] - df.groupby('Season')['Date'].transform('min')).dt.days
        df['DAYS_TO_SEASON_END'] = (df.groupby('Season')['Date'].transform('max') - df['Date']).dt.days
        df['IS_BACK_TO_BACK'] = (df['HOME_REST_DAYS'] <= 1) | (df['AWAY_REST_DAYS'] <= 1)
        
        feature_columns.extend([
            'DAYS_SINCE_SEASON_START',
            'DAYS_TO_SEASON_END',
            'IS_BACK_TO_BACK'
        ])
        
        # Ensure all feature columns exist in the dataset
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Split features and target
        X = df[feature_columns]
        y = df['Home_Team_Win']
        
        # Get list of seasons for temporal splits
        seasons = df['Season'].unique()
        seasons.sort()
        
        # Use the last season as holdout set
        holdout_season = seasons[-1]
        train_seasons = seasons[:-1]
        
        # Create train and holdout sets
        train_mask = df['Season'].isin(train_seasons)
        holdout_mask = df['Season'] == holdout_season
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_holdout = X[holdout_mask]
        y_holdout = y[holdout_mask]
        
        # Store feature names for importance analysis
        self.feature_names = feature_columns
        
        return X_train, X_holdout, y_train, y_holdout, train_seasons

    def create_temporal_cv(self, X, y, seasons, n_splits=5):
        """Create temporal cross-validation splits"""
        splits = []
        for i in range(len(seasons) - n_splits + 1, len(seasons)):
            train_seasons = seasons[:i]
            val_season = seasons[i]
            
            train_idx = X.index[X.index.map(lambda x: x in train_seasons)]
            val_idx = X.index[X.index.map(lambda x: x == val_season)]
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    def evaluate_predictions(self, y_true, y_pred, y_prob, threshold=0.55):
        """Evaluate predictions with realistic metrics including ROI analysis"""
        # Basic classification metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }
        
        # High confidence predictions
        high_conf_mask = (y_prob >= threshold) | (y_prob <= (1-threshold))
        if high_conf_mask.any():
            high_conf_pred = y_pred[high_conf_mask]
            high_conf_true = y_true[high_conf_mask]
            metrics['High_Conf_Accuracy'] = accuracy_score(high_conf_true, high_conf_pred)
            metrics['High_Conf_Count'] = len(high_conf_pred)
            metrics['High_Conf_Percentage'] = len(high_conf_pred) / len(y_pred) * 100
        
        # ROI Analysis (assuming -110 odds)
        bet_amount = 100
        win_payout = bet_amount * (100/110)
        
        # Calculate ROI for different confidence thresholds
        for thresh in [0.55, 0.60, 0.65]:
            conf_mask = (y_prob >= thresh) | (y_prob <= (1-thresh))
            if conf_mask.any():
                conf_pred = y_pred[conf_mask]
                conf_true = y_true[conf_mask]
                
                # Calculate profits
                correct_bets = (conf_pred == conf_true).sum()
                wrong_bets = (conf_pred != conf_true).sum()
                total_profit = (correct_bets * win_payout) - (wrong_bets * bet_amount)
                roi = (total_profit / (bet_amount * len(conf_pred))) * 100
                
                metrics[f'ROI_{thresh}'] = roi
                metrics[f'Bets_{thresh}'] = len(conf_pred)
        
        return metrics

    def analyze_feature_importance(self, model, X):
        """Analyze and return feature importance scores"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return None
            
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_imp

    def train_and_evaluate(self, X_train, X_holdout, y_train, y_holdout, train_seasons):
        """Train and evaluate models with stacking ensemble"""
        results = {}
        
        # Create temporal CV splits
        cv_splits = self.create_temporal_cv(X_train, y_train, train_seasons)
        
        # Generate meta-features
        meta_features_train = np.zeros((X_train.shape[0], len(self.base_models)))
        meta_features_holdout = np.zeros((X_holdout.shape[0], len(self.base_models)))
        
        print("\nTraining base models and generating meta-features...")
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"\nTraining {name}...")
            
            # Generate meta-features using cross-validation
            for train_idx, val_idx in cv_splits:
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_cv_train, y_cv_train)
                meta_features_train[val_idx, i] = model.predict_proba(X_cv_val)[:, 1]
            
            # Train on full training set and predict holdout
            model.fit(X_train, y_train)
            meta_features_holdout[:, i] = model.predict_proba(X_holdout)[:, 1]
            
            # Calculate base model metrics
            y_pred = model.predict(X_holdout)
            y_prob = model.predict_proba(X_holdout)[:, 1]
            metrics = self.evaluate_predictions(y_holdout, y_pred, y_prob)
            
            # Store results
            results[name] = {
                'metrics': metrics,
                'model': model
            }
        
        # Train meta-learner
        print("\nTraining meta-learner...")
        self.meta_learner.fit(meta_features_train, y_train)
        
        # Make ensemble predictions
        ensemble_pred_prob = self.meta_learner.predict_proba(meta_features_holdout)[:, 1]
        ensemble_pred = self.meta_learner.predict(meta_features_holdout)
        
        # Calculate ensemble metrics
        ensemble_metrics = self.evaluate_predictions(y_holdout, ensemble_pred, ensemble_pred_prob)
        results['Stacking_Ensemble'] = {
            'metrics': ensemble_metrics,
            'model': self.meta_learner
        }
        
        return results

def main():
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    db_path = os.path.join(project_root, 'Data', 'dataset.sqlite')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM dataset_2012_24_enhanced", conn)
    conn.close()
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    # Prepare data
    X_train, X_holdout, y_train, y_holdout, train_seasons = trainer.prepare_data(df)
    
    print("\nStarting model training and evaluation...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Holdout data shape: {X_holdout.shape}")
    print(f"Training seasons: {train_seasons}")
    print(f"Holdout season: {train_seasons[-1] + 1}")
    
    # Train and evaluate models
    results = trainer.train_and_evaluate(X_train, X_holdout, y_train, y_holdout, train_seasons)
    
    # Print results
    print("\n=== Model Performance Summary ===")
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print("\nMetrics:")
        for metric, value in metrics['metrics'].items():
            print(f"{metric}: {value:.2f}")
    
    # Save models
    print("\nSaving models...")
    models_dir = os.path.join(project_root, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trainer.base_models.items():
        model_path = os.path.join(models_dir, f'{name.lower()}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")
    
    model_path = os.path.join(models_dir, 'stacking_ensemble_model.joblib')
    joblib.dump(trainer.meta_learner, model_path)
    print(f"Saved Stacking Ensemble model to {model_path}")

if __name__ == "__main__":
    main()
