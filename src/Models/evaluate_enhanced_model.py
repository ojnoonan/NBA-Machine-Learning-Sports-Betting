import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sqlite3
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedModelEvaluator:
    def __init__(self, stats_db_path, models_dir, test_mode=True):
        self.stats_db_path = stats_db_path
        self.models_dir = models_dir
        self.test_mode = test_mode  # If True, run on a smaller dataset
        logging.info(f"Initializing evaluator with test_mode={test_mode}")
        self.models = self._load_models()
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(models_dir), 'Results', 'enhanced_evaluation')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_models(self):
        """Load all prediction models"""
        models = {}
        for target in ['points', 'assists', 'rebounds', 'efficiency']:
            model_path = os.path.join(self.models_dir, f'{target}_prediction_model.joblib')
            if os.path.exists(model_path):
                logging.info(f"Loading {target} model...")
                model_data = joblib.load(model_path)
                models[target] = {
                    'model': model_data['model'],
                    'scaler': model_data['scaler']
                }
                logging.info(f"Loaded {target} model successfully")
        return models
    
    def load_test_data(self, start_date='2023-03-01', end_date='2023-04-09'):
        """Load test data with enhanced features"""
        if self.test_mode:
            # Use a smaller date range for testing
            end_date = start_date = '2023-03-01'
            logging.info("Test mode: Using single day of data")
        
        logging.info(f"Loading test data from {start_date} to {end_date}...")
        conn = sqlite3.connect(self.stats_db_path)
        
        # First, check if we have data for this period
        check_query = """
            SELECT COUNT(*) as count, COUNT(DISTINCT game_id) as games
            FROM player_game_stats
            WHERE game_date LIKE ?
        """
        stats = pd.read_sql_query(check_query, conn, params=[f"{start_date}%"]).iloc[0]
        logging.info(f"Found {stats['count']} player records across {stats['games']} games in date range")
        
        if stats['count'] == 0:
            logging.warning("No data found in the specified date range")
            conn.close()
            return pd.DataFrame()

        logging.info("Loading player game stats...")
        # Load data in chunks to avoid memory issues
        query = """
        SELECT pgs.*, pf.home_team_id, pf.away_team_id,
               pgs.rebounds_off + pgs.rebounds_def as rebounds,
               pgs.points + pgs.assists + pgs.rebounds_off + pgs.rebounds_def + 
               pgs.steals + pgs.blocks - (pgs.fga - pgs.fgm) - 
               (pgs.fta - pgs.ftm) - pgs.turnovers as efficiency
        FROM player_game_stats pgs
        JOIN pregame_features pf ON pgs.game_id = pf.game_id
        WHERE pgs.game_date LIKE ?
        """
        base_df = pd.read_sql_query(query, conn, params=[f"{start_date}%"])
        logging.info(f"Loaded {len(base_df)} base records")

        # Add opponent team ID
        base_df['opponent_team_id'] = np.where(
            base_df['game_id'].isin(base_df.query('player_id == player_id.iloc[0]')['game_id']),
            base_df['away_team_id'],
            base_df['home_team_id']
        )
        
        logging.info("Loading team defensive stats...")
        tds_query = "SELECT * FROM team_defensive_stats WHERE game_date LIKE ?"
        tds_df = pd.read_sql_query(tds_query, conn, params=[f"{start_date}%"])
        logging.info(f"Loaded {len(tds_df)} team defensive records")
        
        logging.info("Loading player vs team stats...")
        pvt_query = "SELECT * FROM player_vs_team_stats"
        pvt_df = pd.read_sql_query(pvt_query, conn)
        logging.info(f"Loaded {len(pvt_df)} player vs team records")
        
        logging.info("Loading player vs player stats...")
        pvp_query = "SELECT * FROM player_vs_player_stats"
        pvp_df = pd.read_sql_query(pvp_query, conn)
        logging.info(f"Loaded {len(pvp_df)} player vs player records")
        
        # Merge all data
        logging.info("Merging data...")
        df = base_df.merge(
            tds_df,
            left_on=['opponent_team_id', 'game_date'],
            right_on=['team_id', 'game_date'],
            how='left'
        )
        
        df = df.merge(
            pvt_df,
            left_on=['player_id', 'opponent_team_id'],
            right_on=['player_id', 'opponent_team_id'],
            how='left'
        )
        
        df = df.merge(
            pvp_df,
            left_on=['player_id'],
            right_on=['player_id'],
            how='left'
        )
        
        logging.info(f"Final dataset has {len(df)} records")
        
        # Log feature availability
        feature_columns = [
            'points_avg', 'assists_avg', 'rebounds_avg', 'efficiency_avg',
            'points_per_36', 'assists_per_36', 'rebounds_per_36'
        ]
        
        # Add defensive_rating if it exists
        if 'defensive_rating' in df.columns:
            feature_columns.append('defensive_rating')
            
        available_columns = [col for col in feature_columns if col in df.columns]
        if available_columns:
            null_counts = df[available_columns].isnull().sum()
            logging.info("Feature availability:")
            for col, null_count in null_counts.items():
                available_pct = (1 - null_count/len(df)) * 100
                logging.info(f"  {col}: {available_pct:.1f}% available")
        else:
            logging.warning("No advanced features found in dataset")

        conn.close()
        return df
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        logging.info("Preparing features for prediction...")
        
        # Calculate the required features
        df['fg_pct'] = df['fgm'] / df['fga'].replace(0, 1)
        df['fg3_pct'] = df['fg3m'] / df['fg3a'].replace(0, 1)
        df['ft_pct'] = df['ftm'] / df['fta'].replace(0, 1)
        df['total_rebounds'] = df['rebounds_off'] + df['rebounds_def']
        
        # Define feature sets matching the trained model
        feature_sets = {
            'shooting': [
                'fg_pct', 'fg3_pct', 'ft_pct'
            ],
            'game_stats': [
                'points', 'assists', 'total_rebounds',
                'steals', 'blocks', 'turnovers', 'fouls'
            ],
            'averages': [
                'points_avg', 'assists_avg', 'rebounds_avg',
                'efficiency_avg'
            ]
        }
        
        # Get available features
        available_features = []
        for feature_set in feature_sets.values():
            available_features.extend([f for f in feature_set if f in df.columns])
        
        logging.info(f"Using {len(available_features)} features for prediction")
        
        # Select features and handle missing values
        X = df[available_features].copy()
        X.fillna(0, inplace=True)  # Replace missing values with 0 instead of mean
        
        return X

    def evaluate_model(self, model_name, X, y):
        """Evaluate a single model's performance"""
        logging.info(f"\nEvaluating {model_name} predictions...")
        
        model_data = self.models[model_name]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get the features used during training
        train_features = scaler.feature_names_in_
        
        # Ensure X has the same features as the training data
        missing_features = set(train_features) - set(X.columns)
        extra_features = set(X.columns) - set(train_features)
        
        if missing_features:
            logging.warning(f"Missing features for {model_name} model: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
                
        if extra_features:
            logging.warning(f"Extra features found, removing: {extra_features}")
            X = X[train_features]
            
        # Ensure features are in the same order as during training
        X = X[train_features]
        
        logging.info("Scaling features...")
        X_scaled = scaler.transform(X)
        
        # Make predictions
        logging.info("Making predictions...")
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        logging.info(f"Model performance metrics:")
        logging.info(f"  MAE: {mae:.2f}")
        logging.info(f"  RMSE: {rmse:.2f}")
        logging.info(f"  R2: {r2:.3f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report"""
        logging.info("Starting evaluation report generation...")
        
        # Load test data
        df = self.load_test_data()
        if len(df) == 0:
            logging.error("No data available for evaluation")
            return None
            
        # Prepare features
        X = self.prepare_features(df)
        
        # Evaluate each model
        results = {}
        for target in ['points', 'assists', 'rebounds', 'efficiency']:
            if target in self.models:
                logging.info(f"\nEvaluating {target} predictions...")
                y = df[target]
                
                # Handle missing values
                y.fillna(y.mean(), inplace=True)
                
                results[target] = self.evaluate_model(target, X, y)
            else:
                logging.warning(f"No model found for {target}")
        
        return results

def main():
    logging.info("Starting enhanced model evaluation...")
    
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Initialize evaluator in test mode with absolute paths
    evaluator = EnhancedModelEvaluator(
        stats_db_path=os.path.join(project_root, 'Data', 'player_stats.sqlite'),
        models_dir=os.path.join(project_root, 'Models'),
        test_mode=True  # Use a smaller dataset for testing
    )
    
    # Generate evaluation report
    results = evaluator.generate_evaluation_report()
    
    if results:
        logging.info("\nEvaluation Summary:")
        for target, metrics in results.items():
            logging.info(f"\n{target.upper()} Model Performance:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.3f}")
    else:
        logging.error("Evaluation failed to generate results")

if __name__ == "__main__":
    main()
