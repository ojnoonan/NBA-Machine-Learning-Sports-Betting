import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3
import logging
import joblib
from datetime import datetime, timedelta

class PlayerPredictionModel:
    def __init__(self, stats_db_path):
        """Initialize the prediction model"""
        self.stats_db_path = stats_db_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_training_data(self, lookback_days=365):
        """Load and prepare training data from the player stats database"""
        try:
            # Calculate date range
            end_date = '2023-04-09'  # Latest date in our database
            start_date = '2019-11-27'  # Earliest date in our database
            
            # Connect to database
            conn = sqlite3.connect(self.stats_db_path)
            
            # Load rolling stats as features
            query = """
            WITH game_stats AS (
                SELECT 
                    game_id,
                    game_date,
                    player_id,
                    points,
                    assists,
                    rebounds_off + rebounds_def as total_rebounds,
                    points + rebounds_off + rebounds_def + assists + steals + blocks -
                    (fga - fgm) - (fta - ftm) - turnovers as efficiency
                FROM player_game_stats
            )
            SELECT 
                r.*,
                g.points as actual_points,
                g.assists as actual_assists,
                g.total_rebounds as actual_rebounds,
                g.efficiency as actual_efficiency
            FROM player_rolling_stats r
            JOIN game_stats g 
                ON r.player_id = g.player_id 
                AND r.game_date = g.game_date
            WHERE r.game_date BETWEEN ? AND ?
            ORDER BY r.game_date
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            conn.close()
            
            if df.empty:
                raise ValueError("No data found for the specified date range")
            
            logging.info(f"Loaded {len(df)} training samples")
            return df
            
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            raise
            
    def prepare_features(self, df, target='points'):
        """Prepare features and target for training"""
        # Select feature columns
        feature_columns = [
            'points_avg', 'assists_avg', 'rebounds_avg', 
            'steals_avg', 'blocks_avg', 'turnovers_avg',
            'fouls_avg', 'fg_pct', 'fg3_pct', 'ft_pct'
        ]
        
        # Map target names to actual columns
        target_map = {
            'points': 'actual_points',
            'assists': 'actual_assists',
            'rebounds': 'actual_rebounds',
            'efficiency': 'actual_efficiency'
        }
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_map[target]]
        
        return X, y
        
    def train_model(self, target='points', test_size=0.2, lookback_days=365):
        """Train the prediction model"""
        try:
            # Load and prepare data
            df = self.load_training_data(lookback_days)
            X, y = self.prepare_features(df, target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model Performance for {target}:")
            logging.info(f"Mean Squared Error: {mse:.2f}")
            logging.info(f"Mean Absolute Error: {mae:.2f}")
            logging.info(f"R² Score: {r2:.2f}")
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logging.info("\nTop 5 Important Features:")
            print(importance.head().to_string())
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'feature_importance': importance
            }
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
            
    def predict_player(self, player_id, target='points'):
        """Predict performance for a specific player"""
        try:
            # Load latest player stats
            conn = sqlite3.connect(self.stats_db_path)
            query = """
            SELECT *
            FROM player_rolling_stats
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT 1
            """
            
            latest_stats = pd.read_sql_query(query, conn, params=[player_id])
            conn.close()
            
            if latest_stats.empty:
                raise ValueError(f"No stats found for player {player_id}")
            
            # Prepare features
            X, _ = self.prepare_features(latest_stats, target)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            return {
                'player_id': player_id,
                'prediction': prediction,
                'target': target,
                'last_game_date': latest_stats['game_date'].iloc[0]
            }
            
        except Exception as e:
            logging.error(f"Error predicting for player {player_id}: {str(e)}")
            raise
            
    def save_model(self, path):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, path)
        logging.info(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load a trained model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        logging.info(f"Model loaded from {path}")

def main():
    """Main function to train and evaluate the model"""
    try:
        # Initialize model
        model = PlayerPredictionModel('/Users/olivernoonan/PythonProjects/NBA-Machine-Learning-Sports-Betting/Data/player_stats.sqlite')
        
        # Train models for different targets
        targets = ['points', 'assists', 'rebounds', 'efficiency']
        results = {}
        
        for target in targets:
            logging.info(f"\nTraining model for {target}...")
            results[target] = model.train_model(target=target)
            
            # Save model
            model.save_model(f'/Users/olivernoonan/PythonProjects/NBA-Machine-Learning-Sports-Betting/Models/{target}_prediction_model.joblib')
        
        # Print overall results
        logging.info("\nTraining Complete!")
        for target, metrics in results.items():
            logging.info(f"\n{target.upper()} Model Performance:")
            logging.info(f"Mean Absolute Error: {metrics['mae']:.2f}")
            logging.info(f"R² Score: {metrics['r2']:.2f}")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
