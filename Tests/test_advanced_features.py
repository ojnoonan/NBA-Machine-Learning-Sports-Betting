import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.Process_Data.Advanced_Features import AdvancedFeatureEngineering
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    raise

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Game_Date': pd.date_range(start='2023-01-01', periods=10),
            'Home_Team_ID': [1, 1, 2, 2, 1, 2, 1, 2, 1, 2],
            'Away_Team_ID': [2, 3, 1, 3, 3, 1, 2, 1, 3, 3],
            'Home_Team_Win': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            'PTS': [100, 105, 98, 110, 95, 102, 108, 96, 112, 99],
            'FG_PCT': [0.45, 0.48, 0.42, 0.50, 0.44, 0.47, 0.49, 0.43, 0.51, 0.46],
            'is_home': [True] * 10
        })
        
        self.feature_engineer = AdvancedFeatureEngineering()

    def test_rest_days_calculation(self):
        result = self.feature_engineer.add_advanced_features(self.sample_data)
        
        # Check if DAYS_REST is calculated correctly
        self.assertIn('DAYS_REST', result.columns)
        # First game should have mean rest days
        self.assertFalse(pd.isna(result['DAYS_REST'].iloc[0]))
        # Rest days should be positive numbers
        self.assertTrue(all(result['DAYS_REST'] >= 0))

    def test_back_to_back_indicator(self):
        result = self.feature_engineer.add_advanced_features(self.sample_data)
        
        self.assertIn('IS_BACK_TO_BACK', result.columns)
        # Check if back-to-back is binary
        self.assertTrue(set(result['IS_BACK_TO_BACK'].unique()).issubset({0, 1}))

    def test_streak_calculation(self):
        result = self.feature_engineer.add_advanced_features(self.sample_data)
        
        self.assertIn('WIN_STREAK', result.columns)
        self.assertIn('LOSS_STREAK', result.columns)
        # Check if streaks are non-negative
        self.assertTrue(all(result['WIN_STREAK'] >= 0))
        self.assertTrue(all(result['LOSS_STREAK'] >= 0))
        # Check if a team can't have both win and loss streaks
        self.assertTrue(all((result['WIN_STREAK'] == 0) | (result['LOSS_STREAK'] == 0)))

    def test_matchup_history(self):
        result = self.feature_engineer.add_advanced_features(self.sample_data)
        
        self.assertIn('MATCHUP_ID', result.columns)
        self.assertIn('HOME_TEAM_MATCHUP_WIN_PCT', result.columns)
        # Check if win percentages are between 0 and 1
        self.assertTrue(all((0 <= result['HOME_TEAM_MATCHUP_WIN_PCT']) & 
                          (result['HOME_TEAM_MATCHUP_WIN_PCT'] <= 1)))

    def test_fatigue_indicators(self):
        result = self.feature_engineer.add_advanced_features(self.sample_data)
        
        self.assertIn('GAMES_LAST_7D', result.columns)
        # Check if games in last 7 days is non-negative
        self.assertTrue(all(result['GAMES_LAST_7D'] >= 0))
        # Check if games in last 7 days is reasonable (shouldn't be more than 4-5 games)
        self.assertTrue(all(result['GAMES_LAST_7D'] <= 5))

if __name__ == '__main__':
    unittest.main()
