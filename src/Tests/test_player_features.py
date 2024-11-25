import os
import sys
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.Process_Data.Get_Player_Data import PlayerDataCollector

class TestPlayerFeatures(unittest.TestCase):
    def setUp(self):
        self.collector = PlayerDataCollector()
        self.season = "2023-24"
        self.date_from = "2023-10-23"
        self.date_to = datetime.now().strftime("%Y-%m-%d")
        
    def test_player_data_collection(self):
        """Test if we can collect player data successfully"""
        stats_df = self.collector.get_player_stats(self.date_from, self.date_to, self.season)
        self.assertIsNotNone(stats_df)
        self.assertTrue(len(stats_df) > 0)
        
        # Check required columns
        required_columns = ['PLAYER_ID', 'TEAM_ID', 'MIN', 'PTS', 'FGA']
        for col in required_columns:
            self.assertIn(col, stats_df.columns)
            
    def test_player_availability(self):
        """Test player availability data collection"""
        availability_df = self.collector.get_player_availability(self.date_from, self.season)
        self.assertIsNotNone(availability_df)
        self.assertTrue(len(availability_df) > 0)
        
    def test_team_stats_processing(self):
        """Test processing of team-level player stats"""
        stats_df = self.collector.get_player_stats(self.date_from, self.date_to, self.season)
        if stats_df is not None and len(stats_df) > 0:
            # Get first team_id from data
            team_id = stats_df['TEAM_ID'].iloc[0]
            team_stats = self.collector.process_team_player_stats(stats_df, team_id)
            
            self.assertIsNotNone(team_stats)
            self.assertIn('NUM_PLAYERS_AVAILABLE', team_stats)
            self.assertIn('TOP_PLAYERS_MINUTES', team_stats)
            self.assertIn('TOP_SCORERS_PPG', team_stats)
            self.assertIn('BENCH_SCORING', team_stats)
            self.assertIn('STARTER_EFFICIENCY', team_stats)
            
    def test_database_operations(self):
        """Test database operations"""
        stats_df = self.collector.get_player_stats(self.date_from, self.date_to, self.season)
        if stats_df is not None and len(stats_df) > 0:
            # Get first two teams for testing
            team1_id = stats_df['TEAM_ID'].iloc[0]
            team2_id = stats_df['TEAM_ID'].iloc[1]
            
            # Process stats
            home_stats = self.collector.process_team_player_stats(stats_df, team1_id)
            away_stats = self.collector.process_team_player_stats(stats_df, team2_id)
            
            if home_stats and away_stats:
                stats_dict = {
                    'home': home_stats,
                    'away': away_stats
                }
                
                # Test saving to database
                test_date = datetime.now().strftime("%Y-%m-%d")
                self.collector.save_to_database(test_date, team1_id, team2_id, stats_dict)
                
                # Verify data was saved
                import sqlite3
                conn = sqlite3.connect('../../Data/PlayerData.sqlite')
                df = pd.read_sql_query(f"select * from player_stats_{test_date}", conn)
                conn.close()
                
                self.assertTrue(len(df) > 0)
                self.assertEqual(df['HOME_TEAM_ID'].iloc[0], team1_id)
                self.assertEqual(df['AWAY_TEAM_ID'].iloc[0], team2_id)

def main():
    # Run the tests
    unittest.main()

if __name__ == '__main__':
    main()
