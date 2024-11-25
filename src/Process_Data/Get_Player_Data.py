import os
import sys
import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class PlayerDataCollector:
    def __init__(self):
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.player_stats_url = "https://stats.nba.com/stats/leaguedashplayerstats"
        self.player_status_url = "https://stats.nba.com/stats/commonallplayers"
        self.db_path = os.path.join(project_root, 'Data', 'PlayerData.sqlite')
        
    def get_player_stats(self, date_from, date_to, season):
        """Fetch player statistics for a given date range"""
        params = {
            'DateFrom': date_from,
            'DateTo': date_to,
            'LastNGames': '0',
            'LeagueID': '00',
            'MeasureType': 'Base',
            'Month': '0',
            'OpponentTeamID': '0',
            'PORound': '0',
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': '0',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': season,
            'SeasonType': 'Regular Season',
            'TeamID': '0'
        }
        
        try:
            response = requests.get(self.player_stats_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract headers and rows
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Add advanced metrics
            df['SCORING_EFFICIENCY'] = df['PTS'] / df['FGA'] if df['FGA'].sum() > 0 else 0
            df['MINUTES_IMPACT'] = df['PLUS_MINUS'] / df['MIN'] if df['MIN'].sum() > 0 else 0
            
            return df
            
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return None

    def process_team_player_stats(self, stats_df, team_id):
        """Process player stats for a specific team with advanced metrics"""
        team_players = stats_df[stats_df['TEAM_ID'] == team_id]
        
        if team_players.empty:
            return None
            
        # Calculate advanced team-level aggregates
        team_stats = {
            'NUM_PLAYERS_AVAILABLE': len(team_players),
            'TOP_PLAYERS_MINUTES': team_players.nlargest(5, 'MIN')['MIN'].mean(),
            'TOP_SCORERS_PPG': team_players.nlargest(3, 'PTS')['PTS'].mean(),
            'BENCH_SCORING': team_players.nsmallest(len(team_players)-5, 'MIN')['PTS'].mean(),
            'STARTER_EFFICIENCY': team_players.nlargest(5, 'MIN')['SCORING_EFFICIENCY'].mean(),
            'TEAM_PLUS_MINUS': team_players['PLUS_MINUS'].mean(),
            'MINUTES_DISTRIBUTION': team_players['MIN'].std(),  # How evenly minutes are distributed
            'SCORING_CONSISTENCY': team_players['PTS'].std(),  # Variation in scoring
            'BENCH_IMPACT': team_players.nsmallest(len(team_players)-5, 'MIN')['MINUTES_IMPACT'].mean()
        }
        
        return team_stats

    def save_to_database(self, date, home_team_id, away_team_id, stats_dict):
        """Save processed player stats to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.DataFrame([{
                'Date': date,
                'HOME_TEAM_ID': home_team_id,
                'AWAY_TEAM_ID': away_team_id,
                'HOME_NUM_PLAYERS': stats_dict['home']['NUM_PLAYERS_AVAILABLE'],
                'HOME_TOP_PLAYERS_MIN': stats_dict['home']['TOP_PLAYERS_MINUTES'],
                'HOME_TOP_SCORERS': stats_dict['home']['TOP_SCORERS_PPG'],
                'HOME_BENCH_SCORING': stats_dict['home']['BENCH_SCORING'],
                'HOME_STARTER_EFF': stats_dict['home']['STARTER_EFFICIENCY'],
                'HOME_PLUS_MINUS': stats_dict['home']['TEAM_PLUS_MINUS'],
                'HOME_MIN_DIST': stats_dict['home']['MINUTES_DISTRIBUTION'],
                'HOME_SCORING_CONS': stats_dict['home']['SCORING_CONSISTENCY'],
                'HOME_BENCH_IMPACT': stats_dict['home']['BENCH_IMPACT'],
                'AWAY_NUM_PLAYERS': stats_dict['away']['NUM_PLAYERS_AVAILABLE'],
                'AWAY_TOP_PLAYERS_MIN': stats_dict['away']['TOP_PLAYERS_MINUTES'],
                'AWAY_TOP_SCORERS': stats_dict['away']['TOP_SCORERS_PPG'],
                'AWAY_BENCH_SCORING': stats_dict['away']['BENCH_SCORING'],
                'AWAY_STARTER_EFF': stats_dict['away']['STARTER_EFFICIENCY'],
                'AWAY_PLUS_MINUS': stats_dict['away']['TEAM_PLUS_MINUS'],
                'AWAY_MIN_DIST': stats_dict['away']['MINUTES_DISTRIBUTION'],
                'AWAY_SCORING_CONS': stats_dict['away']['SCORING_CONSISTENCY'],
                'AWAY_BENCH_IMPACT': stats_dict['away']['BENCH_IMPACT']
            }])
            
            df.to_sql(f'player_stats_{date}', conn, if_exists='replace', index=False)
            print(f"Successfully saved player stats for {date}")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            
        finally:
            conn.close()

def main():
    collector = PlayerDataCollector()
    
    # Example usage
    season = "2023-24"
    date_from = "2023-10-23"
    date_to = datetime.now().strftime("%Y-%m-%d")
    
    # Get player stats
    stats_df = collector.get_player_stats(date_from, date_to, season)
    if stats_df is not None:
        print(f"Successfully collected player stats for {len(stats_df)} players")
    
    # Example processing for a specific game
    example_game_date = "2023-10-24"
    home_team_id = 1610612744  # Example team ID
    away_team_id = 1610612756  # Example team ID
    
    if stats_df is not None:
        home_stats = collector.process_team_player_stats(stats_df, home_team_id)
        away_stats = collector.process_team_player_stats(stats_df, away_team_id)
        
        if home_stats and away_stats:
            stats_dict = {
                'home': home_stats,
                'away': away_stats
            }
            collector.save_to_database(example_game_date, home_team_id, away_team_id, stats_dict)

if __name__ == "__main__":
    main()
