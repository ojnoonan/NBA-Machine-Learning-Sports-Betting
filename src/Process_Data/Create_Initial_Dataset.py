import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class InitialDatasetCreator:
    def __init__(self):
        self.db_path = os.path.join(project_root, 'Data', 'dataset.sqlite')
        
    def get_season_games(self, season_year):
        """Get all games for a season"""
        season = f"{season_year}-{str(season_year + 1)[-2:]}"
        
        try:
            print(f"Fetching all games for season {season}...")
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00',
                season_type_nullable='Regular Season'
            )
            time.sleep(1)  # Respect API rate limits
            
            games_df = pd.DataFrame(gamefinder.get_data_frames()[0])
            if games_df.empty:
                print(f"No games found for season {season}")
                return pd.DataFrame()
                
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE')
            
            print(f"Found {len(games_df)} game records for season {season}")
            return games_df
            
        except Exception as e:
            print(f"Error getting season games: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_stats(self, team_games):
        """Calculate rolling statistics for a team's games"""
        # Base stats to calculate rolling averages and std devs for
        stats_cols = [
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
            'TOV', 'STL', 'BLK', 'PTS'
        ]
        
        rolling_stats = {}
        
        # Calculate rolling averages and standard deviations (10 game window)
        window = 10
        for col in stats_cols:
            rolling_stats[f'ROLLING_AVG_{col}'] = team_games[col].rolling(window=window, min_periods=3).mean()
            rolling_stats[f'ROLLING_STD_{col}'] = team_games[col].rolling(window=window, min_periods=3).std()
        
        # Calculate win/loss streaks
        team_games['WIN'] = (team_games['WL'] == 'W').astype(int)
        rolling_stats['WIN_STREAK'] = team_games['WIN'].rolling(window=5, min_periods=1).sum()
        
        # Calculate home/away splits
        team_games['IS_HOME'] = (~team_games['MATCHUP'].str.contains('@')).astype(int)
        rolling_stats['HOME_WIN_PCT'] = team_games['WIN'][team_games['IS_HOME'] == 1].rolling(window=10, min_periods=1).mean()
        rolling_stats['AWAY_WIN_PCT'] = team_games['WIN'][team_games['IS_HOME'] == 0].rolling(window=10, min_periods=1).mean()
        
        return pd.DataFrame(rolling_stats, index=team_games.index)
    
    def process_season_games(self, season_games):
        """Process all games in a season"""
        if season_games.empty:
            return pd.DataFrame()
            
        processed_games = []
        unique_games = season_games.drop_duplicates(subset=['GAME_DATE', 'MATCHUP'])
        total_games = len(unique_games)  # Count only unique games
        processed = 0
        
        print(f"Processing {total_games} games...")
        
        # Group games by team for efficient rolling calculations
        team_stats = {}
        for team_id in season_games['TEAM_ID'].unique():
            team_games = season_games[season_games['TEAM_ID'] == team_id].sort_values('GAME_DATE')
            team_stats[team_id] = pd.concat([team_games, self.calculate_rolling_stats(team_games)], axis=1)
        
        # Process each unique game
        for _, game in unique_games.iterrows():
            processed += 1
            if processed % 50 == 0:
                print(f"Progress: {processed}/{total_games} games processed")
            
            game_date = game['GAME_DATE']
            is_home = '@' not in game['MATCHUP']
            
            if is_home:
                home_team_id = game['TEAM_ID']
                # Extract away team ID from matchup
                away_team_id = season_games[
                    (season_games['GAME_DATE'] == game_date) & 
                    (season_games['TEAM_ID'] != home_team_id)
                ]['TEAM_ID'].iloc[0]
                
                # Get stats from games before this one
                home_prev_games = team_stats[home_team_id][team_stats[home_team_id]['GAME_DATE'] < game_date]
                away_prev_games = team_stats[away_team_id][team_stats[away_team_id]['GAME_DATE'] < game_date]
                
                if not home_prev_games.empty and not away_prev_games.empty:
                    home_prev_stats = home_prev_games.iloc[-1]
                    away_prev_stats = away_prev_games.iloc[-1]
                    
                    # Get the most recent game dates for both teams
                    home_last_game = home_prev_games.iloc[-1]['GAME_DATE']
                    away_last_game = away_prev_games.iloc[-1]['GAME_DATE']
                    
                    # Calculate rest days as days since last game
                    home_rest_days = (game_date - home_last_game).days
                    away_rest_days = (game_date - away_last_game).days
                    
                    # Calculate relative statistics (home team advantage/disadvantage)
                    relative_stats = {
                        'REL_PTS': home_prev_stats['ROLLING_AVG_PTS'] - away_prev_stats['ROLLING_AVG_PTS'],
                        'REL_FG_PCT': home_prev_stats['ROLLING_AVG_FG_PCT'] - away_prev_stats['ROLLING_AVG_FG_PCT'],
                        'REL_FG3_PCT': home_prev_stats['ROLLING_AVG_FG3_PCT'] - away_prev_stats['ROLLING_AVG_FG3_PCT'],
                        'REL_FT_PCT': home_prev_stats['ROLLING_AVG_FT_PCT'] - away_prev_stats['ROLLING_AVG_FT_PCT'],
                        'REL_REB': home_prev_stats['ROLLING_AVG_REB'] - away_prev_stats['ROLLING_AVG_REB'],
                        'REL_AST': home_prev_stats['ROLLING_AVG_AST'] - away_prev_stats['ROLLING_AVG_AST'],
                        'REL_TOV': home_prev_stats['ROLLING_AVG_TOV'] - away_prev_stats['ROLLING_AVG_TOV'],
                        'REL_STL': home_prev_stats['ROLLING_AVG_STL'] - away_prev_stats['ROLLING_AVG_STL'],
                        'REL_BLK': home_prev_stats['ROLLING_AVG_BLK'] - away_prev_stats['ROLLING_AVG_BLK']
                    }
                    
                    game_dict = {
                        'Date': game_date.strftime('%Y-%m-%d'),
                        'Home_Team_ID': home_team_id,
                        'Away_Team_ID': away_team_id,
                        'Home_Team_Win': 1 if game['WL'] == 'W' else 0,
                        
                        # Pre-game rolling averages - Home team
                        'HOME_AVG_PTS': home_prev_stats['ROLLING_AVG_PTS'],
                        'HOME_AVG_FG_PCT': home_prev_stats['ROLLING_AVG_FG_PCT'],
                        'HOME_AVG_FG3_PCT': home_prev_stats['ROLLING_AVG_FG3_PCT'],
                        'HOME_AVG_FT_PCT': home_prev_stats['ROLLING_AVG_FT_PCT'],
                        'HOME_AVG_REB': home_prev_stats['ROLLING_AVG_REB'],
                        'HOME_AVG_AST': home_prev_stats['ROLLING_AVG_AST'],
                        'HOME_AVG_TOV': home_prev_stats['ROLLING_AVG_TOV'],
                        'HOME_AVG_STL': home_prev_stats['ROLLING_AVG_STL'],
                        'HOME_AVG_BLK': home_prev_stats['ROLLING_AVG_BLK'],
                        
                        # Pre-game rolling standard deviations - Home team
                        'HOME_STD_PTS': home_prev_stats['ROLLING_STD_PTS'],
                        'HOME_STD_FG_PCT': home_prev_stats['ROLLING_STD_FG_PCT'],
                        'HOME_STD_FG3_PCT': home_prev_stats['ROLLING_STD_FG3_PCT'],
                        'HOME_STD_FT_PCT': home_prev_stats['ROLLING_STD_FT_PCT'],
                        'HOME_STD_REB': home_prev_stats['ROLLING_STD_REB'],
                        'HOME_STD_AST': home_prev_stats['ROLLING_STD_AST'],
                        'HOME_STD_TOV': home_prev_stats['ROLLING_STD_TOV'],
                        'HOME_STD_STL': home_prev_stats['ROLLING_STD_STL'],
                        'HOME_STD_BLK': home_prev_stats['ROLLING_STD_BLK'],
                        
                        # Pre-game rolling averages - Away team
                        'AWAY_AVG_PTS': away_prev_stats['ROLLING_AVG_PTS'],
                        'AWAY_AVG_FG_PCT': away_prev_stats['ROLLING_AVG_FG_PCT'],
                        'AWAY_AVG_FG3_PCT': away_prev_stats['ROLLING_AVG_FG3_PCT'],
                        'AWAY_AVG_FT_PCT': away_prev_stats['ROLLING_AVG_FT_PCT'],
                        'AWAY_AVG_REB': away_prev_stats['ROLLING_AVG_REB'],
                        'AWAY_AVG_AST': away_prev_stats['ROLLING_AVG_AST'],
                        'AWAY_AVG_TOV': away_prev_stats['ROLLING_AVG_TOV'],
                        'AWAY_AVG_STL': away_prev_stats['ROLLING_AVG_STL'],
                        'AWAY_AVG_BLK': away_prev_stats['ROLLING_AVG_BLK'],
                        
                        # Pre-game rolling standard deviations - Away team
                        'AWAY_STD_PTS': away_prev_stats['ROLLING_STD_PTS'],
                        'AWAY_STD_FG_PCT': away_prev_stats['ROLLING_STD_FG_PCT'],
                        'AWAY_STD_FG3_PCT': away_prev_stats['ROLLING_STD_FG3_PCT'],
                        'AWAY_STD_FT_PCT': away_prev_stats['ROLLING_STD_FT_PCT'],
                        'AWAY_STD_REB': away_prev_stats['ROLLING_STD_REB'],
                        'AWAY_STD_AST': away_prev_stats['ROLLING_STD_AST'],
                        'AWAY_STD_TOV': away_prev_stats['ROLLING_STD_TOV'],
                        'AWAY_STD_STL': away_prev_stats['ROLLING_STD_STL'],
                        'AWAY_STD_BLK': away_prev_stats['ROLLING_STD_BLK'],
                        
                        # Team form features
                        'HOME_WIN_STREAK': home_prev_stats['WIN_STREAK'],
                        'AWAY_WIN_STREAK': away_prev_stats['WIN_STREAK'],
                        'HOME_WIN_PCT': home_prev_stats['HOME_WIN_PCT'],
                        'AWAY_WIN_PCT': away_prev_stats['AWAY_WIN_PCT'],
                        'HOME_REST_DAYS': home_rest_days,  # Use directly calculated rest days
                        'AWAY_REST_DAYS': away_rest_days,  # Use directly calculated rest days
                        
                        # Relative statistics
                        **relative_stats
                    }
                    
                    processed_games.append(game_dict)
        
        print(f"Successfully processed {len(processed_games)} games with complete statistics")
        return pd.DataFrame(processed_games)
    
    def create_dataset(self, start_year=2012, end_year=2024):
        """Create the initial dataset with pre-game statistics"""
        all_games = []
        
        for year in range(start_year, end_year):
            print(f"\nProcessing season {year}-{year+1}...")
            
            # Get all games for the season
            season_games = self.get_season_games(year)
            if not season_games.empty:
                processed_games = self.process_season_games(season_games)
                if not processed_games.empty:
                    all_games.append(processed_games)
            
            print(f"Season {year}-{year+1} completed.")
            time.sleep(2)  # Respect API rate limits between seasons
        
        if not all_games:
            print("\nNo games were processed successfully")
            return None
            
        final_dataset = pd.concat(all_games, ignore_index=True)
        
        # Remove any rows with missing values instead of imputing
        initial_rows = len(final_dataset)
        final_dataset = final_dataset.dropna()
        removed_rows = initial_rows - len(final_dataset)
        if removed_rows > 0:
            print(f"\nRemoved {removed_rows} rows with missing values")
        
        # Ensure proper data types for SQLite
        dtypes = {
            'Date': 'str',
            'Home_Team_ID': 'int64',
            'Away_Team_ID': 'int64',
            'Home_Team_Win': 'int64'
        }
        
        # All other columns should be float64
        for col in final_dataset.columns:
            if col not in dtypes:
                dtypes[col] = 'float64'
        
        # Convert all columns to their specified types
        for col, dtype in dtypes.items():
            if dtype == 'str':
                final_dataset[col] = final_dataset[col].astype(str)
            elif dtype == 'int64':
                final_dataset[col] = pd.to_numeric(final_dataset[col], errors='coerce').astype('int64')
            else:
                final_dataset[col] = pd.to_numeric(final_dataset[col], errors='coerce').astype('float64')
        
        # Save to database with explicit dtype mapping
        print("\nSaving dataset to database...")
        conn = sqlite3.connect(self.db_path)
        final_dataset.to_sql('dataset_2012_24', conn, if_exists='replace', index=False, dtype={
            'Date': 'TEXT',
            'Home_Team_ID': 'INTEGER',
            'Away_Team_ID': 'INTEGER',
            'Home_Team_Win': 'INTEGER'
        })
        conn.close()
        
        print(f"\nDataset created successfully with {len(final_dataset)} total games")
        return final_dataset

def main():
    creator = InitialDatasetCreator()
    dataset = creator.create_dataset()
    
    if dataset is not None:
        # Print some basic statistics
        print("\nDataset Statistics:")
        print(f"Total Games: {len(dataset)}")
        print(f"Date Range: {dataset['Date'].min()} to {dataset['Date'].max()}")
        print(f"Home Team Win Rate: {dataset['Home_Team_Win'].mean():.3f}")
        
        # Verify data quality
        print("\nMissing Values:")
        print(dataset.isnull().sum())
        
        # Print correlation with target (excluding non-numeric columns)
        numeric_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
        correlations = dataset[numeric_cols].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nTop 10 Feature Correlations with Home Team Win:")
        print(correlations.head(10))
        
        print("\nBottom 10 Feature Correlations with Home Team Win:")
        print(correlations.tail(10))
        
        # Print relative feature correlations
        rel_cols = [col for col in numeric_cols if col.startswith('REL_')]
        rel_correlations = dataset[rel_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nRelative Feature Correlations with Home Team Win:")
        print(rel_correlations)
        
        # Print form feature correlations
        form_cols = ['HOME_WIN_STREAK', 'AWAY_WIN_STREAK', 'HOME_WIN_PCT', 
                    'AWAY_WIN_PCT', 'HOME_REST_DAYS', 'AWAY_REST_DAYS']
        form_correlations = dataset[form_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nForm Feature Correlations with Home Team Win:")
        print(form_correlations)
        
        # New: Standard Deviation Feature Correlations
        std_cols = [col for col in numeric_cols if 'STD_' in col]
        std_correlations = dataset[std_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nStandard Deviation Feature Correlations with Home Team Win:")
        print(std_correlations)
        
        # New: Home Team Performance Metrics
        home_cols = [col for col in numeric_cols if col.startswith('HOME_AVG_')]
        home_correlations = dataset[home_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nHome Team Average Performance Correlations with Home Team Win:")
        print(home_correlations)
        
        # New: Away Team Performance Metrics
        away_cols = [col for col in numeric_cols if col.startswith('AWAY_AVG_')]
        away_correlations = dataset[away_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nAway Team Average Performance Correlations with Home Team Win:")
        print(away_correlations)
        
        # New: Create and analyze absolute difference features
        for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK']:
            dataset[f'ABS_DIFF_{stat}'] = abs(
                dataset[f'HOME_AVG_{stat}'] - dataset[f'AWAY_AVG_{stat}']
            )
        
        abs_diff_cols = [col for col in dataset.columns if col.startswith('ABS_DIFF_')]
        abs_diff_correlations = dataset[abs_diff_cols + ['Home_Team_Win']].corr()['Home_Team_Win'].sort_values(ascending=False)
        
        print("\nAbsolute Difference Feature Correlations with Home Team Win:")
        print(abs_diff_correlations)
        
if __name__ == "__main__":
    main()
