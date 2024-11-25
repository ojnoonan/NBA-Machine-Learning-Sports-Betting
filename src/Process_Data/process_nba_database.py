import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class NBADataProcessor:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.source_db_path = os.path.join(project_root, 'Data', 'nba.sqlite')
        self.target_db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
        
    def load_game_data(self, start_date=None, end_date=None):
        """Load game data from the database"""
        conn = sqlite3.connect(self.source_db_path)
        
        query = """
        SELECT 
            g.game_id,
            g.game_date,
            g.season_id,
            g.season_type,
            -- Home team
            g.team_id_home,
            g.team_name_home,
            g.matchup_home,
            g.wl_home,
            g.pts_home,
            g.ast_home,
            g.reb_home,
            g.fg_pct_home,
            g.fg3_pct_home,
            g.ft_pct_home,
            g.oreb_home,
            g.dreb_home,
            g.stl_home,
            g.blk_home,
            g.tov_home,
            -- Away team
            g.team_id_away,
            g.team_name_away,
            g.matchup_away,
            g.wl_away,
            g.pts_away,
            g.ast_away,
            g.reb_away,
            g.fg_pct_away,
            g.fg3_pct_away,
            g.ft_pct_away,
            g.oreb_away,
            g.dreb_away,
            g.stl_away,
            g.blk_away,
            g.tov_away
        FROM game g
        WHERE g.season_type = 'Regular Season'
        """
        
        if start_date:
            query += f" AND g.game_date >= '{start_date}'"
        if end_date:
            query += f" AND g.game_date <= '{end_date}'"
            
        query += " ORDER BY g.game_date"
        
        games_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date column
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        
        return games_df
    
    def calculate_team_rolling_stats(self, games_df, window=10):
        """Calculate rolling statistics for each team"""
        team_stats = {}
        
        # Stats to calculate rolling averages for
        stats_cols = ['pts', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct', 
                     'oreb', 'dreb', 'stl', 'blk', 'tov']
        
        # Process home and away games
        for team_id in pd.concat([games_df['team_id_home'], games_df['team_id_away']]).unique():
            # Get all games for this team
            home_games = games_df[games_df['team_id_home'] == team_id]
            away_games = games_df[games_df['team_id_away'] == team_id]
            
            # Combine home and away games
            team_games = pd.DataFrame()
            all_stats = pd.DataFrame()
            
            for stat in stats_cols:
                # Home games
                home_stats = home_games[[f'{stat}_home']].copy()
                home_stats.columns = [stat]
                home_stats['game_date'] = home_games['game_date']
                home_stats['is_home'] = 1
                home_stats['win'] = (home_games['wl_home'] == 'W').astype(int)
                
                # Away games
                away_stats = away_games[[f'{stat}_away']].copy()
                away_stats.columns = [stat]
                away_stats['game_date'] = away_games['game_date']
                away_stats['is_home'] = 0
                away_stats['win'] = (away_games['wl_away'] == 'W').astype(int)
                
                # Combine and sort by date
                stat_data = pd.concat([home_stats, away_stats])
                stat_data = stat_data.sort_values('game_date')
                
                if all_stats.empty:
                    all_stats = stat_data[['game_date', 'is_home', 'win']]
                
                # Calculate rolling stats
                team_games[f'rolling_avg_{stat}'] = stat_data[stat].rolling(window=window, min_periods=3).mean()
                team_games[f'rolling_std_{stat}'] = stat_data[stat].rolling(window=window, min_periods=3).std()
            
            # Calculate win streaks and home/away performance
            team_games['win_streak'] = all_stats['win'].rolling(window=5, min_periods=1).sum()
            team_games['home_win_pct'] = all_stats.loc[all_stats['is_home'] == 1, 'win'].rolling(window=10, min_periods=1).mean()
            team_games['away_win_pct'] = all_stats.loc[all_stats['is_home'] == 0, 'win'].rolling(window=10, min_periods=1).mean()
            
            team_games['game_date'] = all_stats['game_date'].values
            team_stats[team_id] = team_games
            
        return team_stats
    
    def create_pregame_features(self, games_df, team_stats):
        """Create pre-game features for each game"""
        features = []
        
        for _, game in games_df.iterrows():
            game_date = game['game_date']
            home_team_id = game['team_id_home']
            away_team_id = game['team_id_away']
            
            # Get team stats before this game
            home_stats = team_stats[home_team_id][team_stats[home_team_id]['game_date'] < game_date]
            away_stats = team_stats[away_team_id][team_stats[away_team_id]['game_date'] < game_date]
            
            if not home_stats.empty and not away_stats.empty:
                home_prev_stats = home_stats.iloc[-1]
                away_prev_stats = away_stats.iloc[-1]
                
                # Calculate rest days
                home_last_game = home_stats['game_date'].iloc[-1]
                away_last_game = away_stats['game_date'].iloc[-1]
                home_rest_days = (game_date - home_last_game).days
                away_rest_days = (game_date - away_last_game).days
                
                # Create feature dictionary
                game_features = {
                    'game_id': game['game_id'],
                    'game_date': game_date,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_rest_days': home_rest_days,
                    'away_rest_days': away_rest_days,
                    'home_win_streak': home_prev_stats['win_streak'],
                    'away_win_streak': away_prev_stats['win_streak'],
                    'home_win_pct_home': home_prev_stats['home_win_pct'],
                    'away_win_pct_away': away_prev_stats['away_win_pct']
                }
                
                # Add rolling stats
                for stat in ['pts', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct', 
                           'oreb', 'dreb', 'stl', 'blk', 'tov']:
                    # Home team stats
                    game_features[f'home_avg_{stat}'] = home_prev_stats[f'rolling_avg_{stat}']
                    game_features[f'home_std_{stat}'] = home_prev_stats[f'rolling_std_{stat}']
                    
                    # Away team stats
                    game_features[f'away_avg_{stat}'] = away_prev_stats[f'rolling_avg_{stat}']
                    game_features[f'away_std_{stat}'] = away_prev_stats[f'rolling_std_{stat}']
                    
                    # Relative stats (home advantage/disadvantage)
                    game_features[f'rel_{stat}'] = (home_prev_stats[f'rolling_avg_{stat}'] - 
                                                  away_prev_stats[f'rolling_avg_{stat}'])
                
                features.append(game_features)
        
        return pd.DataFrame(features)
    
    def save_to_database(self, df, table_name):
        """Save DataFrame to SQLite database"""
        conn = sqlite3.connect(self.target_db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    
    def process_data(self, start_date=None, end_date=None):
        """Process NBA data and create features"""
        print("Loading game data...")
        games_df = self.load_game_data(start_date, end_date)
        
        print("Calculating team rolling statistics...")
        team_stats = self.calculate_team_rolling_stats(games_df)
        
        print("Creating pre-game features...")
        features_df = self.create_pregame_features(games_df, team_stats)
        
        print("Saving processed data...")
        self.save_to_database(features_df, 'pregame_features')
        
        print("Data processing complete!")
        return features_df

def main():
    processor = NBADataProcessor()
    
    # Process last 5 seasons of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    features_df = processor.process_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    print("\nFeature Statistics:")
    print(features_df.describe())
    
    print("\nSample Features:")
    print(features_df.head())

if __name__ == "__main__":
    main()
