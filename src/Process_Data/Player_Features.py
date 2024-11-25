import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime, timedelta

class PlayerFeatureGenerator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.window_sizes = [5, 10, 15, 20]  # Rolling window sizes for player stats
        self.positions = ['PG', 'SG', 'SF', 'PF', 'C']
        self.position_weights = {  # Importance weights for different positions
            'PG': {'assists': 1.5, 'steals': 1.3, 'turnovers': 1.2},
            'SG': {'points': 1.3, 'fg3_made': 1.4, 'steals': 1.2},
            'SF': {'points': 1.2, 'rebounds': 1.1, 'fg_pct': 1.2},
            'PF': {'rebounds': 1.4, 'blocks': 1.3, 'fg_pct': 1.3},
            'C': {'rebounds': 1.5, 'blocks': 1.5, 'fg_pct': 1.4}
        }
        
    def get_player_data(self) -> pd.DataFrame:
        """Load player game logs and injury data"""
        conn = sqlite3.connect(self.db_path)
        
        # Load player game logs
        player_stats = pd.read_sql("""
            SELECT 
                player_id,
                game_id,
                team_id,
                game_date,
                starter,
                minutes_played,
                points,
                rebounds,
                assists,
                steals,
                blocks,
                turnovers,
                personal_fouls,
                fg_made,
                fg_attempted,
                fg3_made,
                fg3_attempted,
                ft_made,
                ft_attempted,
                plus_minus,
                position
            FROM player_game_logs
        """, conn)
        
        # Load injury data
        injuries = pd.read_sql("""
            SELECT 
                player_id,
                injury_date,
                return_date,
                injury_type,
                games_missed
            FROM player_injuries
        """, conn)
        
        conn.close()
        return player_stats, injuries
    
    def calculate_player_recent_performance(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages of player performance metrics"""
        metrics = [
            'points', 'rebounds', 'assists', 'steals', 'blocks', 
            'turnovers', 'personal_fouls', 'plus_minus'
        ]
        
        shooting_metrics = {
            'fg_pct': ('fg_made', 'fg_attempted'),
            'fg3_pct': ('fg3_made', 'fg3_attempted'),
            'ft_pct': ('ft_made', 'ft_attempted')
        }
        
        performance_data = []
        
        for player_id in player_stats['player_id'].unique():
            player_games = player_stats[player_stats['player_id'] == player_id].sort_values('game_date')
            
            for window in self.window_sizes:
                # Basic stats rolling averages
                rolling_stats = player_games[metrics].rolling(window=window, min_periods=1).mean()
                rolling_stats.columns = [f'{col}_{window}g_avg' for col in rolling_stats.columns]
                
                # Shooting percentages
                for metric, (made, attempted) in shooting_metrics.items():
                    rolling_made = player_games[made].rolling(window=window, min_periods=1).sum()
                    rolling_attempted = player_games[attempted].rolling(window=window, min_periods=1).sum()
                    rolling_stats[f'{metric}_{window}g'] = rolling_made / rolling_attempted
                
                # Minutes played trend
                rolling_stats[f'minutes_{window}g_avg'] = player_games['minutes_played'].rolling(window=window, min_periods=1).mean()
                
                # Performance variability
                rolling_stats[f'performance_volatility_{window}g'] = player_games['plus_minus'].rolling(window=window, min_periods=1).std()
                
                player_data = pd.concat([player_games[['player_id', 'game_id', 'team_id', 'game_date', 'position']], 
                                      rolling_stats], axis=1)
                performance_data.append(player_data)
        
        return pd.concat(performance_data)
    
    def calculate_fatigue_metrics(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate player fatigue metrics based on recent workload"""
        result_df = player_stats.copy()
        
        # Time windows for fatigue analysis (in days)
        fatigue_windows = [3, 7, 14, 30]
        
        for player_id in result_df['player_id'].unique():
            player_games = result_df[result_df['player_id'] == player_id].sort_values('game_date')
            
            for window in fatigue_windows:
                # Convert game_date to datetime if it's not already
                player_games['game_date'] = pd.to_datetime(player_games['game_date'])
                
                # Calculate games and minutes in rolling window
                player_games[f'games_last_{window}d'] = player_games['game_date'].rolling(
                    window=f'{window}D'
                ).count()
                
                player_games[f'minutes_last_{window}d'] = player_games['minutes_played'].rolling(
                    window=f'{window}D'
                ).sum()
                
                # Calculate average minutes per game in window
                player_games[f'avg_minutes_last_{window}d'] = (
                    player_games[f'minutes_last_{window}d'] / 
                    player_games[f'games_last_{window}d']
                )
                
                # Calculate back-to-back games
                if window >= 2:
                    player_games['days_since_last_game'] = (
                        player_games['game_date'] - 
                        player_games['game_date'].shift(1)
                    ).dt.days
                    
                    player_games['is_back_to_back'] = (
                        player_games['days_since_last_game'] == 1
                    ).astype(int)
                    
                    # Rolling sum of back-to-backs
                    player_games[f'back_to_backs_last_{window}d'] = (
                        player_games['is_back_to_back'].rolling(
                            window=f'{window}D'
                        ).sum()
                    )
                
                # Calculate fatigue score (weighted combination of metrics)
                player_games[f'fatigue_score_{window}d'] = (
                    0.4 * (player_games[f'minutes_last_{window}d'] / (48 * window)) +
                    0.3 * (player_games[f'games_last_{window}d'] / window) +
                    0.3 * (player_games['back_to_backs_last_{window}d'] if window >= 2 else 0)
                )
            
            # Update the main dataframe
            result_df.loc[player_games.index] = player_games
        
        return result_df
    
    def calculate_position_matchup_stats(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-specific matchup statistics"""
        matchup_stats = []
        
        for position in self.positions:
            position_stats = player_stats[player_stats['position'] == position]
            
            # Calculate league averages for the position
            position_avgs = position_stats[['points', 'rebounds', 'assists', 'plus_minus']].mean()
            
            # Calculate player performance vs. position average
            for player_id in position_stats['player_id'].unique():
                player_games = position_stats[position_stats['player_id'] == player_id]
                
                for metric in ['points', 'rebounds', 'assists', 'plus_minus']:
                    vs_position_avg = player_games[metric] - position_avgs[metric]
                    player_games[f'{metric}_vs_pos_avg'] = vs_position_avg
                
                matchup_stats.append(player_games)
        
        return pd.concat(matchup_stats)
    
    def calculate_defender_matchups(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate player vs specific defender performance metrics"""
        result_df = player_stats.copy()
        matchup_windows = [5, 10, 15]  # Last N matchups to consider
        
        # Create a unique game-player-defender ID
        result_df['matchup_id'] = result_df.apply(
            lambda row: f"{row['game_id']}_{min(row['player_id'], row['defender_id'])}_{max(row['player_id'], row['defender_id'])}",
            axis=1
        )
        
        for player_id in result_df['player_id'].unique():
            player_matchups = result_df[
                (result_df['player_id'] == player_id) | 
                (result_df['defender_id'] == player_id)
            ].sort_values('game_date')
            
            for window in matchup_windows:
                # Offensive success metrics
                player_matchups[f'pts_vs_defender_{window}g'] = player_matchups.groupby(
                    'defender_id'
                )['points'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                
                player_matchups[f'fg_pct_vs_defender_{window}g'] = player_matchups.groupby(
                    'defender_id'
                ).apply(
                    lambda x: (x['fg_made'].rolling(window=window, min_periods=1).sum() /
                             x['fg_attempted'].rolling(window=window, min_periods=1).sum())
                )
                
                # Defensive success metrics
                player_matchups[f'stops_vs_player_{window}g'] = player_matchups.groupby(
                    'player_id'
                ).apply(
                    lambda x: ((x['steals'] + x['blocks']).rolling(window=window, min_periods=1).mean())
                )
                
            # Update the main dataframe
            result_df.loc[player_matchups.index] = player_matchups
        
        return result_df
    
    def calculate_player_chemistry(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate player chemistry metrics based on time played together"""
        result_df = player_stats.copy()
        chemistry_windows = [10, 20, 40]  # Games to consider for chemistry
        
        # Group players by team and game
        team_games = result_df.groupby(['team_id', 'game_id', 'game_date'])
        
        for team_id in result_df['team_id'].unique():
            team_data = result_df[result_df['team_id'] == team_id].sort_values('game_date')
            
            # Get all players who played together
            lineups = team_data.groupby('game_id').agg({
                'player_id': lambda x: list(x),
                'minutes_played': lambda x: list(x)
            })
            
            # Calculate chemistry metrics for each game
            for window in chemistry_windows:
                # Minutes played together
                lineups[f'lineup_minutes_{window}g'] = lineups['minutes_played'].rolling(
                    window=window, min_periods=1
                ).apply(lambda x: sum(min(a, b) for a, b in zip(x[0], x[1:])))
                
                # Lineup consistency (% of minutes with same players)
                lineups[f'lineup_consistency_{window}g'] = lineups['player_id'].rolling(
                    window=window, min_periods=1
                ).apply(lambda x: len(set(x[0]).intersection(*x[1:])) / len(set(x[0])))
            
            # Map back to individual players
            for window in chemistry_windows:
                team_data[f'team_chemistry_{window}g'] = team_data['game_id'].map(
                    lineups[f'lineup_consistency_{window}g']
                )
            
            # Update the main dataframe
            result_df.loc[team_data.index] = team_data
        
        return result_df
    
    def analyze_injury_impact(self, injuries: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Detailed analysis of injury impact based on injury type and severity"""
        result_df = player_stats.copy()
        
        # Define injury severity weights
        injury_severity = {
            'Minor': 0.2,
            'Moderate': 0.5,
            'Severe': 1.0
        }
        
        # Define recovery impact by injury type
        injury_recovery = {
            'Ankle': 0.7,
            'Knee': 0.9,
            'Back': 0.8,
            'Shoulder': 0.6,
            'Hamstring': 0.75,
            'Concussion': 0.85,
            'Other': 0.5
        }
        
        def calculate_detailed_injury_impact(row):
            game_date = pd.to_datetime(row['game_date'])
            player_injuries = injuries[injuries['player_id'] == row['player_id']]
            
            # Recent injuries analysis
            recent_injuries = player_injuries[
                (pd.to_datetime(player_injuries['injury_date']) > game_date - timedelta(days=90)) &
                (pd.to_datetime(player_injuries['injury_date']) < game_date)
            ]
            
            if recent_injuries.empty:
                return pd.Series({
                    'injury_severity_impact': 0,
                    'injury_recovery_impact': 0,
                    'cumulative_injury_impact': 0,
                    'games_since_injury': 999,
                    'expected_performance_impact': 0
                })
            
            # Calculate various injury impacts
            severity_impact = sum(
                injury_severity.get(inj['severity'], 0.5) * 
                np.exp(-0.1 * (game_date - pd.to_datetime(inj['return_date'])).days)
                for _, inj in recent_injuries.iterrows()
            )
            
            recovery_impact = sum(
                injury_recovery.get(inj['injury_type'], 0.5) * 
                np.exp(-0.05 * (game_date - pd.to_datetime(inj['return_date'])).days)
                for _, inj in recent_injuries.iterrows()
            )
            
            # Games since most recent injury
            last_injury = recent_injuries.iloc[-1]
            games_since = (game_date - pd.to_datetime(last_injury['return_date'])).days
            
            # Calculate expected performance impact
            performance_impact = (severity_impact + recovery_impact) * np.exp(-0.05 * games_since)
            
            return pd.Series({
                'injury_severity_impact': severity_impact,
                'injury_recovery_impact': recovery_impact,
                'cumulative_injury_impact': severity_impact + recovery_impact,
                'games_since_injury': games_since,
                'expected_performance_impact': performance_impact
            })
        
        # Calculate detailed injury metrics for each player-game
        injury_metrics = result_df.apply(calculate_detailed_injury_impact, axis=1)
        result_df = pd.concat([result_df, injury_metrics], axis=1)
        
        return result_df
    
    def process_injury_data(self, injuries: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Process injury data and create injury-related features"""
        player_games = player_stats.copy()
        
        def calculate_injury_features(row):
            game_date = pd.to_datetime(row['game_date'])
            player_injuries = injuries[injuries['player_id'] == row['player_id']]
            
            # Days since last injury
            last_injury = player_injuries[player_injuries['injury_date'] < game_date]
            days_since_injury = None if last_injury.empty else (game_date - pd.to_datetime(last_injury['return_date'].iloc[-1])).days
            
            # Recent injury history (last 90 days)
            recent_injuries = player_injuries[
                (pd.to_datetime(player_injuries['injury_date']) > game_date - timedelta(days=90)) &
                (pd.to_datetime(player_injuries['injury_date']) < game_date)
            ]
            
            return pd.Series({
                'days_since_injury': days_since_injury if days_since_injury and days_since_injury > 0 else 999,
                'recent_injuries_90d': len(recent_injuries),
                'recent_games_missed_90d': recent_injuries['games_missed'].sum() if not recent_injuries.empty else 0
            })
        
        injury_features = player_games.apply(calculate_injury_features, axis=1)
        return pd.concat([player_games, injury_features], axis=1)
    
    def aggregate_team_player_features(self, player_features: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player features to team level for each game"""
        team_features = []
        
        for game_id in player_features['game_id'].unique():
            game_players = player_features[player_features['game_id'] == game_id]
            
            for team_id in game_players['team_id'].unique():
                team_players = game_players[game_players['team_id'] == team_id]
                
                # Starters vs Bench split
                starters = team_players[team_players['starter'] == 1]
                bench = team_players[team_players['starter'] == 0]
                
                # Aggregate features
                for window in self.window_sizes:
                    team_aggs = {
                        f'team_starters_pts_{window}g_avg': starters[f'points_{window}g_avg'].mean(),
                        f'team_bench_pts_{window}g_avg': bench[f'points_{window}g_avg'].mean(),
                        f'team_starters_plus_minus_{window}g_avg': starters[f'plus_minus_{window}g_avg'].mean(),
                        f'team_performance_volatility_{window}g': team_players[f'performance_volatility_{window}g'].mean(),
                    }
                    
                    # Position-specific features
                    for pos in self.positions:
                        pos_players = team_players[team_players['position'] == pos]
                        if not pos_players.empty:
                            team_aggs.update({
                                f'team_{pos}_pts_{window}g_avg': pos_players[f'points_{window}g_avg'].mean(),
                                f'team_{pos}_plus_minus_{window}g_avg': pos_players[f'plus_minus_{window}g_avg'].mean()
                            })
                    
                    # Injury impact features
                    team_aggs.update({
                        'team_players_returning': sum(team_players['days_since_injury'].between(0, 5)),
                        'team_recent_injuries': team_players['recent_injuries_90d'].sum(),
                        'team_games_missed_90d': team_players['recent_games_missed_90d'].sum()
                    })
                    
                    team_features.append({
                        'game_id': game_id,
                        'team_id': team_id,
                        **team_aggs
                    })
        
        return pd.DataFrame(team_features)
    
    def calculate_player_efficiency(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced efficiency metrics for players
        """
        df = stats_df.copy()
        
        # True Shooting Percentage (TS%)
        df['true_shooting_pct'] = (
            df['points'] / (2 * (df['fg_attempted'] + 0.44 * df['ft_attempted']))
        ).fillna(0)
        
        # Usage Rate
        df['usage_rate'] = (
            (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) /
            (df['minutes_played'] + 1e-6)
        ).fillna(0)
        
        # Assist to Turnover Ratio
        df['ast_to_ratio'] = (df['assists'] / (df['turnovers'] + 1e-6)).fillna(0)
        
        # Position-adjusted efficiency
        for pos in self.positions:
            pos_mask = df['position'] == pos
            for stat, weight in self.position_weights[pos].items():
                if stat in df.columns:
                    df.loc[pos_mask, f'{stat}_adjusted'] = df.loc[pos_mask, stat] * weight
        
        return df

    def calculate_lineup_chemistry(self, df: pd.DataFrame, team_id: str) -> pd.DataFrame:
        """
        Calculate lineup chemistry metrics based on player combinations
        """
        result_df = df.copy()
        
        # Get starters for the team
        starters = result_df[
            (result_df['team_id'] == team_id) & 
            (result_df['starter'] == 1)
        ]['player_id'].unique()
        
        if len(starters) >= 5:
            # Calculate chemistry score based on games played together
            starter_games = result_df[
                result_df['player_id'].isin(starters)
            ].groupby('game_id').size()
            
            # Weight the chemistry score by number of starters playing together
            chemistry_score = (starter_games * (starter_games / 5)).mean()
            result_df['lineup_chemistry'] = chemistry_score
        
        return result_df

    def add_fatigue_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fatigue indicators based on recent workload
        """
        result_df = df.copy()
        
        # Calculate minutes played in last 5 games
        result_df['recent_minutes'] = result_df.groupby('player_id')['minutes_played'].transform(
            lambda x: x.rolling(window=5, min_periods=1).sum()
        )
        
        # Calculate games played in last 10 days
        result_df['games_10d'] = result_df.groupby('player_id').apply(
            lambda x: x['game_date'].rolling('10D').count()
        ).reset_index(level=0, drop=True)
        
        # Create fatigue score
        result_df['fatigue_score'] = (
            (result_df['recent_minutes'] / (5 * 48)) +  # Normalized minutes load
            (result_df['games_10d'] / 10)               # Normalized game frequency
        )
        
        return result_df
    
    def generate_player_features(self) -> pd.DataFrame:
        """Main function to generate all player-related features"""
        # Load data
        player_stats, injuries = self.get_player_data()
        
        # Calculate recent performance metrics
        print("Calculating player performance metrics...")
        player_performance = self.calculate_player_recent_performance(player_stats)
        
        # Add fatigue metrics
        print("Calculating fatigue metrics...")
        player_performance = self.calculate_fatigue_metrics(player_performance)
        
        # Add defender matchup statistics
        print("Analyzing defender matchups...")
        player_performance = self.calculate_defender_matchups(player_performance)
        
        # Add player chemistry metrics
        print("Calculating player chemistry metrics...")
        player_performance = self.calculate_player_chemistry(player_performance)
        
        # Add position matchup statistics
        print("Analyzing position matchups...")
        player_matchups = self.calculate_position_matchup_stats(player_performance)
        
        # Process detailed injury data
        print("Analyzing injury impacts...")
        player_features = self.analyze_injury_impact(injuries, player_matchups)
        
        # Calculate advanced player efficiency metrics
        print("Calculating player efficiency metrics...")
        player_features = self.calculate_player_efficiency(player_features)
        
        # Calculate lineup chemistry metrics
        print("Calculating lineup chemistry metrics...")
        for team_id in player_features['team_id'].unique():
            team_players = player_features[player_features['team_id'] == team_id]
            player_features.loc[team_players.index] = self.calculate_lineup_chemistry(team_players, team_id)
        
        # Add fatigue indicators
        print("Adding fatigue indicators...")
        player_features = self.add_fatigue_indicators(player_features)
        
        # Aggregate to team level
        print("Aggregating team-level features...")
        team_features = self.aggregate_team_player_features(player_features)
        
        return team_features

def main():
    # Example usage
    db_path = "nba_data.db"
    feature_generator = PlayerFeatureGenerator(db_path)
    player_features = feature_generator.generate_player_features()
    
    # Save features
    player_features.to_csv("player_features.csv", index=False)
    print("Player features generated and saved successfully!")

if __name__ == "__main__":
    main()
