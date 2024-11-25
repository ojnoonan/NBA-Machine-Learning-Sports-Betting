import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from .Player_Features import PlayerFeatureGenerator

class AdvancedFeatureEngineering:
    def __init__(self, window_sizes: List[int] = [5, 10, 15, 20]):
        """
        Initialize with different window sizes for rolling statistics
        Args:
            window_sizes: List of window sizes for rolling calculations
        """
        self.window_sizes = window_sizes
        self.player_feature_generator = PlayerFeatureGenerator(os.path.join(project_root, 'Data', 'dataset.sqlite'))
        self.momentum_weights = {5: 0.4, 10: 0.3, 15: 0.2, 20: 0.1}  # Weights for momentum calculation
        
        # Define decay factors for different stat types
        self.decay_factors = {
            'fast': {  # Fast-changing stats
                'PTS': 0.94,
                'FG_PCT': 0.92,
                'FG3_PCT': 0.90,
                'FT_PCT': 0.91
            },
            'medium': {  # Medium-changing stats
                'AST': 0.88,
                'TOV': 0.87,
                'STL': 0.86
            },
            'slow': {  # Slower-changing stats
                'REB': 0.85,
                'BLK': 0.84,
                'DRTG': 0.83,
                'ORTG': 0.83
            }
        }
        
    def calculate_weighted_stats(self, df: pd.DataFrame, team_id_col: str, date_col: str, 
                               stat_cols: List[str]) -> pd.DataFrame:
        """
        Calculate context-aware exponentially weighted moving averages
        """
        result_df = df.copy()
        result_df = result_df.sort_values([team_id_col, date_col])
        
        # Calculate base weighted averages
        for stat in stat_cols:
            # Determine appropriate decay factor
            decay = 0.9  # default decay
            for speed, stats in self.decay_factors.items():
                if stat in stats:
                    decay = stats[stat]
                    break
            
            # Calculate home/away specific EWMA
            home_mask = result_df['is_home'] == 1
            away_mask = result_df['is_home'] == 0
            
            for mask, context in [(home_mask, 'HOME'), (away_mask, 'AWAY')]:
                weighted_col = f'{context}_WEIGHTED_{stat}'
                result_df.loc[mask, weighted_col] = result_df[mask].groupby(team_id_col)[stat].transform(
                    lambda x: x.ewm(alpha=1-decay, adjust=False).mean()
                )
            
            # Adjust for opponent strength
            if 'opponent_rank' in result_df.columns:
                result_df[f'ADJ_WEIGHTED_{stat}'] = result_df[f'HOME_WEIGHTED_{stat}'] * \
                    (1 + 0.1 * (result_df['opponent_rank'] / result_df['opponent_rank'].max()))
            
            # Adjust for rest days if available
            if 'rest_days' in result_df.columns:
                rest_factor = 1 + (result_df['rest_days'] - result_df['rest_days'].mean()) * 0.05
                result_df[f'REST_ADJ_WEIGHTED_{stat}'] = result_df[f'ADJ_WEIGHTED_{stat}'] * rest_factor
        
        return result_df
    
    def add_performance_windows(self, df: pd.DataFrame, team_id_col: str, 
                              date_col: str, stat_cols: List[str]) -> pd.DataFrame:
        """
        Add rolling statistics for different window sizes
        """
        result_df = df.copy()
        
        # Sort by date for each team
        result_df = result_df.sort_values([team_id_col, date_col])
        
        # Calculate rolling stats for each window size
        for window in self.window_sizes:
            for stat in stat_cols:
                # Average
                avg_col = f'ROLLING_{stat}_{window}G'
                result_df[avg_col] = result_df.groupby(team_id_col)[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Standard deviation (volatility)
                std_col = f'STD_{stat}_{window}G'
                result_df[std_col] = result_df.groupby(team_id_col)[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                
        return result_df
    
    def add_home_away_splits(self, df: pd.DataFrame, team_id_col: str, 
                           is_home_col: str, stat_cols: List[str]) -> pd.DataFrame:
        """
        Calculate separate statistics for home and away games
        """
        result_df = df.copy()
        
        for stat in stat_cols:
            # Home stats
            home_mask = result_df[is_home_col] == 1
            result_df[f'HOME_{stat}_AVG'] = result_df[home_mask].groupby(team_id_col)[stat].transform('mean')
            
            # Away stats
            away_mask = result_df[is_home_col] == 0
            result_df[f'AWAY_{stat}_AVG'] = result_df[away_mask].groupby(team_id_col)[stat].transform('mean')
            
        return result_df
    
    def add_opponent_strength(self, df: pd.DataFrame, team_id_col: str, 
                            opp_id_col: str, stat_cols: List[str]) -> pd.DataFrame:
        """
        Add opponent quality adjustments
        """
        result_df = df.copy()
        
        # Calculate season averages for each team
        team_averages = df.groupby(team_id_col)[stat_cols].mean()
        
        # Calculate league averages
        league_averages = df[stat_cols].mean()
        
        # Calculate opponent strength (relative to league average)
        for stat in stat_cols:
            opp_strength = team_averages[stat] / league_averages[stat]
            result_df[f'OPP_STRENGTH_{stat}'] = result_df[opp_id_col].map(opp_strength)
            
        return result_df
    
    def add_momentum_indicators(self, df: pd.DataFrame, team_id_col: str, 
                              date_col: str, point_diff_col: str) -> pd.DataFrame:
        """
        Add momentum-based indicators
        """
        result_df = df.copy()
        
        # Sort by date for each team
        result_df = result_df.sort_values([team_id_col, date_col])
        
        # Calculate point differential trends
        for window in self.window_sizes:
            # Rolling average point differential
            result_df[f'POINT_DIFF_TREND_{window}G'] = result_df.groupby(team_id_col)[point_diff_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Point differential volatility
            result_df[f'POINT_DIFF_VOL_{window}G'] = result_df.groupby(team_id_col)[point_diff_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return result_df
    
    def add_matchup_features(self, df: pd.DataFrame, team_id_col: str, 
                           opp_id_col: str, date_col: str) -> pd.DataFrame:
        """
        Add comprehensive head-to-head matchup features
        """
        result_df = df.copy()
        
        # Create a unique matchup ID (combining both teams)
        result_df['matchup_id'] = result_df.apply(
            lambda row: f"{min(row[team_id_col], row[opp_id_col])}_{max(row[team_id_col], row[opp_id_col])}", 
            axis=1
        )
        
        # Calculate historical head-to-head records with different windows
        for window in self.window_sizes:
            # Win rate
            result_df[f'H2H_WIN_RATE_{window}G'] = result_df.groupby('matchup_id')['Home_Team_Win'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Point differential
            result_df[f'H2H_POINT_DIFF_{window}G'] = result_df.groupby('matchup_id')['POINT_DIFF'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Scoring differential
            result_df[f'H2H_SCORING_DIFF_{window}G'] = result_df.groupby('matchup_id').apply(
                lambda x: (x['HOME_AVG_PTS'] - x['AWAY_AVG_PTS']).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            # Shooting efficiency differential
            result_df[f'H2H_FG_PCT_DIFF_{window}G'] = result_df.groupby('matchup_id').apply(
                lambda x: (x['HOME_AVG_FG_PCT'] - x['AWAY_AVG_FG_PCT']).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            result_df[f'H2H_FG3_PCT_DIFF_{window}G'] = result_df.groupby('matchup_id').apply(
                lambda x: (x['HOME_AVG_FG3_PCT'] - x['AWAY_AVG_FG3_PCT']).rolling(window=window, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
        
        return result_df
    
    def add_style_matchup_features(self, df: pd.DataFrame, team_id_col: str, 
                                 opp_id_col: str) -> pd.DataFrame:
        """
        Add features that capture playing style matchups and mismatches
        """
        result_df = df.copy()
        
        # Calculate team style metrics
        team_metrics = {}
        
        # Calculate average metrics for each team
        for team_id in df[team_id_col].unique():
            team_data = df[df[team_id_col] == team_id]
            team_metrics[team_id] = {
                'PACE': team_data['HOME_AVG_PTS'].mean() + team_data['AWAY_AVG_PTS'].mean(),
                'FG3_RATE': team_data['HOME_AVG_FG3_PCT'].mean(),
                'PAINT_RATE': 100 - team_data['HOME_AVG_FG3_PCT'].mean(),
                'AST_RATE': (team_data['HOME_AVG_AST'] / team_data['HOME_AVG_PTS']).mean(),
                'TOV_RATE': (team_data['HOME_AVG_TOV'] / team_data['HOME_AVG_PTS']).mean(),
                'REB_RATE': (team_data['HOME_AVG_REB'] / (team_data['HOME_AVG_REB'] + team_data['AWAY_AVG_REB'])).mean(),
                'EFFICIENCY': team_data['HOME_AVG_PTS'].mean() / team_data['HOME_AVG_FGA'].mean() if 'HOME_AVG_FGA' in team_data else 100
            }
        
        # Add style differential features
        metrics = ['PACE', 'FG3_RATE', 'PAINT_RATE', 'AST_RATE', 'TOV_RATE', 'REB_RATE', 'EFFICIENCY']
        
        for metric in metrics:
            # Raw differential
            result_df[f'STYLE_DIFF_{metric}'] = result_df.apply(
                lambda row: team_metrics[row[team_id_col]][metric] - team_metrics[row[opp_id_col]][metric],
                axis=1
            )
            
            # Relative strength (ratio)
            result_df[f'STYLE_RATIO_{metric}'] = result_df.apply(
                lambda row: team_metrics[row[team_id_col]][metric] / team_metrics[row[opp_id_col]][metric],
                axis=1
            )
        
        # Add specific matchup advantage indicators
        result_df['PACE_ADVANTAGE'] = (
            (result_df['STYLE_DIFF_PACE'] > 0) & 
            (result_df['HOME_AVG_PTS'] > result_df['AWAY_AVG_PTS'])
        ).astype(int)
        
        result_df['SHOOTING_ADVANTAGE'] = (
            (result_df['STYLE_DIFF_FG3_RATE'] > 0) & 
            (result_df['HOME_AVG_FG_PCT'] > result_df['AWAY_AVG_FG_PCT'])
        ).astype(int)
        
        result_df['PAINT_ADVANTAGE'] = (
            (result_df['STYLE_DIFF_PAINT_RATE'] > 0) & 
            (result_df['HOME_AVG_REB'] > result_df['AWAY_AVG_REB'])
        ).astype(int)
        
        result_df['BALL_CONTROL_ADVANTAGE'] = (
            (result_df['STYLE_DIFF_AST_RATE'] > 0) & 
            (result_df['STYLE_DIFF_TOV_RATE'] < 0)
        ).astype(int)
        
        # Calculate overall style mismatch score
        style_columns = [col for col in result_df.columns if col.startswith('STYLE_DIFF_')]
        result_df['STYLE_MISMATCH_SCORE'] = result_df[style_columns].abs().mean(axis=1)
        
        return result_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Applying feature engineering...")
        
        # Generate team-based features
        result_df = self.add_rolling_averages(df, 'Home_Team_ID', 'HOME')
        result_df = self.add_rolling_averages(result_df, 'Away_Team_ID', 'AWAY')
        result_df = self.add_relative_stats(result_df)
        result_df = self.add_momentum_features(result_df)
        result_df = self.add_matchup_features(result_df, 'Home_Team_ID', 'Away_Team_ID', 'Game_Date')
        result_df = self.add_style_matchup_features(result_df, 'Home_Team_ID', 'Away_Team_ID')
        result_df = self.add_advanced_features(result_df)
        
        # Generate and merge player-based features
        print("Generating player-based features...")
        player_features = self.player_feature_generator.generate_player_features()
        
        # Merge home team player features
        home_features = player_features.copy()
        home_features.columns = ['HOME_' + col if col not in ['game_id', 'team_id'] else col for col in home_features.columns]
        result_df = result_df.merge(
            home_features,
            left_on=['Game_ID', 'Home_Team_ID'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        
        # Merge away team player features
        away_features = player_features.copy()
        away_features.columns = ['AWAY_' + col if col not in ['game_id', 'team_id'] else col for col in away_features.columns]
        result_df = result_df.merge(
            away_features,
            left_on=['Game_ID', 'Away_Team_ID'],
            right_on=['game_id', 'team_id'],
            how='left'
        )
        
        # Calculate player matchup advantages
        for pos in self.player_feature_generator.positions:
            for window in self.window_sizes:
                # Scoring advantage
                result_df[f'{pos}_MATCHUP_PTS_ADV_{window}G'] = (
                    result_df[f'HOME_team_{pos}_pts_{window}g_avg'] - 
                    result_df[f'AWAY_team_{pos}_pts_{window}g_avg']
                )
                
                # Plus/minus advantage
                result_df[f'{pos}_MATCHUP_PLUS_MINUS_ADV_{window}G'] = (
                    result_df[f'HOME_team_{pos}_plus_minus_{window}g_avg'] - 
                    result_df[f'AWAY_team_{pos}_plus_minus_{window}g_avg']
                )
        
        # Calculate overall injury impact
        result_df['HOME_INJURY_IMPACT'] = (
            result_df['HOME_team_players_returning'] * 0.5 +
            result_df['HOME_team_recent_injuries'] * 0.3 +
            result_df['HOME_team_games_missed_90d'] * 0.2
        )
        
        result_df['AWAY_INJURY_IMPACT'] = (
            result_df['AWAY_team_players_returning'] * 0.5 +
            result_df['AWAY_team_recent_injuries'] * 0.3 +
            result_df['AWAY_team_games_missed_90d'] * 0.2
        )
        
        result_df['RELATIVE_INJURY_IMPACT'] = result_df['HOME_INJURY_IMPACT'] - result_df['AWAY_INJURY_IMPACT']
        
        # Add interaction features
        result_df = self.add_interaction_features(result_df)
        
        # Calculate momentum score
        result_df = self.calculate_momentum_score(result_df, 'Home_Team_ID')
        
        # Add matchup features
        result_df = self.add_matchup_features(result_df)
        
        return result_df
    
    def add_rolling_averages(self, df: pd.DataFrame, team_id_col: str, context: str) -> pd.DataFrame:
        """
        Add rolling averages for the specified team context (HOME/AWAY)
        """
        result_df = df.copy()
        
        # Ensure proper date sorting
        result_df = result_df.sort_values([team_id_col, 'GAME_DATE'])
        
        # Add is_home column if not present
        if 'is_home' not in result_df.columns:
            result_df['is_home'] = (context == 'HOME').astype(int)
        
        # Calculate EWMA for each statistic
        for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK']:
            # Determine appropriate decay factor
            decay = 0.9  # default decay
            for speed, stats in self.decay_factors.items():
                if stat in stats:
                    decay = stats[stat]
                    break
            
            # Calculate context-specific EWMA
            col_prefix = f'{context}_WEIGHTED_{stat}'
            result_df[col_prefix] = result_df.groupby(team_id_col)[stat].transform(
                lambda x: x.ewm(alpha=1-decay, adjust=False).mean()
            )
            
            # Add opponent strength adjustment if available
            opp_rank_col = 'OPPONENT_RANK' if 'OPPONENT_RANK' in result_df.columns else None
            if opp_rank_col:
                result_df[f'{col_prefix}_ADJ'] = result_df[col_prefix] * \
                    (1 + 0.1 * (result_df[opp_rank_col] / result_df[opp_rank_col].max()))
            
            # Add rest days adjustment if available
            rest_col = 'REST_DAYS' if 'REST_DAYS' in result_df.columns else None
            if rest_col:
                rest_factor = 1 + (result_df[rest_col] - result_df[rest_col].mean()) * 0.05
                result_df[f'{col_prefix}_REST_ADJ'] = result_df[col_prefix] * rest_factor
        
        return result_df
    
    def add_relative_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative statistics
        """
        result_df = df.copy()
        
        # Calculate relative statistics
        for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK']:
            result_df[f'RELATIVE_{stat}'] = result_df[f'HOME_{stat}'] - result_df[f'AWAY_{stat}']
        
        return result_df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based indicators
        """
        result_df = df.copy()
        
        # Sort by date for each team
        result_df = result_df.sort_values(['Home_Team_ID', 'Game_Date'])
        
        # Calculate point differential trends
        for window in self.window_sizes:
            # Rolling average point differential
            result_df[f'POINT_DIFF_TREND_{window}G'] = result_df.groupby('Home_Team_ID')['POINT_DIFF'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Point differential volatility
            result_df[f'POINT_DIFF_VOL_{window}G'] = result_df.groupby('Home_Team_ID')['POINT_DIFF'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return result_df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between important statistics
        """
        result_df = df.copy()
        
        # Offensive vs Defensive Interactions
        offensive_stats = ['FG_PCT', 'FG3_PCT', 'POINTS', 'ASSISTS']
        defensive_stats = ['BLOCKS', 'STEALS', 'DEFENSIVE_RATING']
        
        for off_stat in offensive_stats:
            for def_stat in defensive_stats:
                if f'HOME_{off_stat}' in result_df.columns and f'AWAY_{def_stat}' in result_df.columns:
                    # Offensive efficiency vs defensive metrics
                    result_df[f'HOME_OFF_DEF_{off_stat}_{def_stat}'] = (
                        result_df[f'HOME_{off_stat}'] / (result_df[f'AWAY_{def_stat}'] + 1e-6)
                    )
                    result_df[f'AWAY_OFF_DEF_{off_stat}_{def_stat}'] = (
                        result_df[f'AWAY_{off_stat}'] / (result_df[f'HOME_{def_stat}'] + 1e-6)
                    )
        
        # Momentum and Form Interactions
        for size in self.window_sizes:
            if f'WIN_PCT_{size}G' in result_df.columns:
                # Momentum-adjusted offensive rating
                result_df[f'HOME_MOMENTUM_RATING_{size}G'] = (
                    result_df[f'HOME_OFFENSIVE_RATING'] * 
                    result_df[f'HOME_WIN_PCT_{size}G']
                )
                result_df[f'AWAY_MOMENTUM_RATING_{size}G'] = (
                    result_df[f'AWAY_OFFENSIVE_RATING'] * 
                    result_df[f'AWAY_WIN_PCT_{size}G']
                )
        
        return result_df

    def calculate_momentum_score(self, df: pd.DataFrame, team_id_col: str) -> pd.DataFrame:
        """
        Calculate a composite momentum score using multiple window sizes
        """
        result_df = df.copy()
        
        # Calculate weighted momentum score
        momentum_cols = []
        for size in self.window_sizes:
            win_pct_col = f'WIN_PCT_{size}G'
            if win_pct_col in result_df.columns:
                momentum_cols.append((win_pct_col, self.momentum_weights[size]))
        
        if momentum_cols:
            result_df['MOMENTUM_SCORE'] = sum(
                result_df[col] * weight for col, weight in momentum_cols
            )
        
        return result_df

    def add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specific to team matchups
        """
        result_df = df.copy()
        
        # Pace-adjusted scoring potential
        if all(col in result_df.columns for col in ['HOME_PACE', 'AWAY_PACE', 'HOME_OFFENSIVE_RATING', 'AWAY_OFFENSIVE_RATING']):
            result_df['EXPECTED_TOTAL_SCORE'] = (
                (result_df['HOME_PACE'] + result_df['AWAY_PACE']) / 2 *
                (result_df['HOME_OFFENSIVE_RATING'] + result_df['AWAY_OFFENSIVE_RATING']) / 200
            )
        
        # Style matchup indicators
        if 'HOME_FG3_PCT' in result_df.columns:
            result_df['THREE_POINT_DOMINANCE'] = (
                result_df['HOME_FG3_PCT'] - result_df['AWAY_FG3_PCT']
            )
        
        return result_df
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced context-specific features"""
        result_df = df.copy()
        
        # Calculate days between games (rest days)
        result_df = result_df.sort_values(['Home_Team_ID', 'Game_Date'])
        result_df['DAYS_REST'] = result_df.groupby('Home_Team_ID')['Game_Date'].diff().dt.days
        
        # Fill first game of season with average rest
        result_df['DAYS_REST'] = result_df['DAYS_REST'].fillna(result_df['DAYS_REST'].mean())
        
        # Back-to-back games indicator
        result_df['IS_BACK_TO_BACK'] = (result_df['DAYS_REST'] <= 1).astype(int)
        
        # Games in last 7 days (fatigue indicator)
        result_df['GAMES_LAST_7D'] = 0
        for team_id in result_df['Home_Team_ID'].unique():
            team_games = result_df[result_df['Home_Team_ID'] == team_id]
            dates = team_games['Game_Date'].values
            games_in_window = [
                self.calculate_games_in_window(dates, date) 
                for date in dates
            ]
            result_df.loc[team_games.index, 'GAMES_LAST_7D'] = games_in_window
        
        # Win/Loss streaks (using only past games)
        result_df['WIN_STREAK'] = 0
        result_df['LOSS_STREAK'] = 0
        
        for team_id in result_df['Home_Team_ID'].unique():
            team_mask = result_df['Home_Team_ID'] == team_id
            team_games = result_df[team_mask].copy()
            
            # Calculate streaks
            streak = 0
            streaks = []
            
            for won in team_games['Home_Team_Win'].values:
                if won:
                    streak = max(1, streak + 1)
                else:
                    streak = min(-1, streak - 1)
                streaks.append(streak)
            
            result_df.loc[team_mask, 'WIN_STREAK'] = [max(0, s) for s in streaks]
            result_df.loc[team_mask, 'LOSS_STREAK'] = [max(0, -s) for s in streaks]
        
        # Home/Away performance features
        result_df['HOME_GAME_COUNT'] = result_df.groupby('Home_Team_ID').cumcount()
        result_df['AWAY_GAME_COUNT'] = result_df.groupby('Away_Team_ID').cumcount()
        
        # Add matchup history features
        result_df = self.add_matchup_history(result_df)
        
        return result_df

    def calculate_games_in_window(self, dates, current_date, days=7):
        """Calculate number of games in the last N days"""
        window_start = current_date - pd.Timedelta(days=days)
        return sum((dates <= current_date) & (dates > window_start))

    def add_matchup_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on historical matchups between teams"""
        result_df = df.copy()
        
        # Create unique matchup identifier
        result_df['MATCHUP_ID'] = result_df.apply(
            lambda row: '_'.join(sorted([str(row['Home_Team_ID']), str(row['Away_Team_ID'])])),
            axis=1
        )
        
        # Calculate historical matchup stats
        matchup_stats = []
        
        for matchup in result_df['MATCHUP_ID'].unique():
            matchup_games = result_df[result_df['MATCHUP_ID'] == matchup].copy()
            matchup_games = matchup_games.sort_values('Game_Date')
            
            # Calculate home team win rate in matchup
            home_team = matchup_games['Home_Team_ID'].iloc[0]
            matchup_games['HOME_TEAM_MATCHUP_WINS'] = (
                matchup_games['Home_Team_Win'] & (matchup_games['Home_Team_ID'] == home_team)
            ).cumsum()
            
            matchup_games['MATCHUP_GAMES'] = range(1, len(matchup_games) + 1)
            matchup_games['HOME_TEAM_MATCHUP_WIN_PCT'] = (
                matchup_games['HOME_TEAM_MATCHUP_WINS'] / matchup_games['MATCHUP_GAMES']
            )
            
            matchup_stats.append(matchup_games)
        
        result_df = pd.concat(matchup_stats)
        
        return result_df

def main():
    # Example usage
    feature_engineer = AdvancedFeatureEngineering(window_sizes=[5, 10, 15, 20])
    
    # Load your dataset
    print("Loading dataset...")
    conn = sqlite3.connect(os.path.join(project_root, 'Data', 'dataset.sqlite'))
    df = pd.read_sql_query("SELECT * FROM dataset_2012_24", conn)
    conn.close()
    
    # Add IS_HOME column based on Home_Team_ID
    print("Preparing base features...")
    df['IS_HOME'] = 1  # All rows are from home team perspective in our schema
    
    # Basic stats to engineer features from
    basic_stats = ['HOME_AVG_PTS', 'HOME_AVG_FG_PCT', 'HOME_AVG_FG3_PCT', 
                   'HOME_AVG_REB', 'HOME_AVG_AST', 'HOME_AVG_TOV', 
                   'HOME_AVG_STL', 'HOME_AVG_BLK']
    
    print("Applying feature engineering...")
    # Apply feature engineering using the full column names
    df = feature_engineer.engineer_features(df)
    
    # Save the enhanced dataset
    print("Saving enhanced dataset...")
    conn = sqlite3.connect(os.path.join(project_root, 'Data', 'dataset.sqlite'))
    df.to_sql('dataset_2012_24_enhanced', conn, if_exists='replace', index=False)
    
    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    
    # Get all numeric columns except index-like columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Home_Team_ID', 'Away_Team_ID']]
    
    # Calculate correlations with Home_Team_Win
    correlations = df[numeric_cols].corr()['Home_Team_Win'].sort_values(ascending=False)
    
    # Print top 20 most predictive features
    print("\nTop 20 Most Predictive Features:")
    print(correlations.head(20))
    
    # Save correlations to a file
    print("\nSaving correlation analysis...")
    correlations.to_csv(os.path.join(project_root, 'Data', 'feature_correlations.csv'))
    
    conn.close()
    print("\nEnhanced features have been created and saved to dataset_2012_24_enhanced")
    print("Correlation analysis has been saved to feature_correlations.csv")

if __name__ == "__main__":
    main()
