import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

class AdvancedFeatureGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all features"""
        print("Generating advanced features...")
        
        print("Generating team form features...")
        self.df = self.add_enhanced_streak_features()
        
        print("Generating advanced shooting metrics...")
        self.df = self.add_rolling_shooting_metrics()
        
        print("Generating pace-adjusted stats...")
        # Calculate league average pace first
        self.df['league_avg_pace'] = (
            (self.df['HOME_AVG_FGA'] + self.df['AWAY_AVG_FGA']) / 2 + 
            0.4 * (self.df['HOME_AVG_FTA'] + self.df['AWAY_AVG_FTA']) / 2 - 
            (self.df['HOME_AVG_OREB'] + self.df['AWAY_AVG_OREB']) / 2 + 
            (self.df['HOME_AVG_TOV'] + self.df['AWAY_AVG_TOV']) / 2
        )
        self.df = self.add_pace_adjusted_stats()
        
        print("Generating matchup features...")
        self.df = self.add_head_to_head_features()
        self.df = self.add_matchup_shooting_metrics()
        self.df = self.add_advanced_matchup_features()
        
        print("Generating interaction features...")
        self.df = self.add_interaction_features()
        
        print("Generating time-based features...")
        self.df = self.add_time_based_features()
        
        print("Generating momentum indicators...")
        self.df = self.add_momentum_indicators()
        
        return self.df
    
    def add_enhanced_streak_features(self) -> pd.DataFrame:
        """Add enhanced winning streak features"""
        df = self.df.copy()
        
        # Calculate weighted streaks (more recent games have higher weight)
        def calculate_weighted_streak(group: pd.DataFrame, window: int = 10) -> float:
            if len(group) < window:
                return 0
            recent_games = group.tail(window)
            weights = np.exp(np.linspace(-1, 0, len(recent_games)))
            weighted_wins = (recent_games['Home_Team_Win'].astype(float) * weights).sum()
            return weighted_wins / weights.sum()
        
        # Add weighted streaks for home and away teams
        for team_type in ['HOME', 'AWAY']:
            team_id_col = 'Home_Team_ID' if team_type == 'HOME' else 'Away_Team_ID'
            df_team = df.groupby(team_id_col).apply(
                lambda x: x.assign(
                    weighted_streak=calculate_weighted_streak(x)
                )
            ).reset_index(drop=True)
            
            df[f'{team_type}_WEIGHTED_STREAK'] = df_team['weighted_streak']
            
            # Add streak momentum (acceleration of wins)
            df[f'{team_type}_STREAK_MOMENTUM'] = df.groupby(team_id_col)[f'{team_type}_WEIGHTED_STREAK'].diff()
        
        return df
    
    def add_head_to_head_features(self) -> pd.DataFrame:
        """Add historical head-to-head performance features"""
        df = self.df.copy()
        
        def calculate_h2h_stats(group: pd.DataFrame) -> Dict:
            if len(group) < 1:
                return {
                    'H2H_WIN_RATE': 0.5,
                    'H2H_AVG_POINT_DIFF': 0,
                    'H2H_GAMES_PLAYED': 0
                }
            
            wins = group['Home_Team_Win'].mean()
            point_diff = (group['HOME_AVG_PTS'] - group['AWAY_AVG_PTS']).mean()
            games = len(group)
            
            return {
                'H2H_WIN_RATE': wins,
                'H2H_AVG_POINT_DIFF': point_diff,
                'H2H_GAMES_PLAYED': games
            }
        
        # Calculate H2H features
        h2h_features = []
        for idx, row in df.iterrows():
            past_matchups = df[
                (df['Date'] < row['Date']) &
                (
                    ((df['Home_Team_ID'] == row['Home_Team_ID']) & 
                     (df['Away_Team_ID'] == row['Away_Team_ID'])) |
                    ((df['Home_Team_ID'] == row['Away_Team_ID']) & 
                     (df['Away_Team_ID'] == row['Home_Team_ID']))
                )
            ].copy()
            
            h2h_stats = calculate_h2h_stats(past_matchups)
            h2h_features.append(h2h_stats)
        
        # Add H2H features to dataframe
        h2h_df = pd.DataFrame(h2h_features)
        for col in h2h_df.columns:
            df[col] = h2h_df[col]
        
        return df
    
    def add_rolling_shooting_metrics(self) -> pd.DataFrame:
        """Add rolling window shooting metrics"""
        df = self.df.copy()
        windows = [5, 10, 15]
        
        shooting_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        
        for window in windows:
            for team_type in ['HOME', 'AWAY']:
                team_id_col = 'Home_Team_ID' if team_type == 'HOME' else 'Away_Team_ID'
                for col in shooting_cols:
                    # Calculate rolling average
                    df[f'{team_type}_ROLLING_{window}_{col}'] = df.groupby(team_id_col)[
                        f'{team_type}_AVG_{col}'
                    ].transform(lambda x: x.rolling(window, min_periods=1).mean())
                    
                    # Calculate rolling standard deviation
                    df[f'{team_type}_ROLLING_{window}_{col}_STD'] = df.groupby(team_id_col)[
                        f'{team_type}_AVG_{col}'
                    ].transform(lambda x: x.rolling(window, min_periods=1).std())
        
        return df
    
    def add_matchup_shooting_metrics(self) -> pd.DataFrame:
        """Add matchup-specific shooting metrics"""
        df = self.df.copy()
        
        # Calculate offensive vs defensive ratings
        for team_type in ['HOME', 'AWAY']:
            opponent_type = 'AWAY' if team_type == 'HOME' else 'HOME'
            team_id_col = 'Home_Team_ID' if team_type == 'HOME' else 'Away_Team_ID'
            
            # Offensive efficiency vs opponent's defense
            df[f'{team_type}_OFF_EFF_VS_DEF'] = df.groupby(team_id_col).apply(
                lambda x: x[f'{team_type}_AVG_PTS'] / 
                         x[f'{opponent_type}_AVG_PTS'].rolling(10, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            # Shooting efficiency vs opponent's defense
            for metric in ['FG_PCT', 'FG3_PCT']:
                df[f'{team_type}_{metric}_VS_DEF'] = df.groupby(team_id_col).apply(
                    lambda x: x[f'{team_type}_AVG_{metric}'] / 
                             x[f'{opponent_type}_AVG_{metric}'].rolling(10, min_periods=1).mean()
                ).reset_index(level=0, drop=True)
        
        return df
    
    def add_interaction_features(self) -> pd.DataFrame:
        """Add interaction features between key metrics"""
        df = self.df.copy()
        
        # Team strength differential features
        for metric in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
            df[f'DIFF_{metric}'] = df[f'HOME_AVG_{metric}'] - df[f'AWAY_AVG_{metric}']
            
        # Efficiency interaction features
        df['HOME_OFF_DEF_RATIO'] = df['HOME_OFF_EFF_VS_DEF'] / df['AWAY_OFF_EFF_VS_DEF']
        df['PACE_ADVANTAGE'] = df['HOME_PACE'] - df['AWAY_PACE']
        
        # Recent form interaction
        df['MOMENTUM_ADVANTAGE'] = df['HOME_OVERALL_MOMENTUM'] - df['AWAY_OVERALL_MOMENTUM']
        df['STREAK_ADVANTAGE'] = df['HOME_WEIGHTED_STREAK'] - df['AWAY_WEIGHTED_STREAK']
        
        # Shooting efficiency interactions
        for window in [5, 10, 15]:
            df[f'FG3_ADVANTAGE_{window}'] = (
                df[f'HOME_ROLLING_{window}_FG3_PCT'] - 
                df[f'AWAY_ROLLING_{window}_FG3_PCT']
            )
            df[f'FG_CONSISTENCY_{window}'] = (
                df[f'HOME_ROLLING_{window}_FG_PCT_STD'] /
                df[f'AWAY_ROLLING_{window}_FG_PCT_STD']
            )
        
        # Create composite score
        df['COMPOSITE_ADVANTAGE'] = (
            0.3 * df['HOME_OFF_DEF_RATIO'] +
            0.2 * df['MOMENTUM_ADVANTAGE'] +
            0.2 * df['FG3_ADVANTAGE_5'] +
            0.15 * df['STREAK_ADVANTAGE'] +
            0.15 * df['H2H_WIN_RATE']
        )
        
        return df
    
    def add_advanced_matchup_features(self) -> pd.DataFrame:
        """Add advanced matchup-specific features"""
        df = self.df.copy()
        
        # Style matchup features
        df['PACE_MISMATCH'] = abs(df['HOME_PACE'] - df['AWAY_PACE'])
        df['SIZE_MISMATCH'] = abs(
            (df['HOME_AVG_REB'] + df['HOME_AVG_BLK']) -
            (df['AWAY_AVG_REB'] + df['AWAY_AVG_BLK'])
        )
        
        # Defensive matchup features
        df['PERIMETER_DEF_ADVANTAGE'] = (
            df['HOME_AVG_STL'] / df['AWAY_AVG_STL'] *
            df['HOME_AVG_BLK'] / df['AWAY_AVG_BLK']
        )
        
        # Offensive style matchup
        df['THREE_POINT_RELIANCE'] = (
            df['HOME_ROLLING_10_FG3_PCT'] / df['HOME_ROLLING_10_FG_PCT'] -
            df['AWAY_ROLLING_10_FG3_PCT'] / df['AWAY_ROLLING_10_FG_PCT']
        )
        
        # Form-based matchup features
        df['FORM_STABILITY'] = (
            df['HOME_ROLLING_10_FG_PCT_STD'] / df['HOME_AVG_FG_PCT'] -
            df['AWAY_ROLLING_10_FG_PCT_STD'] / df['AWAY_AVG_FG_PCT']
        )
        
        # Create matchup score
        df['MATCHUP_SCORE'] = (
            0.25 * df['PERIMETER_DEF_ADVANTAGE'] +
            0.25 * df['SIZE_MISMATCH'] +
            0.2 * df['FORM_STABILITY'] +
            0.15 * df['PACE_MISMATCH'] +
            0.15 * df['THREE_POINT_RELIANCE']
        )
        
        return df
    
    def add_time_based_features(self) -> pd.DataFrame:
        """Add time-based features"""
        df = self.df.copy()
        
        # Convert date to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Days since last game
        for team_type in ['HOME', 'AWAY']:
            team_id_col = f'{team_type}_Team_ID'
            df[f'{team_type}_DAYS_REST'] = df.groupby(team_id_col)['Date'].diff().dt.days
            
            # Fill NaN with median rest days
            median_rest = df[f'{team_type}_DAYS_REST'].median()
            df[f'{team_type}_DAYS_REST'].fillna(median_rest, inplace=True)
        
        # Rest advantage
        df['REST_ADVANTAGE'] = df['HOME_DAYS_REST'] - df['AWAY_DAYS_REST']
        
        # Season progress (0-1 scale)
        df['SEASON_PROGRESS'] = (df['Date'] - df.groupby(df['Date'].dt.year)['Date'].transform('min')) / \
                               (df.groupby(df['Date'].dt.year)['Date'].transform('max') - 
                                df.groupby(df['Date'].dt.year)['Date'].transform('min'))
        
        return df
    
    def add_momentum_indicators(self) -> pd.DataFrame:
        """Add team momentum indicators"""
        df = self.df.copy()
        
        # Calculate momentum scores based on recent performance trends
        for team_type in ['HOME', 'AWAY']:
            team_id_col = 'Home_Team_ID' if team_type == 'HOME' else 'Away_Team_ID'
            
            # Scoring momentum
            df[f'{team_type}_SCORING_MOMENTUM'] = df.groupby(team_id_col)[
                f'{team_type}_AVG_PTS'
            ].transform(lambda x: x.ewm(span=5).mean() - x.ewm(span=15).mean())
            
            # Defensive momentum
            df[f'{team_type}_DEFENSIVE_MOMENTUM'] = -1 * df.groupby(team_id_col)[
                f'{team_type}_AVG_PTS'
            ].transform(lambda x: x.ewm(span=5).mean() - x.ewm(span=15).mean())
            
            # Overall momentum (composite score)
            df[f'{team_type}_OVERALL_MOMENTUM'] = (
                df[f'{team_type}_SCORING_MOMENTUM'] +
                df[f'{team_type}_DEFENSIVE_MOMENTUM'] +
                df[f'{team_type}_WEIGHTED_STREAK'] * 10
            )
        
        return df
    
    def add_pace_adjusted_stats(self) -> pd.DataFrame:
        """Add pace-adjusted statistics"""
        df = self.df.copy()
        
        # Calculate pace for each team
        for team_type in ['HOME', 'AWAY']:
            # Basic pace calculation (possessions per game)
            df[f'{team_type}_PACE'] = (
                df[f'{team_type}_AVG_FGA'] + 
                0.4 * df[f'{team_type}_AVG_FTA'] - 
                df[f'{team_type}_AVG_OREB'] + 
                df[f'{team_type}_AVG_TOV']
            )
            
            # Offensive efficiency (points per 100 possessions)
            df[f'{team_type}_OFF_EFF'] = (
                df[f'{team_type}_AVG_PTS'] * 100 / df[f'{team_type}_PACE']
            )
            
            # Defensive efficiency (points allowed per 100 possessions)
            df[f'{team_type}_DEF_EFF'] = (
                df[f'{team_type}_AVG_PTS'] * 100 / df[f'{team_type}_PACE']
            ) if team_type == 'HOME' else (
                df[f'{team_type}_AVG_PTS'] * 100 / df[f'{team_type}_PACE']
            )
            
            # Net rating
            df[f'{team_type}_NET_RATING'] = (
                df[f'{team_type}_OFF_EFF'] - df[f'{team_type}_DEF_EFF']
            )
            
            # Pace-adjusted scoring
            df[f'{team_type}_PACE_ADJ_SCORING'] = (
                df[f'{team_type}_AVG_PTS'] * 
                (df['league_avg_pace'] / df[f'{team_type}_PACE'])
            )
            
            # Pace-adjusted efficiency
            df[f'{team_type}_PACE_ADJ_EFF'] = (
                df[f'{team_type}_OFF_EFF'] * 
                (df['league_avg_pace'] / df[f'{team_type}_PACE'])
            )
        
        return df
