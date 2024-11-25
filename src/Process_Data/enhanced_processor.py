import pandas as pd
import numpy as np
from src.Process_Data.process_player_stats import PlayerStatsProcessor

class EnhancedStatsProcessor(PlayerStatsProcessor):
    def calculate_advanced_stats(self, stats_df):
        """Calculate advanced statistics with enhanced metrics"""
        # Call parent method first
        stats_df = super().calculate_advanced_stats(stats_df)
        
        # Add rest days and back-to-back indicators
        stats_df['rest_days'] = stats_df.groupby('player_id')['game_date'].diff().dt.days
        stats_df['is_back_to_back'] = stats_df['rest_days'] == 1
        
        # Calculate shooting metrics
        stats_df['true_shooting_pct'] = stats_df['points'] / (2 * (stats_df['fga'] + 0.44 * stats_df['fta']))
        stats_df['effective_fg_pct'] = (stats_df['fgm'] + 0.5 * stats_df['fg3m']) / stats_df['fga']
        
        # Calculate usage rate
        stats_df['usage_rate'] = ((stats_df['fga'] + 0.44 * stats_df['fta'] + stats_df['turnovers']) * 
                                 (stats_df['minutes_played'] / 48))
        
        # Calculate offensive/defensive rebound split
        stats_df['offensive_rebound_pct'] = stats_df['rebounds_off'] / (stats_df['rebounds_off'] + stats_df['rebounds_def'])
        stats_df['defensive_rebound_pct'] = stats_df['rebounds_def'] / (stats_df['rebounds_off'] + stats_df['rebounds_def'])
        
        # Calculate recent form (last 5 games)
        form_window = 5
        form_cols = ['points', 'assists', 'rebounds', 'efficiency', 'usage_rate']
        
        for col in form_cols:
            # Calculate moving average
            stats_df[f'{col}_form'] = (stats_df.groupby('player_id')[col]
                                      .rolling(window=form_window, min_periods=1)
                                      .mean()
                                      .reset_index(level=0, drop=True))
            
            # Calculate trend (slope over last 5 games)
            stats_df[f'{col}_trend'] = (stats_df.groupby('player_id')[col]
                                       .rolling(window=form_window, min_periods=1)
                                       .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                                       .reset_index(level=0, drop=True))
        
        # Add matchup-specific metrics
        stats_df['matchup_points_avg'] = stats_df.groupby(['player_id', 'opponent_team_id'])['points'].transform('mean')
        stats_df['matchup_assists_avg'] = stats_df.groupby(['player_id', 'opponent_team_id'])['assists'].transform('mean')
        stats_df['matchup_rebounds_avg'] = stats_df.groupby(['player_id', 'opponent_team_id'])['rebounds'].transform('mean')
        
        # Calculate team pace factors
        stats_df['team_pace'] = (stats_df.groupby('team_id')['fga']
                                .transform('mean') + stats_df.groupby('team_id')['fta']
                                .transform('mean') * 0.44)
        
        # Calculate plus/minus impact
        stats_df['plus_minus_per_36'] = stats_df['plus_minus'] * (36 / stats_df['minutes_played'])
        
        return stats_df
        
    def prepare_features(self, df):
        """Prepare features with enhanced feature set"""
        # Standard features
        basic_features = [
            'points_avg', 'assists_avg', 'rebounds_avg',
            'true_shooting_pct', 'effective_fg_pct', 'usage_rate',
            'offensive_rebound_pct', 'defensive_rebound_pct',
            'rest_days', 'is_back_to_back',
            'team_pace', 'plus_minus_per_36'
        ]
        
        # Form features
        form_features = [
            'points_form', 'assists_form', 'rebounds_form',
            'efficiency_form', 'usage_rate_form'
        ]
        
        # Trend features
        trend_features = [
            'points_trend', 'assists_trend', 'rebounds_trend',
            'efficiency_trend', 'usage_rate_trend'
        ]
        
        # Matchup features
        matchup_features = [
            'matchup_points_avg', 'matchup_assists_avg', 'matchup_rebounds_avg',
            'opponent_defensive_rating'
        ]
        
        all_features = basic_features + form_features + trend_features + matchup_features
        
        # Ensure all features exist
        for feature in all_features:
            if feature not in df.columns:
                print(f"Warning: {feature} not found in dataset")
                df[feature] = 0
        
        return df[all_features]
