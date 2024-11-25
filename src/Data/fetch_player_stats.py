import os
import sys
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
import sqlite3
import time
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def get_game_ids(start_date, end_date):
    """Fetch game IDs between start and end dates"""
    logger.info(f"Fetching games between {start_date} and {end_date}")
    
    gamefinder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=start_date,
        date_to_nullable=end_date,
        league_id_nullable='00'  # NBA
    )
    games_df = gamefinder.get_data_frames()[0]
    
    return games_df['GAME_ID'].unique()

def get_player_stats(game_id):
    """Fetch player stats for a specific game"""
    try:
        time.sleep(1)  # Rate limiting
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        stats_df = box.player_stats.get_data_frame()
        
        # Add game date and season
        game_info = box.game_summary.get_data_frame()
        stats_df['GAME_DATE'] = game_info['GAME_DATE'].iloc[0]
        
        # Calculate season (assuming season starts in October)
        game_date = datetime.strptime(game_info['GAME_DATE'].iloc[0], '%Y-%m-%d')
        season = game_date.year if game_date.month >= 10 else game_date.year - 1
        stats_df['Season'] = season
        
        return stats_df
    except Exception as e:
        logger.error(f"Error fetching stats for game {game_id}: {str(e)}")
        return None

def calculate_rest_days(df):
    """Calculate days of rest between games for each player"""
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df['PREV_GAME'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    df['days_rest'] = pd.to_datetime(df['GAME_DATE']) - pd.to_datetime(df['PREV_GAME'])
    df['days_rest'] = df['days_rest'].dt.days
    df['days_rest'] = df['days_rest'].fillna(3)  # Default to 3 days for first game
    return df

def create_database():
    """Create SQLite database and tables"""
    db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Create player_game_stats table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS player_game_stats (
        GAME_ID TEXT,
        PLAYER_ID INTEGER,
        PLAYER_NAME TEXT,
        TEAM_ID INTEGER,
        TEAM_ABBREVIATION TEXT,
        TEAM_CITY TEXT,
        MIN REAL,
        FGM INTEGER,
        FGA INTEGER,
        FG_PCT REAL,
        FG3M INTEGER,
        FG3A INTEGER,
        FG3_PCT REAL,
        FTM INTEGER,
        FTA INTEGER,
        FT_PCT REAL,
        OREB INTEGER,
        DREB INTEGER,
        REB INTEGER,
        AST INTEGER,
        STL INTEGER,
        BLK INTEGER,
        TURNOVERS INTEGER,
        PF INTEGER,
        PTS INTEGER,
        PLUS_MINUS INTEGER,
        GAME_DATE TEXT,
        Season INTEGER,
        days_rest INTEGER,
        OPPONENT_TEAM_ID INTEGER,
        points INTEGER,
        rebounds INTEGER,
        assists INTEGER,
        threes INTEGER,
        PRIMARY KEY (GAME_ID, PLAYER_ID)
    )
    """)
    
    conn.close()

def main():
    # Create database and tables
    create_database()
    
    # Connect to database
    db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Set date range (last 3 seasons)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    # Get game IDs
    game_ids = get_game_ids(start_date.strftime('%m/%d/%Y'), 
                           end_date.strftime('%m/%d/%Y'))
    
    logger.info(f"Found {len(game_ids)} games")
    
    # Process each game
    all_stats = []
    for i, game_id in enumerate(game_ids):
        logger.info(f"Processing game {i+1}/{len(game_ids)}: {game_id}")
        
        stats_df = get_player_stats(game_id)
        if stats_df is not None:
            all_stats.append(stats_df)
    
    # Combine all stats
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        
        # Calculate rest days
        combined_stats = calculate_rest_days(combined_stats)
        
        # Add opponent team ID
        combined_stats['OPPONENT_TEAM_ID'] = combined_stats.groupby('GAME_ID')['TEAM_ID'].transform(lambda x: x.iloc[1] if x.iloc[0] == x.name else x.iloc[0])
        
        # Add target variables
        combined_stats['points'] = combined_stats['PTS']
        combined_stats['rebounds'] = combined_stats['REB']
        combined_stats['assists'] = combined_stats['AST']
        combined_stats['threes'] = combined_stats['FG3M']
        
        # Save to database
        combined_stats.to_sql('player_game_stats', conn, if_exists='replace', index=False)
        logger.info(f"Saved {len(combined_stats)} records to database")
    
    conn.close()

if __name__ == "__main__":
    main()
