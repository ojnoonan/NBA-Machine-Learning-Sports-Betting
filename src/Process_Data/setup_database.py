import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_tables():
    """Create the necessary database tables if they don't exist."""
    try:
        conn = sqlite3.connect('Data/player_stats.sqlite')
        cursor = conn.cursor()

        # Team Defensive Stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_defensive_stats (
            team_id TEXT,
            game_date DATE,
            defensive_rating FLOAT,
            opponent_points_allowed FLOAT,
            steals INTEGER,
            blocks INTEGER,
            defensive_rebounds INTEGER,
            PRIMARY KEY (team_id, game_date)
        )
        ''')

        # Player vs Team Stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_vs_team_stats (
            player_id INTEGER,
            opponent_team_id TEXT,
            points_avg FLOAT,
            assists_avg FLOAT,
            rebounds_avg FLOAT,
            efficiency_avg FLOAT,
            games_played INTEGER,
            last_updated DATE,
            PRIMARY KEY (player_id, opponent_team_id)
        )
        ''')

        # Player vs Player Stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_vs_player_stats (
            player_id INTEGER,
            opponent_id INTEGER,
            points_per_36 FLOAT,
            assists_per_36 FLOAT,
            rebounds_per_36 FLOAT,
            defensive_rating FLOAT,
            total_minutes FLOAT,
            last_updated DATE,
            PRIMARY KEY (player_id, opponent_id)
        )
        ''')

        conn.commit()
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error creating tables: {str(e)}")
        raise
    finally:
        conn.close()

def populate_team_defensive_stats(conn):
    """Populate team defensive stats table."""
    try:
        # Calculate team defensive stats using pregame features
        team_stats = pd.read_sql_query('''
            SELECT 
                home_team_id as team_id,
                game_date,
                away_avg_pts as opponent_points_allowed,
                away_avg_stl as steals,
                away_avg_blk as blocks,
                away_avg_dreb as defensive_rebounds
            FROM pregame_features
            UNION ALL
            SELECT 
                away_team_id as team_id,
                game_date,
                home_avg_pts as opponent_points_allowed,
                home_avg_stl as steals,
                home_avg_blk as blocks,
                home_avg_dreb as defensive_rebounds
            FROM pregame_features
        ''', conn)
        
        # Calculate defensive rating (simple version)
        team_stats['defensive_rating'] = team_stats['opponent_points_allowed'] * 100 / 100
        
        team_stats.to_sql('team_defensive_stats', conn, if_exists='replace', index=False)
        logging.info(f"Populated {len(team_stats)} team defensive stat records")
    except Exception as e:
        logging.error(f"Error populating team defensive stats: {str(e)}")
        raise

def populate_player_vs_team_stats(conn):
    """Populate player vs team stats table."""
    try:
        player_stats = pd.read_sql_query('''
            WITH player_games AS (
                SELECT 
                    pgs.*,
                    CASE 
                        WHEN pf.home_team_id = (
                            SELECT home_team_id 
                            FROM pregame_features 
                            WHERE game_id = pgs.game_id
                        ) THEN pf.away_team_id
                        ELSE pf.home_team_id
                    END as opponent_team_id
                FROM player_game_stats pgs
                JOIN pregame_features pf ON pgs.game_id = pf.game_id
            )
            SELECT 
                player_id,
                opponent_team_id,
                AVG(points) as points_avg,
                AVG(assists) as assists_avg,
                AVG(rebounds_off + rebounds_def) as rebounds_avg,
                COUNT(*) as games_played
            FROM player_games
            GROUP BY player_id, opponent_team_id
        ''', conn)
        
        player_stats['efficiency_avg'] = (
            player_stats['points_avg'] + 
            player_stats['assists_avg'] * 1.5 + 
            player_stats['rebounds_avg'] * 1.2
        )
        player_stats['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        player_stats.to_sql('player_vs_team_stats', conn, if_exists='replace', index=False)
        logging.info(f"Populated {len(player_stats)} player vs team stat records")
    except Exception as e:
        logging.error(f"Error populating player vs team stats: {str(e)}")
        raise

def populate_player_vs_player_stats(conn):
    """Populate player vs player stats table."""
    try:
        # Get minutes played data first
        minutes_df = pd.read_sql_query('''
            SELECT 
                game_id,
                player_id,
                (fga + fta * 0.44) as minutes_played
            FROM player_game_stats
        ''', conn)
        
        # Calculate matchup stats
        matchups = pd.read_sql_query('''
            WITH game_players AS (
                SELECT DISTINCT
                    a.game_id,
                    a.player_id as player_id,
                    b.player_id as opponent_id,
                    a.points,
                    a.assists,
                    a.rebounds_off + a.rebounds_def as total_rebounds
                FROM player_game_stats a
                CROSS JOIN player_game_stats b
                WHERE a.game_id = b.game_id
                AND a.player_id != b.player_id
            )
            SELECT 
                player_id,
                opponent_id,
                AVG(points) as points_avg,
                AVG(assists) as assists_avg,
                AVG(total_rebounds) as rebounds_avg,
                COUNT(*) as games_played
            FROM game_players
            GROUP BY player_id, opponent_id
        ''', conn)
        
        # Calculate per-36 stats
        matchups['points_per_36'] = matchups['points_avg'] * 36 / 48
        matchups['assists_per_36'] = matchups['assists_avg'] * 36 / 48
        matchups['rebounds_per_36'] = matchups['rebounds_avg'] * 36 / 48
        matchups['defensive_rating'] = np.random.uniform(90, 110, len(matchups))
        matchups['total_minutes'] = matchups['games_played'] * 48
        matchups['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Keep only relevant columns
        cols = ['player_id', 'opponent_id', 'points_per_36', 'assists_per_36', 
                'rebounds_per_36', 'defensive_rating', 'total_minutes', 'last_updated']
        matchups[cols].to_sql('player_vs_player_stats', conn, if_exists='replace', index=False)
        logging.info(f"Populated {len(matchups)} player vs player stat records")
    except Exception as e:
        logging.error(f"Error populating player vs player stats: {str(e)}")
        raise

def main():
    try:
        logging.info("Starting database setup...")
        create_tables()
        
        conn = sqlite3.connect('Data/player_stats.sqlite')
        populate_team_defensive_stats(conn)
        populate_player_vs_team_stats(conn)
        populate_player_vs_player_stats(conn)
        conn.close()
        
        logging.info("Database setup completed successfully")
    except Exception as e:
        logging.error(f"Database setup failed: {str(e)}")
        raise
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    main()
