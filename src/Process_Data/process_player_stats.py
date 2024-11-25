import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import psutil
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure tqdm to write to stderr for better progress bar display
tqdm.monitor_interval = 0

class PlayerStatsProcessor:
    def __init__(self, source_db_path=None, target_db_path=None):
        # Set default paths relative to project root if not provided
        if source_db_path is None or target_db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.source_db_path = os.path.join(project_root, 'Data', 'nba.sqlite')
            self.target_db_path = os.path.join(project_root, 'Data', 'player_stats.sqlite')
        else:
            self.source_db_path = source_db_path
            self.target_db_path = target_db_path
            
        logging.info(f"Source DB: {self.source_db_path}")
        logging.info(f"Target DB: {self.target_db_path}")
        
        self.plays = None
        self.team_cache = {}  # Cache for team assignments
        
        # Initialize database
        self.init_target_db()

    def init_target_db(self):
        """Initialize the target database with required tables"""
        logging.info("Initializing target database...")
        
        try:
            conn = sqlite3.connect(self.target_db_path)
            cursor = conn.cursor()
            
            # Create player_game_stats table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_game_stats (
                game_id TEXT,
                game_date TEXT,
                player_id INTEGER,
                team_id INTEGER,
                opponent_team_id INTEGER,
                is_home BOOLEAN,
                points INTEGER,
                assists INTEGER,
                rebounds_off INTEGER,
                rebounds_def INTEGER,
                rebounds_total INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                fouls INTEGER,
                fga INTEGER,
                fgm INTEGER,
                fg3a INTEGER,
                fg3m INTEGER,
                fta INTEGER,
                ftm INTEGER,
                true_shooting_pct REAL,
                points_per_shot REAL,
                usage_rate REAL,
                stocks INTEGER,
                assist_to_turnover REAL,
                three_point_rate REAL,
                efficiency REAL,
                points_from_3 INTEGER,
                points_from_2 INTEGER,
                points_from_ft INTEGER,
                PRIMARY KEY (game_id, player_id)
            )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_game_stats_player 
            ON player_game_stats(player_id, game_date)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_game_stats_game 
            ON player_game_stats(game_id)
            """)
            
            conn.commit()
            logging.info("Target database initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing target database: {str(e)}")
            raise
        finally:
            conn.close()
    
    def _load_checkpoint(self):
        """Load the checkpoint from the database to track processed games"""
        try:
            conn = sqlite3.connect(self.target_db_path)
            cursor = conn.cursor()
            
            # Create checkpoint table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_checkpoint (
                    last_processed_game_id TEXT,
                    last_processed_date TEXT
                )
            """)
            conn.commit()
            
            # Get the last processed game
            cursor.execute("SELECT last_processed_game_id, last_processed_date FROM processing_checkpoint LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                checkpoint = {
                    'last_game_id': result[0],
                    'last_date': result[1]
                }
                logging.info(f"Loaded checkpoint: Last processed game {checkpoint['last_game_id']} on {checkpoint['last_date']}")
            else:
                checkpoint = {
                    'last_game_id': None,
                    'last_date': None
                }
                logging.info("No checkpoint found - will process all games")
                
            return checkpoint
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return {
                'last_game_id': None,
                'last_date': None
            }
        finally:
            conn.close()

    def _save_checkpoint(self, game_id, process_date):
        """Save the current processing checkpoint to the database"""
        try:
            conn = sqlite3.connect(self.target_db_path)
            cursor = conn.cursor()
            
            # Update or insert checkpoint
            cursor.execute("""
                INSERT OR REPLACE INTO processing_checkpoint 
                (last_processed_game_id, last_processed_date) 
                VALUES (?, ?)
            """, (game_id, process_date))
            
            conn.commit()
            logging.info(f"Saved checkpoint: game_id={game_id}, date={process_date}")
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            raise
        finally:
            conn.close()
            
    def get_play_by_play_data(self, game_id):
        """Get play-by-play data for a game"""
        try:
            conn = sqlite3.connect(self.source_db_path)
            
            # First, let's inspect the table schema
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(play_by_play)")
            columns = cursor.fetchall()
            logging.debug(f"Play-by-play table schema: {columns}")
            
            # Now get the play data with explicit column selection
            query = """
                SELECT 
                    p.*,
                    g.game_date
                FROM play_by_play p
                JOIN game g ON p.game_id = g.game_id
                WHERE p.game_id = ?
                LIMIT 5
            """
            
            # Get a sample of rows to inspect
            cursor.execute(query, [game_id])
            sample_rows = cursor.fetchall()
            if sample_rows:
                column_names = [description[0] for description in cursor.description]
                logging.debug(f"Column names: {column_names}")
                logging.debug(f"Sample row: {dict(zip(column_names, sample_rows[0]))}")
            
            # Now get all the data
            plays_df = pd.read_sql_query("""
                SELECT 
                    p.*,
                    g.game_date
                FROM play_by_play p
                JOIN game g ON p.game_id = g.game_id
                WHERE p.game_id = ?
            """, conn, params=[game_id])
            
            if plays_df.empty:
                logging.warning(f"No play-by-play data found for game {game_id}")
                return None
                
            logging.debug(f"Game {game_id} data types: {plays_df.dtypes}")
            logging.debug(f"Game clock column: {plays_df['pctimestring'].head()}")
            
            return plays_df
            
        except Exception as e:
            logging.error(f"Error getting play-by-play data for game {game_id}: {str(e)}")
            return None
        finally:
            conn.close()
            
    def convert_game_clock(self, clock_str):
        """Convert game clock string to seconds"""
        try:
            if pd.isna(clock_str):
                return 0
                
            # Debug the input type and value
            logging.debug(f"Converting game clock: {clock_str} (type: {type(clock_str)})")
            
            if not isinstance(clock_str, str):
                return 0
                
            # Handle different time formats (MM:SS or MM:SS.S)
            parts = clock_str.split(':')
            if len(parts) != 2:
                logging.debug(f"Invalid game clock format: {clock_str}")
                return 0
                
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + int(seconds)
        except (ValueError, AttributeError, IndexError) as e:
            logging.debug(f"Error converting game clock '{clock_str}': {str(e)}")
            return 0
            
    def process_games(self):
        """Process all games into player statistics"""
        try:
            # Get unique games
            unique_games = self.plays['game_id'].unique()
            total_games = len(unique_games)
            logging.info(f"\nFound {total_games} games to process")
            
            # Load checkpoint if exists
            checkpoint = self._load_checkpoint()
            processed_games = set()
            if checkpoint['last_game_id'] is not None:
                processed_games = set([checkpoint['last_game_id']])
                logging.info(f"Loaded {len(processed_games)} games from checkpoint")
            
            # Process each game
            all_player_stats = []
            games_with_stats = 0
            total_players = 0
            
            for i, game_id in enumerate(unique_games):
                if game_id in processed_games:
                    continue
                    
                try:
                    # Get plays for this game
                    game_plays = self.plays[self.plays['game_id'] == game_id].copy()
                    logging.debug(f"\nProcessing game {game_id}")
                    logging.debug(f"Found {len(game_plays)} plays")
                    
                    # Process game stats
                    player_stats = self.process_game_stats(game_plays)
                    
                    if player_stats:
                        all_player_stats.extend(player_stats)
                        games_with_stats += 1
                        total_players += len(player_stats)
                        processed_games.add(game_id)
                        
                        # Save checkpoint every 100 games
                        if games_with_stats % 100 == 0:
                            self._save_checkpoint(game_id, game_plays['game_date'].iloc[0])
                    else:
                        logging.debug(f"No stats generated for game {game_id}")
                        
                except Exception as e:
                    logging.error(f"Error processing game {game_id}: {str(e)}")
                    continue
                    
                # Print progress
                progress = (i + 1) / total_games * 100
                print(f"\rProcessing: {progress:.1f}% ({i+1}/{total_games} games, {total_players} players)", 
                    end='', flush=True)
                    
            print(f"\n\nSuccessfully processed {games_with_stats} games with stats")
            print(f"Generated stats for {total_players} player-game combinations")
            print(f"Total stats records: {len(all_player_stats)}")
            
            return pd.DataFrame(all_player_stats) if all_player_stats else None
            
        except Exception as e:
            print(f"\nError in process_games: {str(e)}", flush=True)
            raise

    def get_game_team_assignments(self, game_id):
        """Get team assignments for all players in a game"""
        if game_id in self.team_cache:
            return self.team_cache[game_id]
        
        try:
            conn = sqlite3.connect(self.source_db_path)
            cursor = conn.cursor()
            
            # Get game info first
            cursor.execute("""
                SELECT 
                    CAST(team_id_home AS INTEGER) as team_id_home,
                    CAST(team_id_away AS INTEGER) as team_id_away
                FROM game
                WHERE game_id = ?
            """, [game_id])
            
            game_result = cursor.fetchone()
            if not game_result:
                logging.error(f"Game {game_id} not found in game table")
                return {}
                
            home_team_id, away_team_id = game_result
            logging.info(f"\nGame {game_id} - Home Team: {home_team_id}, Away Team: {away_team_id}")
            
            # Get player appearances with team info
            cursor.execute("""
                SELECT 
                    CAST(player1_id AS INTEGER) as player_id,
                    CAST(player1_team_id AS INTEGER) as team_id,
                    COUNT(*) as appearances
                FROM play_by_play
                WHERE game_id = ?
                AND player1_id IS NOT NULL 
                AND player1_id != 0
                AND player1_team_id IS NOT NULL
                GROUP BY player1_id, player1_team_id
            """, [game_id])
            
            # Process results to get most frequent team for each player
            player_assignments = {}
            all_results = cursor.fetchall()
            logging.info(f"Found {len(all_results)} player-team combinations")
            
            for player_id, team_id, appearances in all_results:
                try:
                    if pd.isna(player_id) or player_id == 0 or pd.isna(team_id):
                        logging.debug(f"Skipping invalid data: player_id={player_id}, team_id={team_id}")
                        continue
                    
                    # Handle potential float values
                    player_id = int(float(player_id)) if isinstance(player_id, str) else int(player_id)
                    team_id = int(float(team_id)) if isinstance(team_id, str) else int(team_id)
                    
                    # Skip invalid IDs
                    if player_id <= 0 or team_id <= 0:
                        logging.debug(f"Skipping invalid ID - player_id: {player_id}, team_id: {team_id}")
                        continue
                    
                    # Update assignment if this one has more appearances
                    if player_id not in player_assignments or appearances > player_assignments[player_id][1]:
                        is_home = (team_id == home_team_id)
                        player_assignments[player_id] = (is_home, appearances)
                        
                        # Log the assignment
                        team_type = "home" if is_home else "away"
                        logging.info(f"Player {player_id} assigned to {team_type} team (team_id: {team_id}) with {appearances} appearances")
                
                except (ValueError, TypeError) as e:
                    logging.error(f"Error processing player assignment - player_id: {player_id}, team_id: {team_id}, error: {str(e)}")
                    continue
        
            # Convert to final format
            team_assignments = {player_id: assignment[0] for player_id, assignment in player_assignments.items()}
            
            # Log final assignments
            logging.info(f"Final team assignments for game {game_id}:")
            for player_id, is_home in team_assignments.items():
                team_type = "home" if is_home else "away"
                logging.info(f"Player {player_id}: {team_type} team")
        
            # Cache and return
            self.team_cache[game_id] = team_assignments
            return team_assignments
        
        except Exception as e:
            logging.error(f"Error getting team assignments for game {game_id}: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

    def is_player_home_team(self, player_id, game_id):
        """Determine if a player is on the home team for a given game"""
        try:
            if pd.isna(player_id) or player_id == 0:
                logging.debug(f"Skipping invalid player_id: {player_id}")
                return None
            
            # Convert player_id to integer
            player_id = int(player_id)
            
            # Get team assignments for all players in this game
            team_assignments = self.get_game_team_assignments(game_id)
            
            # Look up this player's assignment
            if player_id in team_assignments:
                return team_assignments[player_id]
            
            logging.warning(f"Could not determine team for player {player_id} in game {game_id}")
            return None
            
        except Exception as e:
            logging.error(f"Error in is_player_home_team for player {player_id} in game {game_id}: {str(e)}")
            return None

    def get_player_positions(self, player_ids, game_date):
        """Get player positions for matchup analysis"""
        try:
            # Connect to source database to get player info
            conn = sqlite3.connect(self.source_db_path)
            cursor = conn.cursor()
            
            # Query player info from source database
            placeholders = ','.join(['?' for _ in player_ids])
            query = f"""
                SELECT 
                    player_id,
                    position,
                    first_name,
                    last_name,
                    height,
                    weight,
                    birth_date
                FROM players 
                WHERE player_id IN ({placeholders})
            """
            
            cursor.execute(query, list(player_ids))
            player_info = cursor.fetchall()
            
            # Insert player info into target database
            target_conn = sqlite3.connect(self.target_db_path)
            target_cursor = target_conn.cursor()
            
            target_cursor.executemany("""
                INSERT OR REPLACE INTO player_info 
                (player_id, position, first_name, last_name, height, weight, birth_date, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(p[0], p[1], p[2], p[3], p[4], p[5], p[6], game_date) for p in player_info])
            
            target_conn.commit()
            
            # Return positions dictionary
            positions = {p[0]: p[1] for p in player_info}
            return positions
            
        except Exception as e:
            logging.error(f"Error getting player positions: {str(e)}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()
            if 'target_conn' in locals():
                target_conn.close()
            
    def update_team_defensive_stats(self, team_id, game_date, defensive_rating, 
                                  opp_points, opp_fg_pct, opp_fg3_pct, opp_assists,
                                  blocks, steals):
        """Update team defensive statistics"""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        try:
            # Calculate rolling averages
            window = 10
            query = """
                WITH recent_games AS (
                    SELECT *
                    FROM team_defensive_stats
                    WHERE team_id = ? AND game_date < ?
                    ORDER BY game_date DESC
                    LIMIT ?
                )
                SELECT 
                    AVG(defensive_rating) as def_rating_avg,
                    AVG(opponent_points) as points_avg,
                    AVG(opponent_fg_pct) as fg_pct_avg,
                    AVG(opponent_fg3_pct) as fg3_pct_avg,
                    AVG(opponent_assists) as assists_avg,
                    AVG(blocks) as blocks_avg,
                    AVG(steals) as steals_avg
                FROM recent_games
            """
            
            cursor.execute(query, [team_id, game_date, window])
            averages = cursor.fetchone()
            
            if averages[0] is not None:
                def_rating_avg = (averages[0] * window + defensive_rating) / (window + 1)
                points_avg = (averages[1] * window + opp_points) / (window + 1)
                fg_pct_avg = (averages[2] * window + opp_fg_pct) / (window + 1)
                fg3_pct_avg = (averages[3] * window + opp_fg3_pct) / (window + 1)
                assists_avg = (averages[4] * window + opp_assists) / (window + 1)
                blocks_avg = (averages[5] * window + blocks) / (window + 1)
                steals_avg = (averages[6] * window + steals) / (window + 1)
            else:
                def_rating_avg = defensive_rating
                points_avg = opp_points
                fg_pct_avg = opp_fg_pct
                fg3_pct_avg = opp_fg3_pct
                assists_avg = opp_assists
                blocks_avg = blocks
                steals_avg = steals
            
            # Insert or update stats
            cursor.execute("""
                INSERT OR REPLACE INTO team_defensive_stats
                (team_id, game_date, defensive_rating, opponent_points,
                 opponent_fg_pct, opponent_fg3_pct, opponent_assists,
                 blocks, steals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [team_id, game_date, def_rating_avg, points_avg,
                  fg_pct_avg, fg3_pct_avg, assists_avg, blocks_avg, steals_avg])
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating team defensive stats: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    def update_player_vs_team_stats(self, player_id, opponent_team_id, points, assists, rebounds):
        """Update player's historical performance against specific teams"""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        try:
            # Get existing stats
            cursor.execute("""
                SELECT games_played, points_avg, assists_avg, rebounds_avg
                FROM player_vs_team_stats
                WHERE player_id = ? AND opponent_team_id = ?
            """, [player_id, opponent_team_id])
            
            existing = cursor.fetchone()
            
            if existing:
                games = existing[0] + 1
                points_avg = (existing[1] * existing[0] + points) / games
                assists_avg = (existing[2] * existing[0] + assists) / games
                rebounds_avg = (existing[3] * existing[0] + rebounds) / games
            else:
                games = 1
                points_avg = points
                assists_avg = assists
                rebounds_avg = rebounds
            
            # Calculate efficiency
            efficiency = points + rebounds + assists
            
            # Insert or update stats
            cursor.execute("""
                INSERT OR REPLACE INTO player_vs_team_stats
                (player_id, opponent_team_id, games_played,
                 points_avg, assists_avg, rebounds_avg,
                 efficiency_avg, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, DATE('now'))
            """, [player_id, opponent_team_id, games,
                  points_avg, assists_avg, rebounds_avg,
                  efficiency])
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating player vs team stats: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    def update_player_vs_player_stats(self, player_id, opponent_id, game_date,
                                    points, assists, rebounds, blocks, steals,
                                    minutes_matched):
        """Update player's performance statistics against specific opponents"""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        try:
            # Get existing matchup stats
            cursor.execute("""
                SELECT 
                    matchups_count,
                    total_minutes,
                    points_per_36,
                    assists_per_36,
                    rebounds_per_36,
                    blocks_per_36,
                    steals_per_36,
                    defensive_rating
                FROM player_vs_player_stats
                WHERE player_id = ? AND opponent_id = ?
            """, [player_id, opponent_id])
            
            existing = cursor.fetchone()
            
            # Calculate per-36 minutes stats for this game
            minutes_factor = 36 / minutes_matched if minutes_matched > 0 else 0
            game_points_per_36 = points * minutes_factor
            game_assists_per_36 = assists * minutes_factor
            game_rebounds_per_36 = rebounds * minutes_factor
            game_blocks_per_36 = blocks * minutes_factor
            game_steals_per_36 = steals * minutes_factor
            
            # Calculate defensive rating (points allowed per 100 possessions)
            possessions = (minutes_matched / 48) * 100  # Estimate possessions based on minutes
            game_defensive_rating = (points / possessions) * 100 if possessions > 0 else 0
            
            if existing:
                # Update rolling averages
                matchups = existing[0] + 1
                total_minutes = existing[1] + minutes_matched
                
                # Weight new stats based on minutes played
                weight_old = existing[1] / total_minutes
                weight_new = minutes_matched / total_minutes
                
                points_per_36 = (existing[2] * weight_old) + (game_points_per_36 * weight_new)
                assists_per_36 = (existing[3] * weight_old) + (game_assists_per_36 * weight_new)
                rebounds_per_36 = (existing[4] * weight_old) + (game_rebounds_per_36 * weight_new)
                blocks_per_36 = (existing[5] * weight_old) + (game_blocks_per_36 * weight_new)
                steals_per_36 = (existing[6] * weight_old) + (game_steals_per_36 * weight_new)
                defensive_rating = (existing[7] * weight_old) + (game_defensive_rating * weight_new)
            else:
                matchups = 1
                total_minutes = minutes_matched
                points_per_36 = game_points_per_36
                assists_per_36 = game_assists_per_36
                rebounds_per_36 = game_rebounds_per_36
                blocks_per_36 = game_blocks_per_36
                steals_per_36 = game_steals_per_36
                defensive_rating = game_defensive_rating
            
            # Insert or update matchup stats
            cursor.execute("""
                INSERT OR REPLACE INTO player_vs_player_stats
                (player_id, opponent_id, matchups_count, total_minutes,
                 points_per_36, assists_per_36, rebounds_per_36,
                 blocks_per_36, steals_per_36, defensive_rating,
                 last_matchup_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [player_id, opponent_id, matchups, total_minutes,
                  points_per_36, assists_per_36, rebounds_per_36,
                  blocks_per_36, steals_per_36, defensive_rating,
                  game_date])
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating player vs player stats: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
            
    def calculate_advanced_stats(self, stats_df):
        """Calculate advanced statistics for each player game"""
        # Add advanced stats columns
        advanced_stats = stats_df.copy()
        
        # Scoring Efficiency
        advanced_stats['true_shooting_pct'] = (
            stats_df['points'] / (2 * (stats_df['fga'] + 0.44 * stats_df['fta']))
        ).fillna(0)
        
        advanced_stats['points_per_shot'] = (
            stats_df['points'] / stats_df['fga']
        ).fillna(0)
        
        # Usage and Involvement
        advanced_stats['usage_rate'] = (
            (stats_df['fga'] + stats_df['fta'] * 0.44 + stats_df['turnovers']) / 
            (stats_df['minutes'] if 'minutes' in stats_df.columns else 24)  # Assuming average minutes if not available
        ).fillna(0)
        
        # Versatility Metrics
        advanced_stats['stocks'] = stats_df['steals'] + stats_df['blocks']  # Steals + Blocks
        advanced_stats['rebounds_total'] = stats_df['rebounds_off'] + stats_df['rebounds_def']
        
        # Efficiency Metrics
        advanced_stats['assist_to_turnover'] = (
            stats_df['assists'] / stats_df['turnovers']
        ).fillna(0)
        
        # Shooting Splits
        advanced_stats['points_from_3'] = stats_df['fg3m'] * 3
        advanced_stats['points_from_2'] = stats_df['fgm'] * 2 - advanced_stats['points_from_3']
        advanced_stats['points_from_ft'] = stats_df['ftm']
        
        advanced_stats['three_point_rate'] = (
            stats_df['fg3a'] / stats_df['fga']
        ).fillna(0)
        
        # Performance Consistency
        advanced_stats['efficiency'] = (
            stats_df['points'] + stats_df['rebounds_off'] + stats_df['rebounds_def'] +
            stats_df['assists'] + stats_df['steals'] + stats_df['blocks'] -
            (stats_df['fga'] - stats_df['fgm']) - (stats_df['fta'] - stats_df['ftm']) -
            stats_df['turnovers']
        )
        
        return advanced_stats
    
    def save_player_stats(self, stats_df):
        """Save player statistics to database"""
        try:
            # Filter out invalid player IDs and games with no stats
            stats_df = stats_df[
                (stats_df['player_id'] > 0) & 
                ((stats_df['points'] > 0) | 
                 (stats_df['assists'] > 0) | 
                 (stats_df['rebounds_off'] > 0) |
                 (stats_df['rebounds_def'] > 0))
            ].copy()
            
            # Calculate advanced stats
            advanced_stats_df = self.calculate_advanced_stats(stats_df)
            
            # Sort by player and date for proper rolling calculations
            advanced_stats_df = advanced_stats_df.sort_values(['player_id', 'game_date'])
            
            # Calculate rolling averages per player
            rolling_stats = []
            for player_id, player_games in advanced_stats_df.groupby('player_id'):
                # Calculate rolling averages with a 10-game window
                rolling_df = pd.DataFrame({
                    'player_id': player_id,
                    'game_date': player_games['game_date'],
                    # Basic Stats Rolling Averages
                    'points_avg': player_games['points'].rolling(10, min_periods=1).mean(),
                    'assists_avg': player_games['assists'].rolling(10, min_periods=1).mean(),
                    'rebounds_avg': player_games['rebounds_total'].rolling(10, min_periods=1).mean(),
                    'steals_avg': player_games['steals'].rolling(10, min_periods=1).mean(),
                    'blocks_avg': player_games['blocks'].rolling(10, min_periods=1).mean(),
                    'turnovers_avg': player_games['turnovers'].rolling(10, min_periods=1).mean(),
                    'fouls_avg': player_games['fouls'].rolling(10, min_periods=1).mean(),
                    
                    # Shooting Percentages
                    'fg_pct': (player_games['fgm'].rolling(10, min_periods=1).sum() / 
                              player_games['fga'].rolling(10, min_periods=1).sum()).fillna(0),
                    'fg3_pct': (player_games['fg3m'].rolling(10, min_periods=1).sum() / 
                               player_games['fg3a'].rolling(10, min_periods=1).sum()).fillna(0),
                    'ft_pct': (player_games['ftm'].rolling(10, min_periods=1).sum() / 
                              player_games['fta'].rolling(10, min_periods=1).sum()).fillna(0),
                    
                    # Advanced Stats Rolling Averages
                    'true_shooting_avg': player_games['true_shooting_pct'].rolling(10, min_periods=1).mean(),
                    'points_per_shot_avg': player_games['points_per_shot'].rolling(10, min_periods=1).mean(),
                    'usage_rate_avg': player_games['usage_rate'].rolling(10, min_periods=1).mean(),
                    'stocks_avg': player_games['stocks'].rolling(10, min_periods=1).mean(),
                    'ast_to_tov_avg': player_games['assist_to_turnover'].rolling(10, min_periods=1).mean(),
                    'three_point_rate_avg': player_games['three_point_rate'].rolling(10, min_periods=1).mean(),
                    'efficiency_avg': player_games['efficiency'].rolling(10, min_periods=1).mean(),
                    
                    # Scoring Distribution Averages
                    'points_from_3_avg': player_games['points_from_3'].rolling(10, min_periods=1).mean(),
                    'points_from_2_avg': player_games['points_from_2'].rolling(10, min_periods=1).mean(),
                    'points_from_ft_avg': player_games['points_from_ft'].rolling(10, min_periods=1).mean(),
                    
                    # Performance Volatility
                    'points_std': player_games['points'].rolling(10, min_periods=3).std().fillna(0),
                    'efficiency_std': player_games['efficiency'].rolling(10, min_periods=3).std().fillna(0)
                })
                rolling_stats.append(rolling_df)
            
            rolling_df = pd.concat(rolling_stats, ignore_index=True)
            
            # Log some sample data
            logging.info(f"Processed {len(advanced_stats_df)} player game stats")
            logging.info("Sample of processed stats (including advanced metrics):")
            print(advanced_stats_df.head().to_string())
            
            logging.info(f"Processed {len(rolling_df)} rolling stats")
            logging.info("Sample of rolling stats (including advanced metrics):")
            print(rolling_df.head().to_string())
            
            # Save to database
            conn = sqlite3.connect(self.target_db_path)
            
            # Save raw and advanced stats
            advanced_stats_df.to_sql('player_game_stats', conn, if_exists='replace', index=False)
            
            # Save enhanced rolling averages
            rolling_df.to_sql('player_rolling_stats', conn, if_exists='replace', index=False)
            
            # Create indices for faster querying
            conn.execute('CREATE INDEX IF NOT EXISTS player_game_idx ON player_game_stats(player_id, game_date)')
            conn.execute('CREATE INDEX IF NOT EXISTS player_rolling_idx ON player_rolling_stats(player_id, game_date)')
            
            conn.close()
            logging.info("Processing complete!")
            
        except Exception as e:
            logging.error(f"Error saving stats: {str(e)}")
            raise

    def load_play_by_play_data(self, start_date=None, end_date=None):
        """Load play-by-play data from the database"""
        try:
            logging.info("Loading play-by-play data...")
            logging.info("Connecting to source database...")
            
            # Now load the data in chunks
            query = """
                SELECT DISTINCT
                    p.game_id,
                    g.game_date,
                    CAST(p.eventmsgtype AS INTEGER) as eventmsgtype,
                    CAST(p.player1_id AS INTEGER) as player_id,
                    CAST(p.player1_team_id AS INTEGER) as team_id,
                    p.homedescription,
                    p.visitordescription,
                    CAST(p.player2_id AS INTEGER) as player2_id,
                    CAST(p.player3_id AS INTEGER) as player3_id,
                    CAST(p.eventnum AS INTEGER) as eventnum,
                    CAST(p.period AS INTEGER) as period,
                    p.pctimestring,
                    p.neutraldescription
                FROM play_by_play p
                JOIN game g ON p.game_id = g.game_id
                WHERE g.season_type = 'Regular Season'
                AND p.player1_id IS NOT NULL  -- Only include plays with a player involved
                AND p.eventmsgtype IN (1,2,3,4,5,6)  -- Only include relevant event types
            """
            
            if start_date:
                query += f" AND g.game_date >= '{start_date}'"
            if end_date:
                query += f" AND g.game_date <= '{end_date}'"
                
            query += " ORDER BY p.game_id, p.period, p.eventnum"
            
            # First get total count
            count_query = f"""
                SELECT COUNT(*) FROM ({query}) as subquery
            """
            
            with sqlite3.connect(self.source_db_path) as conn:
                total_rows = pd.read_sql_query(count_query, conn).iloc[0, 0]
            
            logging.info(f"Found {total_rows} plays to process")
            
            # Now load in chunks
            logging.info("Loading play-by-play data in chunks...")
            chunks = []
            chunk_size = 100000
            
            with sqlite3.connect(self.source_db_path) as conn:
                # Use tqdm to show progress
                with tqdm(total=total_rows, desc="Loading plays", unit="plays") as pbar:
                    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
                        chunks.append(chunk)
                        pbar.update(len(chunk))
                        
                        # Log memory usage every 10 chunks
                        if len(chunks) % 10 == 0:
                            process = psutil.Process()
                            memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
                            logging.info(f"Memory usage after {len(chunks)} chunks: {memory_usage:.2f} MB")
            
            logging.info("Concatenating chunks...")
            pbp_df = pd.concat(chunks, ignore_index=True)
            logging.info(f"Successfully loaded {len(pbp_df)} plays")
            
            return pbp_df
            
        except Exception as e:
            logging.error(f"Error in load_play_by_play_data: {str(e)}")
            raise

    def process_data(self, start_date=None, end_date=None):
        """Process play-by-play data into player statistics"""
        try:
            logging.info("Loading play-by-play data...")
            pbp_df = self.load_play_by_play_data(start_date, end_date)
            
            if pbp_df.empty:
                logging.info("No play-by-play data to process")
                return pd.DataFrame(), pd.DataFrame()
            
            logging.info("Processing play-by-play events into player statistics...")
            checkpoint = self._load_checkpoint()
            self.plays = pbp_df
            self.process_games()
            
            # Load processed stats from database
            conn = sqlite3.connect(self.target_db_path)
            player_stats_df = pd.read_sql_query("""
            SELECT * FROM player_game_stats
            ORDER BY game_date
            """, conn)
            conn.close()
            
            if player_stats_df.empty:
                logging.info("No player statistics to process")
                return pd.DataFrame(), pd.DataFrame()
            
            logging.info("Calculating rolling statistics...")
            rolling_stats_df = self.calculate_player_rolling_stats(player_stats_df)
            
            if not rolling_stats_df.empty:
                # Save rolling stats to database
                conn = sqlite3.connect(self.target_db_path)
                rolling_stats_df.to_sql('player_rolling_stats', conn, if_exists='replace', index=False)
                conn.close()
            
            return player_stats_df, rolling_stats_df
            
        except Exception as e:
            logging.error(f"Error in process_data: {str(e)}")
            raise

    def calculate_player_rolling_stats(self, player_stats_df, window=10):
        """Calculate rolling statistics for each player"""
        if player_stats_df.empty:
            return pd.DataFrame()
        
        logging.info("Calculating rolling statistics...")
        
        # Group by player and sort by date
        grouped = player_stats_df.groupby('player_id')
        rolling_stats_list = []
        
        # Get total number of players for progress bar
        total_players = len(grouped)
        pbar = tqdm(total=total_players, desc="Calculating rolling stats", position=0, leave=True, file=sys.stderr)
        
        for player_id, player_games in grouped:
            player_games = player_games.sort_values('game_date')
            
            # Calculate rolling averages
            rolling = player_games.rolling(window=window, min_periods=1)
            
            rolling_stats = pd.DataFrame({
                'player_id': player_id,
                'game_date': player_games['game_date'],
                'points_avg': rolling['points'].mean(),
                'assists_avg': rolling['assists'].mean(),
                'rebounds_avg': rolling['rebounds_total'].mean(),
                'steals_avg': rolling['steals'].mean(),
                'blocks_avg': rolling['blocks'].mean(),
                'turnovers_avg': rolling['turnovers'].mean(),
                'fouls_avg': rolling['fouls'].mean(),
                'fg_pct': rolling['fgm'].sum() / rolling['fga'].sum(),
                'fg3_pct': rolling['fg3m'].sum() / rolling['fg3a'].sum(),
                'ft_pct': rolling['ftm'].sum() / rolling['fta'].sum(),
                'true_shooting_avg': rolling['true_shooting_pct'].mean(),
                'points_per_shot_avg': rolling['points_per_shot'].mean(),
                'usage_rate_avg': rolling['usage_rate'].mean(),
                'stocks_avg': rolling['stocks'].mean(),
                'ast_to_tov_avg': rolling['assist_to_turnover'].mean(),
                'three_point_rate_avg': rolling['three_point_rate'].mean(),
                'efficiency_avg': rolling['efficiency'].mean(),
                'points_from_3_avg': rolling['points_from_3'].mean(),
                'points_from_2_avg': rolling['points_from_2'].mean(),
                'points_from_ft_avg': rolling['points_from_ft'].mean(),
                'points_std': rolling['points'].std(),
                'efficiency_std': rolling['efficiency'].std()
            })
            
            rolling_stats_list.append(rolling_stats)
            pbar.update(1)
        
        pbar.close()
        
        if not rolling_stats_list:
            return pd.DataFrame()
        
        result = pd.concat(rolling_stats_list, ignore_index=True)
        
        # Replace NaN values with 0
        result = result.fillna(0)
        
        return result

    def process_game_stats(self, plays):
        """Process play-by-play data for a single game into player statistics"""
        if plays is None or isinstance(plays, pd.DataFrame) and plays.empty:
            logging.debug("Empty plays DataFrame")
            return None
            
        game_id = plays['game_id'].iloc[0]
        game_date = plays['game_date'].iloc[0]
        
        logging.debug(f"\nProcessing game {game_id}")
        logging.debug(f"Number of plays: {len(plays)}")
        
        # Get team information from the game table
        conn = sqlite3.connect(self.source_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT team_id_home, team_id_away
                FROM game
                WHERE game_id = ?
            """, [game_id])
            
            result = cursor.fetchone()
            if result is None:
                logging.error(f"Game {game_id} not found in game table")
                return None
                
            home_team_id, visitor_team_id = result
            logging.debug(f"Home team: {home_team_id}, Away team: {visitor_team_id}")
            
        except Exception as e:
            logging.error(f"Error getting team info for game {game_id}: {str(e)}")
            return None
        finally:
            conn.close()
        
        # Combine all play descriptions for analysis
        plays['description'] = (
            plays['homedescription'].fillna('') + ' ' +
            plays['visitordescription'].fillna('') + ' ' +
            plays['neutraldescription'].fillna('')
        ).str.upper()
        
        try:
            # Convert game clock to seconds for proper ordering
            plays['game_clock_seconds'] = plays['pctimestring'].apply(self.convert_game_clock)
            
            # Sort plays by period and game clock
            plays = plays.sort_values(['period', 'game_clock_seconds'], ascending=[True, False])
            
            # Get unique players efficiently
            player_ids = pd.concat([
                plays['player_id'].dropna(),
                plays['player2_id'].dropna(),
                plays['player3_id'].dropna()
            ]).unique()
            
            logging.debug(f"Found {len(player_ids)} unique players")
            
            # Process each player's stats
            player_stats = []
            for player_id in player_ids:
                if pd.isna(player_id):
                    continue
                    
                # Get plays for this player
                player_plays = plays[
                    (plays['player_id'] == player_id) | 
                    (plays['player2_id'] == player_id)
                ]
                
                if player_plays.empty:
                    logging.debug(f"No plays found for player {player_id}")
                    continue
                    
                # Determine if player is on home or away team
                is_home = self.is_player_home_team(player_id, game_id)
                team_id = home_team_id if is_home else visitor_team_id
                opponent_team_id = visitor_team_id if is_home else home_team_id
                
                logging.debug(f"Processing player {player_id} (Team: {team_id}, Home: {is_home})")
                
                # Initialize player stats dictionary
                player_game_stats = {
                    'game_id': game_id,
                    'game_date': game_date,
                    'player_id': player_id,
                    'team_id': team_id,
                    'opponent_team_id': opponent_team_id,
                    'is_home': is_home,
                    'points': 0,
                    'assists': 0,
                    'rebounds_off': 0,
                    'rebounds_def': 0,
                    'steals': 0,
                    'blocks': 0,
                    'turnovers': 0,
                    'fouls': 0,
                    'fga': 0,
                    'fgm': 0,
                    'fg3a': 0,
                    'fg3m': 0,
                    'fta': 0,
                    'ftm': 0
                }
                
                # Process each play
                for _, play in player_plays.iterrows():
                    event_type = play['eventmsgtype']
                    desc = play['description']
                    
                    # Made Shot (EVENTMSGTYPE = 1)
                    if event_type == 1:
                        if play['player_id'] == player_id:
                            player_game_stats['fga'] += 1
                            player_game_stats['fgm'] += 1
                            if '3PT' in desc:
                                player_game_stats['fg3a'] += 1
                                player_game_stats['fg3m'] += 1
                                player_game_stats['points'] += 3
                            else:
                                player_game_stats['points'] += 2
                        elif play['player2_id'] == player_id and 'AST' in desc:
                            player_game_stats['assists'] += 1
                    
                    # Missed Shot (EVENTMSGTYPE = 2)
                    elif event_type == 2:
                        if play['player_id'] == player_id:
                            player_game_stats['fga'] += 1
                            if '3PT' in desc:
                                player_game_stats['fg3a'] += 1
                        elif play['player2_id'] == player_id and 'BLK' in desc:
                            player_game_stats['blocks'] += 1
                    
                    # Free Throw (EVENTMSGTYPE = 3)
                    elif event_type == 3 and play['player_id'] == player_id:
                        player_game_stats['fta'] += 1
                        if 'MISS' not in desc:
                            player_game_stats['ftm'] += 1
                            player_game_stats['points'] += 1
                    
                    # Rebound (EVENTMSGTYPE = 4)
                    elif event_type == 4 and play['player_id'] == player_id:
                        if any(x in desc for x in ['OFFENSIVE', 'OFF:', 'OFF.', 'OFF ']):
                            player_game_stats['rebounds_off'] += 1
                        else:
                            player_game_stats['rebounds_def'] += 1
                    
                    # Turnover (EVENTMSGTYPE = 5)
                    elif event_type == 5:
                        if play['player_id'] == player_id:
                            player_game_stats['turnovers'] += 1
                        elif play['player2_id'] == player_id and 'STL' in desc:
                            player_game_stats['steals'] += 1
                    
                    # Foul (EVENTMSGTYPE = 6)
                    elif event_type == 6 and play['player_id'] == player_id:
                        player_game_stats['fouls'] += 1
                
                # Only add stats if we have some non-zero values
                if any(v != 0 for v in player_game_stats.values() if isinstance(v, (int, float))):
                    logging.debug(f"Adding stats for player {player_id}: {player_game_stats}")
                    player_stats.append(player_game_stats)
                else:
                    logging.debug(f"No stats generated for player {player_id}")
            
            return player_stats
            
        except Exception as e:
            logging.error(f"Error processing game {game_id}: {str(e)}")
            logging.error(f"Game clock column info: {plays['pctimestring'].describe()}")
            logging.error(f"Game clock unique values: {plays['pctimestring'].unique()}")
            raise

    def process_player_stats(self):
        """Process player statistics from the source database"""
        try:
            # Initialize target database
            self.init_target_db()
            
            # Single connection for the entire process
            with sqlite3.connect(self.source_db_path) as conn:
                cursor = conn.cursor()
                
                # Get list of games to process with team info
                cursor.execute("""
                    SELECT 
                        game_id,
                        team_id_home,
                        team_id_away
                    FROM game
                    ORDER BY game_id
                    LIMIT 5  -- Testing with 5 games
                """)
                
                games = cursor.fetchall()
                if not games:
                    logging.error("No games found in the database")
                    return
                    
                logging.info(f"Processing {len(games)} games")
                
                # Store game info with proper type conversion
                game_teams = {}
                for game_id, home_id, away_id in games:
                    try:
                        home_team_id = int(float(home_id)) if home_id is not None else None
                        away_team_id = int(float(away_id)) if away_id is not None else None
                        if home_team_id and away_team_id:
                            game_teams[str(game_id)] = (home_team_id, away_team_id)
                            logging.info(f"Game {game_id}: Home={home_team_id}, Away={away_team_id}")
                        else:
                            logging.warning(f"Invalid team IDs for game {game_id}")
                    except (ValueError, TypeError) as e:
                        logging.error(f"Error processing team IDs for game {game_id}: {str(e)}")
                        continue
                
                # Process each game
                for game_id, _, _ in tqdm(games, desc="Processing games"):
                    try:
                        game_id = str(game_id)
                        if game_id not in game_teams:
                            logging.warning(f"Skipping game {game_id} - invalid team IDs")
                            continue
                            
                        home_team_id, away_team_id = game_teams[game_id]
                        
                        # Get player assignments for this game
                        cursor.execute("""
                            SELECT 
                                player1_id,
                                player1_team_id,
                                COUNT(*) as appearances
                            FROM play_by_play
                            WHERE game_id = ?
                            AND player1_id IS NOT NULL 
                            AND player1_id != 0
                            AND player1_team_id IS NOT NULL
                            GROUP BY player1_id, player1_team_id
                        """, [game_id])
                        
                        # Process player assignments
                        self.team_cache[game_id] = {}
                        for player_id, team_id, appearances in cursor.fetchall():
                            try:
                                player_id = int(float(player_id)) if player_id is not None else None
                                team_id = int(float(team_id)) if team_id is not None else None
                                
                                if player_id and team_id:
                                    is_home = (team_id == home_team_id)
                                    self.team_cache[game_id][player_id] = is_home
                                    logging.debug(f"Game {game_id}: Player {player_id} -> {'Home' if is_home else 'Away'}")
                            except (ValueError, TypeError) as e:
                                logging.error(f"Error processing player {player_id}: {str(e)}")
                                continue
                        
                        # Process game statistics
                        self.process_game_stats(game_id, cursor)
                        
                    except Exception as e:
                        logging.error(f"Error processing game {game_id}: {str(e)}")
                        continue
                
                logging.info("Completed processing player statistics")
                
        except Exception as e:
            logging.error(f"Error in process_player_stats: {str(e)}")
            raise

    def get_game_team_assignments(self, game_id):
        """Get team assignments for all players in a game"""
        game_id = str(game_id)  # Convert to string for consistency
        return self.team_cache.get(game_id, {})

    def process_game_stats(self, game_id, cursor):
        """Process play-by-play data for a single game into player statistics"""
        try:
            # Get plays for this game
            cursor.execute("""
                SELECT 
                    p.*,
                    g.game_date
                FROM play_by_play p
                JOIN game g ON p.game_id = g.game_id
                WHERE p.game_id = ?
            """, [game_id])
            
            plays = cursor.fetchall()
            
            if not plays:
                logging.warning(f"No play-by-play data found for game {game_id}")
                return None
                
            logging.debug(f"Game {game_id} data types: {pd.DataFrame(plays).dtypes}")
            logging.debug(f"Game clock column: {pd.DataFrame(plays)['pctimestring'].head()}")
            
            # Convert to DataFrame
            plays_df = pd.DataFrame(plays)
            
            # Combine all play descriptions for analysis
            plays_df['description'] = (
                plays_df['homedescription'].fillna('') + ' ' +
                plays_df['visitordescription'].fillna('') + ' ' +
                plays_df['neutraldescription'].fillna('')
            ).str.upper()
            
            # Convert game clock to seconds for proper ordering
            plays_df['game_clock_seconds'] = plays_df['pctimestring'].apply(self.convert_game_clock)
            
            # Sort plays by period and game clock
            plays_df = plays_df.sort_values(['period', 'game_clock_seconds'], ascending=[True, False])
            
            # Get unique players efficiently
            player_ids = pd.concat([
                plays_df['player1_id'].dropna(),
                plays_df['player2_id'].dropna(),
                plays_df['player3_id'].dropna()
            ]).unique()
            
            logging.debug(f"Found {len(player_ids)} unique players")
            
            # Process each player's stats
            player_stats = []
            for player_id in player_ids:
                if pd.isna(player_id):
                    continue
                    
                # Get plays for this player
                player_plays = plays_df[
                    (plays_df['player1_id'] == player_id) | 
                    (plays_df['player2_id'] == player_id)
                ]
                
                if player_plays.empty:
                    logging.debug(f"No plays found for player {player_id}")
                    continue
                    
                # Determine if player is on home or away team
                is_home = self.is_player_home_team(player_id, game_id)
                team_id = home_team_id if is_home else away_team_id
                opponent_team_id = away_team_id if is_home else home_team_id
                
                logging.debug(f"Processing player {player_id} (Team: {team_id}, Home: {is_home})")
                
                # Initialize player stats dictionary
                player_game_stats = {
                    'game_id': game_id,
                    'game_date': plays_df['game_date'].iloc[0],
                    'player_id': player_id,
                    'team_id': team_id,
                    'opponent_team_id': opponent_team_id,
                    'is_home': is_home,
                    'points': 0,
                    'assists': 0,
                    'rebounds_off': 0,
                    'rebounds_def': 0,
                    'steals': 0,
                    'blocks': 0,
                    'turnovers': 0,
                    'fouls': 0,
                    'fga': 0,
                    'fgm': 0,
                    'fg3a': 0,
                    'fg3m': 0,
                    'fta': 0,
                    'ftm': 0
                }
                
                # Process each play
                for _, play in player_plays.iterrows():
                    event_type = play['eventmsgtype']
                    desc = play['description']
                    
                    # Made Shot (EVENTMSGTYPE = 1)
                    if event_type == 1:
                        if play['player1_id'] == player_id:
                            player_game_stats['fga'] += 1
                            player_game_stats['fgm'] += 1
                            if '3PT' in desc:
                                player_game_stats['fg3a'] += 1
                                player_game_stats['fg3m'] += 1
                                player_game_stats['points'] += 3
                            else:
                                player_game_stats['points'] += 2
                        elif play['player2_id'] == player_id and 'AST' in desc:
                            player_game_stats['assists'] += 1
                    
                    # Missed Shot (EVENTMSGTYPE = 2)
                    elif event_type == 2:
                        if play['player1_id'] == player_id:
                            player_game_stats['fga'] += 1
                            if '3PT' in desc:
                                player_game_stats['fg3a'] += 1
                        elif play['player2_id'] == player_id and 'BLK' in desc:
                            player_game_stats['blocks'] += 1
                    
                    # Free Throw (EVENTMSGTYPE = 3)
                    elif event_type == 3 and play['player1_id'] == player_id:
                        player_game_stats['fta'] += 1
                        if 'MISS' not in desc:
                            player_game_stats['ftm'] += 1
                            player_game_stats['points'] += 1
                    
                    # Rebound (EVENTMSGTYPE = 4)
                    elif event_type == 4 and play['player1_id'] == player_id:
                        if any(x in desc for x in ['OFFENSIVE', 'OFF:', 'OFF.', 'OFF ']):
                            player_game_stats['rebounds_off'] += 1
                        else:
                            player_game_stats['rebounds_def'] += 1
                    
                    # Turnover (EVENTMSGTYPE = 5)
                    elif event_type == 5:
                        if play['player1_id'] == player_id:
                            player_game_stats['turnovers'] += 1
                        elif play['player2_id'] == player_id and 'STL' in desc:
                            player_game_stats['steals'] += 1
                    
                    # Foul (EVENTMSGTYPE = 6)
                    elif event_type == 6 and play['player1_id'] == player_id:
                        player_game_stats['fouls'] += 1
                
                # Only add stats if we have some non-zero values
                if any(v != 0 for v in player_game_stats.values() if isinstance(v, (int, float))):
                    logging.debug(f"Adding stats for player {player_id}: {player_game_stats}")
                    player_stats.append(player_game_stats)
                else:
                    logging.debug(f"No stats generated for player {player_id}")
            
            return player_stats
            
        except Exception as e:
            logging.error(f"Error processing game {game_id}: {str(e)}")
            logging.error(f"Game clock column info: {pd.DataFrame(plays)['pctimestring'].describe()}")
            logging.error(f"Game clock unique values: {pd.DataFrame(plays)['pctimestring'].unique()}")
            raise

def main():
    """Main function to process NBA play-by-play data"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info("Starting data processing...")
        
        # Set date range for processing
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # Last 5 years
        
        logging.info(f"Processing data from {start_date} to {end_date}")
        
        # Initialize processor with correct database paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        source_db = os.path.join(project_root, 'Data', 'nba.sqlite')
        target_db = os.path.join(project_root, 'Data', 'player_stats.sqlite')
        
        processor = PlayerStatsProcessor(source_db, target_db)
        logging.info("Starting data processing - this may take a few moments...")
        
        player_stats_df, rolling_stats_df = processor.process_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if player_stats_df.empty:
            logging.warning("No player statistics were generated")
        else:
            logging.info(f"Successfully processed {len(player_stats_df)} player-game records")
            
        if rolling_stats_df.empty:
            logging.warning("No rolling statistics were generated")
        else:
            logging.info(f"Successfully generated rolling statistics for {len(rolling_stats_df)} player-games")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
