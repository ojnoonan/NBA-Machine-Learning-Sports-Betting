import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def create_sample_data(n_players=50, n_games=82):
    """Create sample player game stats"""
    np.random.seed(42)
    
    # Generate player IDs and names
    player_ids = np.arange(1001, 1001 + n_players)
    player_names = [f"Player_{i}" for i in player_ids]
    
    # Generate team IDs
    team_ids = np.arange(1, 31)
    team_abbrevs = [f"TEAM_{i}" for i in range(1, 31)]
    
    # Generate game dates
    start_date = datetime(2023, 10, 1)
    dates = [(start_date + timedelta(days=i)) for i in range(n_games)]
    
    # Create empty list to store all game records
    all_records = []
    
    # Generate stats for each player
    for player_id, player_name in zip(player_ids, player_names):
        # Assign player to a team
        team_id = np.random.choice(team_ids)
        team_abbrev = team_abbrevs[team_id - 1]
        
        # Player's baseline stats (some players score more, some less)
        base_pts = np.random.normal(15, 5)
        base_reb = np.random.normal(6, 2)
        base_ast = np.random.normal(3, 1)
        base_min = np.random.normal(25, 5)
        
        # Generate stats for each game
        for game_date in dates:
            # Random variation in stats
            minutes = max(0, np.random.normal(base_min, 3))
            if minutes < 5:  # DNP
                continue
                
            # Basic stats
            pts = max(0, np.random.normal(base_pts, 4))
            reb = max(0, np.random.normal(base_reb, 2))
            ast = max(0, np.random.normal(base_ast, 1.5))
            
            # Shooting splits based on points
            fg3m = max(0, np.random.normal(pts * 0.2, 1))
            fgm = max(0, np.random.normal(pts * 0.4, 2))
            ftm = max(0, pts - (fgm * 2 + fg3m * 3))
            
            # Attempts
            fg3a = max(fg3m, np.random.normal(fg3m * 1.5, 1))
            fga = max(fgm, np.random.normal(fgm * 2, 2))
            fta = max(ftm, np.random.normal(ftm * 1.2, 1))
            
            # Other stats
            stl = max(0, np.random.normal(1, 0.5))
            blk = max(0, np.random.normal(0.5, 0.3))
            turnovers = max(0, np.random.normal(2, 1))
            pf = max(0, np.random.normal(2.5, 1))
            
            # Calculate percentages
            fg_pct = fgm / fga if fga > 0 else 0
            fg3_pct = fg3m / fg3a if fg3a > 0 else 0
            ft_pct = ftm / fta if fta > 0 else 0
            
            # Random opponent
            opponent_team = np.random.choice([t for t in team_ids if t != team_id])
            
            # Create game record
            record = {
                'GAME_ID': f"{game_date.strftime('%Y%m%d')}_{team_id}_{opponent_team}",
                'PLAYER_ID': player_id,
                'PLAYER_NAME': player_name,
                'TEAM_ID': team_id,
                'TEAM_ABBREVIATION': team_abbrev,
                'TEAM_CITY': f"City_{team_id}",
                'MIN': minutes,
                'FGM': int(fgm),
                'FGA': int(fga),
                'FG_PCT': fg_pct,
                'FG3M': int(fg3m),
                'FG3A': int(fg3a),
                'FG3_PCT': fg3_pct,
                'FTM': int(ftm),
                'FTA': int(fta),
                'FT_PCT': ft_pct,
                'OREB': int(np.random.normal(reb * 0.3, 1)),
                'DREB': int(np.random.normal(reb * 0.7, 1)),
                'REB': int(reb),
                'AST': int(ast),
                'STL': int(stl),
                'BLK': int(blk),
                'TURNOVERS': int(turnovers),
                'PF': int(pf),
                'PTS': int(pts),
                'PLUS_MINUS': int(np.random.normal(0, 10)),
                'GAME_DATE': game_date.strftime('%Y-%m-%d'),
                'Season': 2023,
                'days_rest': np.random.choice([1, 2, 3, 4]),
                'OPPONENT_TEAM_ID': opponent_team,
                'points': int(pts),
                'rebounds': int(reb),
                'assists': int(ast),
                'threes': int(fg3m)
            }
            
            all_records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    return df

def create_database():
    """Create SQLite database and save sample data"""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_root, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Create and connect to database
    db_path = os.path.join(data_dir, 'player_stats.sqlite')
    conn = sqlite3.connect(db_path)
    
    # Generate sample data
    print("Generating sample data...")
    df = create_sample_data()
    
    # Save to database
    print("Saving to database...")
    df.to_sql('player_game_stats', conn, if_exists='replace', index=False)
    print(f"Saved {len(df)} records to database")
    
    conn.close()
    print("Database created successfully!")

if __name__ == "__main__":
    create_database()
