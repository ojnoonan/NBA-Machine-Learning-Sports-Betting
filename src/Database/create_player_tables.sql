-- Player game logs table
CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    starter INTEGER NOT NULL,  -- 1 if starter, 0 if bench
    minutes_played INTEGER NOT NULL,
    points INTEGER NOT NULL,
    rebounds INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    steals INTEGER NOT NULL,
    blocks INTEGER NOT NULL,
    turnovers INTEGER NOT NULL,
    personal_fouls INTEGER NOT NULL,
    fg_made INTEGER NOT NULL,
    fg_attempted INTEGER NOT NULL,
    fg3_made INTEGER NOT NULL,
    fg3_attempted INTEGER NOT NULL,
    ft_made INTEGER NOT NULL,
    ft_attempted INTEGER NOT NULL,
    plus_minus INTEGER NOT NULL,
    position TEXT NOT NULL,
    defender_id INTEGER,  -- ID of primary defender
    PRIMARY KEY (player_id, game_id)
);

-- Player injuries table
CREATE TABLE IF NOT EXISTS player_injuries (
    injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    injury_date DATE NOT NULL,
    return_date DATE,
    injury_type TEXT NOT NULL,  -- e.g., Ankle, Knee, Back, etc.
    severity TEXT NOT NULL,  -- Minor, Moderate, Severe
    games_missed INTEGER,
    FOREIGN KEY (player_id) REFERENCES player_game_logs(player_id)
);

-- Player lineup combinations table
CREATE TABLE IF NOT EXISTS player_lineups (
    game_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    lineup_id TEXT NOT NULL,  -- Concatenated player IDs
    minutes_played REAL NOT NULL,
    plus_minus INTEGER NOT NULL,
    points_scored INTEGER NOT NULL,
    points_allowed INTEGER NOT NULL,
    PRIMARY KEY (game_id, lineup_id)
);

-- Player defensive matchups table
CREATE TABLE IF NOT EXISTS player_matchups (
    game_id INTEGER NOT NULL,
    offensive_player_id INTEGER NOT NULL,
    defensive_player_id INTEGER NOT NULL,
    possessions INTEGER NOT NULL,
    points_allowed INTEGER NOT NULL,
    fg_made INTEGER NOT NULL,
    fg_attempted INTEGER NOT NULL,
    turnovers_forced INTEGER NOT NULL,
    PRIMARY KEY (game_id, offensive_player_id, defensive_player_id),
    FOREIGN KEY (offensive_player_id) REFERENCES player_game_logs(player_id),
    FOREIGN KEY (defensive_player_id) REFERENCES player_game_logs(player_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_player_game_logs_date ON player_game_logs(game_date);
CREATE INDEX IF NOT EXISTS idx_player_game_logs_team ON player_game_logs(team_id);
CREATE INDEX IF NOT EXISTS idx_player_injuries_date ON player_injuries(injury_date);
CREATE INDEX IF NOT EXISTS idx_player_matchups_players ON player_matchups(offensive_player_id, defensive_player_id);
