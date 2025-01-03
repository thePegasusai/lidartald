-- SQLite 3.42.0 Migration Script
-- Session Management Tables with Enhanced Security and Performance Features

-- Enable foreign key constraints and strict mode
PRAGMA foreign_keys = ON;
PRAGMA strict = ON;

-- Begin transaction with SERIALIZABLE isolation
BEGIN TRANSACTION;

-- Create session type enum via check constraints since SQLite doesn't support ENUMs
CREATE TABLE session_type_enum (
    type TEXT PRIMARY KEY CHECK (
        type IN ('SOLO', 'MULTIPLAYER', 'FLEET', 'TOURNAMENT')
    )
);

INSERT INTO session_type_enum (type) VALUES 
    ('SOLO'), ('MULTIPLAYER'), ('FLEET'), ('TOURNAMENT');

-- Create session status enum via check constraints
CREATE TABLE session_status_enum (
    status TEXT PRIMARY KEY CHECK (
        status IN ('INITIALIZING', 'ACTIVE', 'PAUSED', 'COMPLETED', 'TERMINATED')
    )
);

INSERT INTO session_status_enum (status) VALUES 
    ('INITIALIZING'), ('ACTIVE'), ('PAUSED'), ('COMPLETED'), ('TERMINATED');

-- Create game_sessions table
CREATE TABLE game_sessions (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    fleet_session_id UUID REFERENCES fleet_sessions(id) ON DELETE SET NULL,
    environment_id UUID NOT NULL REFERENCES environments(id),
    session_type TEXT NOT NULL CHECK (
        session_type IN (SELECT type FROM session_type_enum)
    ),
    status TEXT NOT NULL DEFAULT 'INITIALIZING' CHECK (
        status IN (SELECT status FROM session_status_enum)
    ),
    game_config JSON NOT NULL DEFAULT '{}' CHECK (json_valid(game_config)),
    game_state JSON NOT NULL DEFAULT '{}' CHECK (json_valid(game_state)),
    max_players INTEGER NOT NULL CHECK (max_players > 0 AND max_players <= 32),
    current_players INTEGER DEFAULT 0 CHECK (current_players >= 0 AND current_players <= max_players),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create optimized indexes for game_sessions
CREATE INDEX idx_game_sessions_fleet_session 
ON game_sessions(fleet_session_id) WHERE fleet_session_id IS NOT NULL;

CREATE INDEX idx_game_sessions_environment 
ON game_sessions(environment_id);

CREATE INDEX idx_game_sessions_type_status 
ON game_sessions(session_type, status);

CREATE INDEX idx_game_sessions_active_players 
ON game_sessions(status, current_players) WHERE status = 'ACTIVE';

-- Create session_players table
CREATE TABLE session_players (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    player_state JSON NOT NULL DEFAULT '{}' CHECK (json_valid(player_state)),
    game_stats JSON NOT NULL DEFAULT '{}' CHECK (json_valid(game_stats)),
    is_active BOOLEAN NOT NULL DEFAULT true,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_active_player UNIQUE (session_id, user_id, is_active)
    CHECK (NOT is_active OR left_at IS NULL)
);

-- Create optimized indexes for session_players
CREATE UNIQUE INDEX idx_session_players_session_user 
ON session_players(session_id, user_id);

CREATE INDEX idx_session_players_user_active 
ON session_players(user_id, is_active);

CREATE INDEX idx_session_players_session_active 
ON session_players(session_id, is_active) WHERE is_active = true;

-- Create session_environments table
CREATE TABLE session_environments (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    environment_id UUID NOT NULL REFERENCES environments(id),
    modifications JSON NOT NULL DEFAULT '{}' CHECK (json_valid(modifications)),
    feature_states JSON NOT NULL DEFAULT '{}' CHECK (json_valid(feature_states)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_session_environment UNIQUE (session_id, environment_id)
);

-- Create optimized indexes for session_environments
CREATE INDEX idx_session_environments_env_modified 
ON session_environments(environment_id, updated_at);

-- Create update timestamp trigger function
CREATE TRIGGER game_sessions_updated_at 
AFTER UPDATE ON game_sessions
BEGIN
    UPDATE game_sessions 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

CREATE TRIGGER session_players_updated_at 
AFTER UPDATE ON session_players
BEGIN
    UPDATE session_players 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

CREATE TRIGGER session_environments_updated_at 
AFTER UPDATE ON session_environments
BEGIN
    UPDATE session_environments 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Create player count management trigger
CREATE TRIGGER session_players_count_update
AFTER INSERT OR UPDATE OR DELETE ON session_players
BEGIN
    UPDATE game_sessions 
    SET current_players = (
        SELECT COUNT(*) 
        FROM session_players 
        WHERE session_id = CASE
            WHEN OLD IS NOT NULL THEN OLD.session_id
            ELSE NEW.session_id
        END
        AND is_active = true
    )
    WHERE id = CASE
        WHEN OLD IS NOT NULL THEN OLD.session_id
        ELSE NEW.session_id
    END;
END;

-- Add audit logging trigger
CREATE TRIGGER session_audit_log
AFTER INSERT OR UPDATE OR DELETE ON game_sessions
BEGIN
    INSERT INTO audit_log (
        table_name,
        record_id,
        action,
        old_values,
        new_values,
        timestamp
    )
    VALUES (
        'game_sessions',
        CASE
            WHEN OLD IS NOT NULL THEN OLD.id
            ELSE NEW.id
        END,
        CASE
            WHEN OLD IS NULL THEN 'INSERT'
            WHEN NEW IS NULL THEN 'DELETE'
            ELSE 'UPDATE'
        END,
        CASE WHEN OLD IS NOT NULL THEN json_object(
            'status', OLD.status,
            'current_players', OLD.current_players,
            'game_state', OLD.game_state
        ) ELSE NULL END,
        CASE WHEN NEW IS NOT NULL THEN json_object(
            'status', NEW.status,
            'current_players', NEW.current_players,
            'game_state', NEW.game_state
        ) ELSE NULL END,
        CURRENT_TIMESTAMP
    );
END;

-- Create maintenance cleanup trigger
CREATE TRIGGER session_cleanup
AFTER UPDATE OF status ON game_sessions
WHEN NEW.status = 'COMPLETED' OR NEW.status = 'TERMINATED'
BEGIN
    UPDATE session_players
    SET is_active = false,
        left_at = CURRENT_TIMESTAMP
    WHERE session_id = NEW.id
    AND is_active = true;
    
    UPDATE game_sessions
    SET ended_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id
    AND ended_at IS NULL;
END;

-- Commit transaction
COMMIT;

-- Verify schema integrity
PRAGMA integrity_check;
PRAGMA foreign_key_check;