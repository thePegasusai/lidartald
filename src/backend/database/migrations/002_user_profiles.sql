-- SQLite 3.42.0 Migration: User Profiles and Gaming Statistics
-- Security Classification: RESTRICTED
-- Retention Policy: 30 DAYS
-- Transaction Isolation Level: SERIALIZABLE

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;

BEGIN TRANSACTION;

-- Create function for timestamp updates with validation
CREATE TRIGGER update_timestamp
AFTER UPDATE ON user_profiles
BEGIN
    UPDATE user_profiles 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Create function for JSON schema validation
CREATE TRIGGER validate_json_schema
BEFORE INSERT ON user_profiles
BEGIN
    SELECT CASE
        WHEN NEW.preferences IS NOT NULL AND json_valid(NEW.preferences) = 0 THEN
            RAISE(ROLLBACK, 'Invalid preferences JSON format')
        WHEN NEW.device_settings IS NOT NULL AND json_valid(NEW.device_settings) = 0 THEN
            RAISE(ROLLBACK, 'Invalid device_settings JSON format')
    END;
END;

-- Create user_profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id UUID UNIQUE NOT NULL,
    display_name VARCHAR(50) NOT NULL CHECK (length(display_name) >= 3 AND length(display_name) <= 50),
    avatar_url VARCHAR(255) CHECK (avatar_url IS NULL OR length(avatar_url) <= 255),
    bio TEXT CHECK (bio IS NULL OR length(bio) <= 1000),
    preferences JSON NOT NULL DEFAULT '{}' CHECK (json_valid(preferences)),
    device_settings JSON NOT NULL DEFAULT '{}' CHECK (json_valid(device_settings)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for user_profiles
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_display_name ON user_profiles(display_name);
CREATE INDEX IF NOT EXISTS idx_user_profiles_last_active ON user_profiles(last_active_at);

-- Create gaming_stats table
CREATE TABLE IF NOT EXISTS gaming_stats (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id UUID UNIQUE NOT NULL,
    games_played INTEGER DEFAULT 0 CHECK (games_played >= 0),
    games_won INTEGER DEFAULT 0 CHECK (games_won >= 0),
    total_playtime_minutes INTEGER DEFAULT 0 CHECK (total_playtime_minutes >= 0),
    environments_mapped INTEGER DEFAULT 0 CHECK (environments_mapped >= 0),
    fleet_participations INTEGER DEFAULT 0 CHECK (fleet_participations >= 0),
    achievements JSON NOT NULL DEFAULT '[]' CHECK (json_valid(achievements)),
    game_history JSON NOT NULL DEFAULT '[]' CHECK (json_valid(game_history)),
    last_game_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for gaming_stats
CREATE INDEX IF NOT EXISTS idx_gaming_stats_user_id ON gaming_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_gaming_stats_games_played ON gaming_stats(games_played);
CREATE INDEX IF NOT EXISTS idx_gaming_stats_last_game_at ON gaming_stats(last_game_at);
CREATE INDEX IF NOT EXISTS idx_gaming_stats_fleet_participations ON gaming_stats(fleet_participations);

-- Create trigger for gaming_stats timestamp updates
CREATE TRIGGER gaming_stats_updated_at
AFTER UPDATE ON gaming_stats
BEGIN
    UPDATE gaming_stats 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Create trigger for JSON validation on gaming_stats
CREATE TRIGGER validate_gaming_stats_json
BEFORE INSERT ON gaming_stats
BEGIN
    SELECT CASE
        WHEN NEW.achievements IS NOT NULL AND json_valid(NEW.achievements) = 0 THEN
            RAISE(ROLLBACK, 'Invalid achievements JSON format')
        WHEN NEW.game_history IS NOT NULL AND json_valid(NEW.game_history) = 0 THEN
            RAISE(ROLLBACK, 'Invalid game_history JSON format')
    END;
END;

-- Create cleanup trigger for 30-day retention policy
CREATE TRIGGER cleanup_inactive_profiles
AFTER UPDATE ON user_profiles
BEGIN
    DELETE FROM user_profiles 
    WHERE last_active_at < datetime('now', '-30 days');
END;

-- Create audit logging trigger
CREATE TABLE IF NOT EXISTS profile_audit_log (
    id UUID PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    record_id UUID NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changes JSON NOT NULL
);

CREATE TRIGGER audit_profile_changes
AFTER UPDATE ON user_profiles
BEGIN
    INSERT INTO profile_audit_log (table_name, operation, record_id, changes)
    VALUES ('user_profiles', 'UPDATE', NEW.id, 
        json_object('old', json_object('display_name', OLD.display_name, 'bio', OLD.bio),
                   'new', json_object('display_name', NEW.display_name, 'bio', NEW.bio)));
END;

COMMIT;

-- Error handling wrapper
PRAGMA integrity_check;
PRAGMA foreign_key_check;