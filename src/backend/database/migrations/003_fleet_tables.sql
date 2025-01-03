-- Migration script for fleet management system tables
-- SQLite version: 3.42.0

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Begin transaction with SERIALIZABLE isolation
BEGIN TRANSACTION;

-- Create fleet member role enum type equivalent using CHECK constraints
CREATE TABLE IF NOT EXISTS fleet_role_enum (
    role TEXT PRIMARY KEY
) WITHOUT ROWID;
INSERT OR IGNORE INTO fleet_role_enum (role) VALUES ('OWNER'), ('ADMIN'), ('MEMBER');

-- Create fleet_members table
CREATE TABLE IF NOT EXISTS fleet_members (
    id UUID PRIMARY KEY,
    fleet_id UUID NOT NULL,
    user_id UUID NOT NULL,
    role TEXT NOT NULL DEFAULT 'MEMBER',
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fleet_id) REFERENCES fleets(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role) REFERENCES fleet_role_enum(role),
    CHECK (role IN (SELECT role FROM fleet_role_enum))
);

-- Create indexes for fleet_members
CREATE UNIQUE INDEX IF NOT EXISTS idx_fleet_members_fleet_user ON fleet_members(fleet_id, user_id);
CREATE INDEX IF NOT EXISTS idx_fleet_members_user_id ON fleet_members(user_id);
CREATE INDEX IF NOT EXISTS idx_fleet_members_role ON fleet_members(role);
CREATE INDEX IF NOT EXISTS idx_fleet_members_last_active ON fleet_members(last_active_at);

-- Create fleet_sessions table
CREATE TABLE IF NOT EXISTS fleet_sessions (
    id UUID PRIMARY KEY,
    fleet_id UUID NOT NULL,
    environment_id UUID,
    session_type TEXT NOT NULL,
    session_config JSON NOT NULL DEFAULT '{}',
    mesh_state JSON NOT NULL DEFAULT '{}',
    active_devices INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fleet_id) REFERENCES fleets(id) ON DELETE CASCADE,
    FOREIGN KEY (environment_id) REFERENCES environments(id)
);

-- Create indexes for fleet_sessions
CREATE INDEX IF NOT EXISTS idx_fleet_sessions_fleet_id ON fleet_sessions(fleet_id);
CREATE INDEX IF NOT EXISTS idx_fleet_sessions_environment_id ON fleet_sessions(environment_id);
CREATE INDEX IF NOT EXISTS idx_fleet_sessions_session_type ON fleet_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_fleet_sessions_active_devices ON fleet_sessions(active_devices);

-- Create fleet_devices table
CREATE TABLE IF NOT EXISTS fleet_devices (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id UUID NOT NULL,
    device_id TEXT NOT NULL,
    device_type TEXT NOT NULL,
    device_metadata JSON NOT NULL DEFAULT '{}',
    connection_info JSON NOT NULL DEFAULT '{}',
    connection_status TEXT NOT NULL,
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES fleet_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Create indexes for fleet_devices
CREATE UNIQUE INDEX IF NOT EXISTS idx_fleet_devices_session_device ON fleet_devices(session_id, device_id);
CREATE INDEX IF NOT EXISTS idx_fleet_devices_user_id ON fleet_devices(user_id);
CREATE INDEX IF NOT EXISTS idx_fleet_devices_status ON fleet_devices(connection_status);
CREATE INDEX IF NOT EXISTS idx_fleet_devices_heartbeat ON fleet_devices(last_heartbeat);

-- Create trigger for updating timestamps
CREATE TRIGGER IF NOT EXISTS fleet_members_updated_at
AFTER UPDATE ON fleet_members
BEGIN
    UPDATE fleet_members SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS fleet_sessions_updated_at
AFTER UPDATE ON fleet_sessions
BEGIN
    UPDATE fleet_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS fleet_devices_updated_at
AFTER UPDATE ON fleet_devices
BEGIN
    UPDATE fleet_devices SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger for updating active device count
CREATE TRIGGER IF NOT EXISTS fleet_devices_count_insert
AFTER INSERT ON fleet_devices
BEGIN
    UPDATE fleet_sessions 
    SET active_devices = (
        SELECT COUNT(*) 
        FROM fleet_devices 
        WHERE session_id = NEW.session_id 
        AND connection_status = 'CONNECTED'
    )
    WHERE id = NEW.session_id;
END;

CREATE TRIGGER IF NOT EXISTS fleet_devices_count_update
AFTER UPDATE ON fleet_devices
WHEN OLD.connection_status != NEW.connection_status
BEGIN
    UPDATE fleet_sessions 
    SET active_devices = (
        SELECT COUNT(*) 
        FROM fleet_devices 
        WHERE session_id = NEW.session_id 
        AND connection_status = 'CONNECTED'
    )
    WHERE id = NEW.session_id;
END;

CREATE TRIGGER IF NOT EXISTS fleet_devices_count_delete
AFTER DELETE ON fleet_devices
BEGIN
    UPDATE fleet_sessions 
    SET active_devices = (
        SELECT COUNT(*) 
        FROM fleet_devices 
        WHERE session_id = OLD.session_id 
        AND connection_status = 'CONNECTED'
    )
    WHERE id = OLD.session_id;
END;

-- Commit transaction
COMMIT;

-- Add error handling
PRAGMA integrity_check;
PRAGMA foreign_key_check;