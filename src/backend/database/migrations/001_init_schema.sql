-- SQLite version 3.42.0
-- Initial database migration for TALD UNIA platform
-- Establishes core schema with comprehensive indexing and constraints

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Begin transaction with SERIALIZABLE isolation
BEGIN TRANSACTION;

-- Create role type check constraint values
CREATE TABLE role_types (
    role TEXT PRIMARY KEY
    CHECK (role IN ('GUEST', 'BASIC_USER', 'PREMIUM_USER', 'DEVELOPER', 'ADMIN'))
);

INSERT INTO role_types (role) VALUES
    ('GUEST'),
    ('BASIC_USER'),
    ('PREMIUM_USER'),
    ('DEVELOPER'),
    ('ADMIN');

-- Create fleet status type check constraint values
CREATE TABLE fleet_status_types (
    status TEXT PRIMARY KEY
    CHECK (status IN ('ACTIVE', 'INACTIVE', 'FULL', 'CLOSED'))
);

INSERT INTO fleet_status_types (status) VALUES
    ('ACTIVE'),
    ('INACTIVE'),
    ('FULL'),
    ('CLOSED');

-- Create users table
CREATE TABLE users (
    id TEXT PRIMARY KEY CHECK (length(id) = 36),
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'BASIC_USER'
        REFERENCES role_types(role),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    preferences TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(preferences)),
    is_active BOOLEAN NOT NULL DEFAULT 1,
    last_login DATETIME,
    CONSTRAINT email_format CHECK (email LIKE '%_@_%.__%')
);

-- Create scan_data table
CREATE TABLE scan_data (
    id TEXT PRIMARY KEY CHECK (length(id) = 36),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    point_cloud BLOB NOT NULL,
    metadata TEXT NOT NULL CHECK (json_valid(metadata)),
    resolution REAL NOT NULL CHECK (resolution > 0 AND resolution <= 1.0),
    scan_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    point_count INTEGER NOT NULL CHECK (point_count > 0),
    scan_area REAL CHECK (scan_area > 0),
    compression_type TEXT NOT NULL DEFAULT 'lz4'
        CHECK (compression_type IN ('none', 'lz4', 'zstd')),
    checksum TEXT NOT NULL,
    retention_days INTEGER NOT NULL DEFAULT 7
        CHECK (retention_days BETWEEN 1 AND 90)
);

-- Create environments table
CREATE TABLE environments (
    id TEXT PRIMARY KEY CHECK (length(id) = 36),
    scan_id TEXT NOT NULL REFERENCES scan_data(id) ON DELETE CASCADE,
    boundaries TEXT NOT NULL CHECK (json_valid(boundaries)),
    obstacles TEXT NOT NULL CHECK (json_valid(obstacles)),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1
);

-- Create features table
CREATE TABLE features (
    id TEXT PRIMARY KEY CHECK (length(id) = 36),
    scan_id TEXT NOT NULL REFERENCES scan_data(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    coordinates TEXT NOT NULL CHECK (json_valid(coordinates)),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT CHECK (json_valid(metadata))
);

-- Create fleets table
CREATE TABLE fleets (
    id TEXT PRIMARY KEY CHECK (length(id) = 36),
    host_id TEXT NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'ACTIVE'
        REFERENCES fleet_status_types(status),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    max_devices INTEGER NOT NULL DEFAULT 32
        CHECK (max_devices BETWEEN 2 AND 32),
    current_devices INTEGER NOT NULL DEFAULT 1
        CHECK (current_devices BETWEEN 1 AND 32),
    metadata TEXT CHECK (json_valid(metadata)),
    CONSTRAINT valid_device_count 
        CHECK (current_devices <= max_devices)
);

-- Create fleet_members table
CREATE TABLE fleet_members (
    fleet_id TEXT NOT NULL REFERENCES fleets(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    joined_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL DEFAULT 'MEMBER'
        CHECK (role IN ('HOST', 'MEMBER')),
    PRIMARY KEY (fleet_id, user_id)
);

-- Create indexes for optimal query performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = 1;

CREATE INDEX idx_scan_data_user_id ON scan_data(user_id);
CREATE INDEX idx_scan_data_scan_time ON scan_data(scan_time);
CREATE INDEX idx_scan_data_resolution ON scan_data(resolution);
CREATE INDEX idx_scan_data_point_count ON scan_data(point_count);
CREATE INDEX idx_scan_data_retention ON scan_data(created_at, retention_days);

CREATE INDEX idx_environments_scan_id ON environments(scan_id);
CREATE INDEX idx_environments_active ON environments(is_active) WHERE is_active = 1;

CREATE INDEX idx_features_scan_id ON features(scan_id);
CREATE INDEX idx_features_type ON features(type);
CREATE INDEX idx_features_confidence ON features(confidence);

CREATE INDEX idx_fleets_host_id ON fleets(host_id);
CREATE INDEX idx_fleets_status ON fleets(status);
CREATE INDEX idx_fleet_members_user_id ON fleet_members(user_id);

-- Create trigger for updating timestamps
CREATE TRIGGER update_timestamp_users
AFTER UPDATE ON users
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

CREATE TRIGGER update_timestamp_environments
AFTER UPDATE ON environments
BEGIN
    UPDATE environments SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

CREATE TRIGGER update_timestamp_fleets
AFTER UPDATE ON fleets
BEGIN
    UPDATE fleets SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- Create trigger for fleet member count management
CREATE TRIGGER fleet_member_count_insert
AFTER INSERT ON fleet_members
BEGIN
    UPDATE fleets 
    SET current_devices = current_devices + 1
    WHERE id = NEW.fleet_id;
END;

CREATE TRIGGER fleet_member_count_delete
AFTER DELETE ON fleet_members
BEGIN
    UPDATE fleets 
    SET current_devices = current_devices - 1
    WHERE id = OLD.fleet_id;
END;

-- Create view for active scans
CREATE VIEW active_scans AS
SELECT s.*, u.username
FROM scan_data s
JOIN users u ON s.user_id = u.id
WHERE datetime(s.created_at, '+' || s.retention_days || ' days') > CURRENT_TIMESTAMP;

-- Create view for fleet status
CREATE VIEW fleet_status AS
SELECT 
    f.id,
    f.status,
    f.current_devices,
    f.max_devices,
    u.username as host_name,
    COUNT(fm.user_id) as active_members,
    f.created_at,
    f.updated_at
FROM fleets f
JOIN users u ON f.host_id = u.id
LEFT JOIN fleet_members fm ON f.id = fm.fleet_id
GROUP BY f.id;

COMMIT;

-- Create indexes after commit for better performance
CREATE INDEX IF NOT EXISTS idx_scan_data_composite 
ON scan_data(user_id, created_at, retention_days);

CREATE INDEX IF NOT EXISTS idx_fleet_members_composite 
ON fleet_members(fleet_id, user_id, role);

-- Pragma optimizations for performance
PRAGMA optimize;
PRAGMA analysis_limit=1000;
PRAGMA automatic_index=ON;