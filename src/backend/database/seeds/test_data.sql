-- TALD UNIA Platform Test Data Seed
-- SQLite Version: 3.42.0
-- Isolation Level: SERIALIZABLE

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- Cleanup existing test data with proper order
DELETE FROM game_sessions;
DELETE FROM fleet_members;
DELETE FROM fleets;
DELETE FROM features;
DELETE FROM environments;
DELETE FROM scan_data;
DELETE FROM users;

-- Insert test users with various roles
INSERT INTO users (id, email, username, password, role, created_at, updated_at) VALUES
('usr_admin_01', 'admin@tald.dev', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiLXCJyqJNge', 'ADMIN', datetime('now'), datetime('now')),
('usr_dev_01', 'dev@tald.dev', 'developer', '$2b$12$9tQFp.dY4J/dYz.7FJIrZO.5Tg0sFP.pqT8mA.kh.7zPp/k6HhvX2', 'DEVELOPER', datetime('now'), datetime('now')),
('usr_premium_01', 'premium@tald.dev', 'premium_player', '$2b$12$kP.m3mRPEqG1RxC1C2Xj.OQz5gQoZrr3q.qJ5tP.4V3qf0HU1XZi6', 'PREMIUM_USER', datetime('now'), datetime('now')),
('usr_basic_01', 'basic@tald.dev', 'basic_player', '$2b$12$tP.8j5CmHrVxY3rNXs5d.OZ5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5', 'BASIC_USER', datetime('now'), datetime('now')),
('usr_guest_01', 'guest@tald.dev', 'guest_player', '$2b$12$qP.8j5CmHrVxY3rNXs5d.OZ5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5Z5', 'GUEST', datetime('now'), datetime('now'));

-- Insert test scan data with varying parameters
INSERT INTO scan_data (id, user_id, point_cloud, metadata, resolution, created_at, expires_at) VALUES
('scan_01', 'usr_premium_01', X'0102030405', json('{"device_id":"TALD-TEST-001","scan_mode":"HIGH_RESOLUTION","range":5.0,"lighting":"indoor","confidence":0.98,"point_count":125000}'), 0.01, datetime('now'), datetime('now', '+7 days')),
('scan_02', 'usr_premium_01', X'0506070809', json('{"device_id":"TALD-TEST-002","scan_mode":"FAST_SCAN","range":3.0,"lighting":"outdoor","confidence":0.85,"point_count":50000}'), 0.05, datetime('now'), datetime('now', '+7 days')),
('scan_03', 'usr_basic_01', X'0A0B0C0D0E', json('{"device_id":"TALD-TEST-003","scan_mode":"STANDARD","range":4.0,"lighting":"mixed","confidence":0.92,"point_count":75000}'), 0.03, datetime('now'), datetime('now', '+7 days'));

-- Insert test environments
INSERT INTO environments (id, scan_id, boundaries, obstacles, created_at, expires_at) VALUES
('env_01', 'scan_01', json('{"width":10.5,"height":8.0,"depth":3.2}'), json('{"count":12,"types":["wall","furniture","dynamic"]}'), datetime('now'), datetime('now', '+30 days')),
('env_02', 'scan_02', json('{"width":15.0,"height":10.0,"depth":4.0}'), json('{"count":8,"types":["wall","static"]}'), datetime('now'), datetime('now', '+30 days'));

-- Insert test features
INSERT INTO features (id, scan_id, environment_id, type, coordinates, confidence, created_at) VALUES
('feat_01', 'scan_01', 'env_01', 'WALL', json('{"x":0.0,"y":0.0,"z":0.0,"width":5.0,"height":3.0}'), 0.98, datetime('now')),
('feat_02', 'scan_01', 'env_01', 'FURNITURE', json('{"x":2.5,"y":1.5,"z":0.0,"width":1.2,"height":0.8}'), 0.95, datetime('now')),
('feat_03', 'scan_02', 'env_02', 'WALL', json('{"x":0.0,"y":0.0,"z":0.0,"width":7.0,"height":3.5}'), 0.92, datetime('now')),
('feat_04', 'scan_02', 'env_02', 'STATIC', json('{"x":3.0,"y":2.0,"z":0.0,"width":2.0,"height":1.5}'), 0.88, datetime('now'));

-- Insert test fleets
INSERT INTO fleets (id, name, status, created_at, updated_at) VALUES
('fleet_01', 'Test Squad Alpha', 'ACTIVE', datetime('now'), datetime('now')),
('fleet_02', 'Test Team Beta', 'ACTIVE', datetime('now'), datetime('now'));

-- Insert test fleet members
INSERT INTO fleet_members (id, user_id, fleet_id, role, joined_at) VALUES
('fm_01', 'usr_premium_01', 'fleet_01', 'OWNER', datetime('now')),
('fm_02', 'usr_basic_01', 'fleet_01', 'MEMBER', datetime('now')),
('fm_03', 'usr_dev_01', 'fleet_02', 'OWNER', datetime('now')),
('fm_04', 'usr_premium_01', 'fleet_02', 'MEMBER', datetime('now'));

-- Insert test game sessions
INSERT INTO game_sessions (id, fleet_id, environment_id, type, metadata, start_time, end_time) VALUES
('game_01', 'fleet_01', 'env_01', 'BATTLE_ARENA', json('{"max_players":8,"difficulty":"medium","mode":"team_deathmatch"}'), datetime('now', '-1 hour'), datetime('now')),
('game_02', 'fleet_02', 'env_02', 'CAPTURE_POINT', json('{"max_players":4,"difficulty":"hard","mode":"free_for_all"}'), datetime('now'), NULL);

-- Insert game session participants
INSERT INTO game_session_participants (game_session_id, user_id) VALUES
('game_01', 'usr_premium_01'),
('game_01', 'usr_basic_01'),
('game_02', 'usr_dev_01'),
('game_02', 'usr_premium_01');

-- Create indexes for test data
CREATE INDEX IF NOT EXISTS idx_scan_data_resolution ON scan_data(resolution);
CREATE INDEX IF NOT EXISTS idx_features_confidence ON features(confidence);
CREATE INDEX IF NOT EXISTS idx_game_sessions_type ON game_sessions(type);

COMMIT;

-- Error handling
PRAGMA integrity_check;
PRAGMA foreign_key_check;

-- Cleanup trigger for expired data
CREATE TRIGGER IF NOT EXISTS cleanup_expired_data
AFTER INSERT ON scan_data
BEGIN
    DELETE FROM scan_data WHERE expires_at < datetime('now');
    DELETE FROM environments WHERE expires_at < datetime('now');
END;