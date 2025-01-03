use std::time::Duration;
use tokio_test::{assert_ok, assert_err}; // tokio-test = "0.4"
use mockall::predicate::*; // mockall = "0.11"
use mockall::mock;

use crate::discovery::{DeviceDiscovery, DeviceInfo, DiscoveryConfig, PeerInfo, ProximityData};
use crate::error::FleetError;

// Test constants
const TEST_DEVICE_ID: &str = "test_device_001";
const TEST_TIMEOUT_MS: u64 = 1000;
const MAX_RETRY_ATTEMPTS: u8 = 3;
const DISCOVERY_INTERVAL_MS: u64 = 100;

// Mock peer device for testing
mock! {
    PeerDevice {
        fn device_id(&self) -> String;
        fn get_proximity_data(&self) -> ProximityData;
        fn establish_connection(&self) -> Result<(), FleetError>;
    }
}

/// Test fixture for discovery tests
struct DiscoveryTests {
    discovery_service: DeviceDiscovery,
    mock_peers: Vec<MockPeerDevice>,
}

impl DiscoveryTests {
    async fn new() -> Self {
        let device_info = DeviceInfo {
            device_id: TEST_DEVICE_ID.to_string(),
            hardware_version: "1.0".to_string(),
            firmware_version: "1.0".to_string(),
            capabilities: vec!["lidar".to_string()],
            lidar_resolution: 0.01,
        };

        let config = DiscoveryConfig {
            broadcast_interval: Duration::from_millis(DISCOVERY_INTERVAL_MS),
            timeout: Duration::from_millis(TEST_TIMEOUT_MS),
            ice_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            proximity_threshold: 5.0,
        };

        let discovery_service = DeviceDiscovery::new(device_info, config)
            .expect("Failed to create discovery service");

        Self {
            discovery_service,
            mock_peers: Vec::new(),
        }
    }

    fn setup_mock_peer(&mut self, device_id: &str, distance: f32) -> MockPeerDevice {
        let mut mock = MockPeerDevice::new();
        
        mock.expect_device_id()
            .returning(move || device_id.to_string());
        
        mock.expect_get_proximity_data()
            .returning(move || ProximityData {
                distance_meters: distance,
                signal_strength: -50,
                spatial_confidence: 0.95,
            });
        
        mock.expect_establish_connection()
            .returning(|| Ok(()));
        
        self.mock_peers.push(mock.clone());
        mock
    }

    #[tokio::test]
    async fn test_successful_discovery() -> Result<(), FleetError> {
        let mut test_suite = DiscoveryTests::new().await;
        let mock_peer = test_suite.setup_mock_peer("peer_001", 3.0);

        // Start discovery process
        assert_ok!(test_suite.discovery_service.start_discovery().await);

        // Verify discovery broadcast
        tokio::time::sleep(Duration::from_millis(DISCOVERY_INTERVAL_MS)).await;
        
        // Simulate peer response
        let peer_info = PeerInfo {
            device_id: mock_peer.device_id(),
            connection_params: Default::default(),
            proximity_data: mock_peer.get_proximity_data(),
            last_seen: tokio::time::Instant::now(),
            retry_count: 0,
        };

        assert_ok!(test_suite.discovery_service.handle_discovery_response(peer_info).await);
        
        // Stop discovery
        assert_ok!(test_suite.discovery_service.stop_discovery().await);
        Ok(())
    }

    #[tokio::test]
    async fn test_discovery_timeout() -> Result<(), FleetError> {
        let mut test_suite = DiscoveryTests::new().await;
        let mock_peer = test_suite.setup_mock_peer("peer_002", 3.0);

        // Configure mock to simulate timeout
        mock_peer.expect_establish_connection()
            .times(MAX_RETRY_ATTEMPTS as usize)
            .returning(|| Err(FleetError::DiscoveryError {
                code: 3001,
                message: "Connection timeout".to_string(),
                source: None,
            }));

        // Start discovery
        assert_ok!(test_suite.discovery_service.start_discovery().await);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(TEST_TIMEOUT_MS)).await;

        // Verify error handling
        let result = test_suite.discovery_service.handle_discovery_response(PeerInfo {
            device_id: mock_peer.device_id(),
            connection_params: Default::default(),
            proximity_data: mock_peer.get_proximity_data(),
            last_seen: tokio::time::Instant::now(),
            retry_count: MAX_RETRY_ATTEMPTS,
        }).await;

        assert!(matches!(result, Err(FleetError::DiscoveryError { .. })));
        
        // Cleanup
        assert_ok!(test_suite.discovery_service.stop_discovery().await);
        Ok(())
    }

    #[tokio::test]
    async fn test_max_devices() -> Result<(), FleetError> {
        let mut test_suite = DiscoveryTests::new().await;
        let mut peers = Vec::new();

        // Create 32 mock peers (maximum limit)
        for i in 0..32 {
            let peer = test_suite.setup_mock_peer(&format!("peer_{:03}", i), 3.0);
            peers.push(peer);
        }

        // Start discovery
        assert_ok!(test_suite.discovery_service.start_discovery().await);

        // Add all peers
        for peer in &peers {
            let peer_info = PeerInfo {
                device_id: peer.device_id(),
                connection_params: Default::default(),
                proximity_data: peer.get_proximity_data(),
                last_seen: tokio::time::Instant::now(),
                retry_count: 0,
            };
            assert_ok!(test_suite.discovery_service.handle_discovery_response(peer_info).await);
        }

        // Try to add 33rd peer (should fail)
        let excess_peer = test_suite.setup_mock_peer("peer_excess", 3.0);
        let excess_peer_info = PeerInfo {
            device_id: excess_peer.device_id(),
            connection_params: Default::default(),
            proximity_data: excess_peer.get_proximity_data(),
            last_seen: tokio::time::Instant::now(),
            retry_count: 0,
        };

        let result = test_suite.discovery_service.handle_discovery_response(excess_peer_info).await;
        assert!(matches!(result, Err(FleetError::DiscoveryError { code: 3006, .. })));

        // Cleanup
        assert_ok!(test_suite.discovery_service.stop_discovery().await);
        Ok(())
    }

    #[tokio::test]
    async fn test_out_of_range_peer() -> Result<(), FleetError> {
        let mut test_suite = DiscoveryTests::new().await;
        let mock_peer = test_suite.setup_mock_peer("peer_003", 6.0); // Beyond 5m threshold

        assert_ok!(test_suite.discovery_service.start_discovery().await);

        let peer_info = PeerInfo {
            device_id: mock_peer.device_id(),
            connection_params: Default::default(),
            proximity_data: mock_peer.get_proximity_data(),
            last_seen: tokio::time::Instant::now(),
            retry_count: 0,
        };

        // Should be ignored due to distance
        assert_ok!(test_suite.discovery_service.handle_discovery_response(peer_info).await);

        assert_ok!(test_suite.discovery_service.stop_discovery().await);
        Ok(())
    }

    #[tokio::test]
    async fn test_discovery_restart() -> Result<(), FleetError> {
        let mut test_suite = DiscoveryTests::new().await;
        
        // First discovery cycle
        assert_ok!(test_suite.discovery_service.start_discovery().await);
        assert_ok!(test_suite.discovery_service.stop_discovery().await);

        // Immediate restart should work
        assert_ok!(test_suite.discovery_service.start_discovery().await);
        assert_ok!(test_suite.discovery_service.stop_discovery().await);
        
        Ok(())
    }
}

// Helper function to create test device configuration
fn setup_test_device() -> DeviceInfo {
    DeviceInfo {
        device_id: TEST_DEVICE_ID.to_string(),
        hardware_version: "1.0".to_string(),
        firmware_version: "1.0".to_string(),
        capabilities: vec!["lidar".to_string()],
        lidar_resolution: 0.01,
    }
}