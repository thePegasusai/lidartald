use std::time::Duration;
use tokio_test; // version = "0.4"
use mockall::predicate::*; // version = "0.11"
use mockall::mock;
use webrtc::api::APIBuilder; // version = "0.7"
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::RTCPeerConnectionState;

use crate::mesh::{DeviceId, MeshNetwork};
use crate::error::{FleetError, FleetResult};
use crate::discovery::{DeviceDiscovery, DeviceInfo, PeerInfo, ProximityData};

// Constants from globals
const TEST_TIMEOUT_MS: u64 = 10000;
const MAX_TEST_PEERS: usize = 32;
const MIN_LATENCY_MS: u64 = 5;
const MAX_LATENCY_MS: u64 = 50;
const TOPOLOGY_UPDATE_INTERVAL_MS: u64 = 1000;

// Mock peer connection for testing
mock! {
    PeerConnection {
        fn device_id(&self) -> DeviceId;
        fn get_latency(&self) -> u64;
        fn get_state(&self) -> RTCPeerConnectionState;
        fn simulate_network_condition(&mut self, latency_ms: u64) -> FleetResult<()>;
    }
}

// Test network configuration
#[derive(Debug, Clone)]
struct NetworkTestConfig {
    base_latency: u64,
    jitter_ms: u64,
    packet_loss_rate: f64,
    topology_update_interval: Duration,
}

impl Default for NetworkTestConfig {
    fn default() -> Self {
        Self {
            base_latency: 10,
            jitter_ms: 5,
            packet_loss_rate: 0.01,
            topology_update_interval: Duration::from_millis(TOPOLOGY_UPDATE_INTERVAL_MS),
        }
    }
}

// Test helper to setup mesh network with mock devices
async fn setup_test_mesh(peer_count: usize, config: NetworkTestConfig) -> FleetResult<MeshNetwork> {
    let device_info = DeviceInfo {
        device_id: "test_device".to_string(),
        hardware_version: "1.0".to_string(),
        firmware_version: "1.0".to_string(),
        capabilities: vec!["mesh".to_string()],
        lidar_resolution: 0.01,
    };

    let discovery = DeviceDiscovery::new(
        device_info,
        RTCConfiguration::default(),
    )?;

    let mesh = MeshNetwork::new("test_device".to_string(), discovery).await?;

    // Add mock peers
    for i in 0..peer_count {
        let peer_info = PeerInfo {
            device_id: format!("peer_{}", i),
            connection_params: RTCConfiguration::default(),
            proximity_data: ProximityData {
                distance_meters: 2.0,
                signal_strength: -50,
                spatial_confidence: 0.95,
            },
            last_seen: tokio::time::Instant::now(),
            retry_count: 0,
        };
        mesh.add_peer(peer_info).await?;
    }

    Ok(mesh)
}

#[tokio::test]
async fn test_mesh_creation() {
    let config = NetworkTestConfig::default();
    let mesh = setup_test_mesh(0, config).await;
    assert!(mesh.is_ok(), "Failed to create mesh network");

    let mesh = mesh.unwrap();
    let metrics = mesh.get_network_metrics().await;
    assert!(metrics.is_ok(), "Failed to get network metrics");
    assert_eq!(metrics.unwrap().peer_count, 0);
}

#[tokio::test]
async fn test_peer_addition() {
    let config = NetworkTestConfig::default();
    let mesh = setup_test_mesh(0, config).await.unwrap();

    // Test adding peers up to maximum
    for i in 0..MAX_TEST_PEERS {
        let peer_info = PeerInfo {
            device_id: format!("peer_{}", i),
            connection_params: RTCConfiguration::default(),
            proximity_data: ProximityData {
                distance_meters: 2.0,
                signal_strength: -50,
                spatial_confidence: 0.95,
            },
            last_seen: tokio::time::Instant::now(),
            retry_count: 0,
        };

        let result = mesh.add_peer(peer_info).await;
        assert!(result.is_ok(), "Failed to add peer {}", i);

        let metrics = mesh.get_network_metrics().await.unwrap();
        assert_eq!(metrics.peer_count, i + 1);
        assert!(metrics.average_latency <= MAX_LATENCY_MS as f64);
    }

    // Test adding beyond maximum
    let excess_peer = PeerInfo {
        device_id: "excess_peer".to_string(),
        connection_params: RTCConfiguration::default(),
        proximity_data: ProximityData {
            distance_meters: 2.0,
            signal_strength: -50,
            spatial_confidence: 0.95,
        },
        last_seen: tokio::time::Instant::now(),
        retry_count: 0,
    };

    let result = mesh.add_peer(excess_peer).await;
    assert!(matches!(result, Err(FleetError::MeshError { code: 3400, .. })));
}

#[tokio::test]
async fn test_topology_optimization() {
    let config = NetworkTestConfig {
        base_latency: 20,
        jitter_ms: 10,
        packet_loss_rate: 0.02,
        topology_update_interval: Duration::from_millis(100),
    };

    let mesh = setup_test_mesh(5, config).await.unwrap();

    // Let the network stabilize
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Test initial topology
    let metrics = mesh.get_network_metrics().await.unwrap();
    let initial_latency = metrics.average_latency;

    // Simulate network degradation
    let peer_id = DeviceId("peer_0".to_string());
    let mut mock_conn = MockPeerConnection::new();
    mock_conn.expect_simulate_network_condition()
        .with(eq(100))
        .times(1)
        .returning(|_| Ok(()));

    // Trigger optimization
    mesh.optimize_topology().await.unwrap();

    // Verify optimization results
    let metrics = mesh.get_network_metrics().await.unwrap();
    assert!(metrics.average_latency <= initial_latency, 
        "Topology optimization failed to improve latency");
    assert!(metrics.average_latency <= MAX_LATENCY_MS as f64,
        "Network latency exceeds maximum allowed");
}

#[tokio::test]
async fn test_path_calculation() {
    let config = NetworkTestConfig::default();
    let mesh = setup_test_mesh(10, config).await.unwrap();

    let source = DeviceId("test_device".to_string());
    let target = DeviceId("peer_5".to_string());

    let path = mesh.get_optimal_path(&source, &target).await;
    assert!(path.is_ok(), "Failed to calculate optimal path");

    let path = path.unwrap();
    assert!(!path.is_empty(), "Path should not be empty");
    assert_eq!(path[0], source, "Path should start with source");
    assert_eq!(path[path.len() - 1], target, "Path should end with target");
}

#[tokio::test]
async fn test_peer_removal() {
    let config = NetworkTestConfig::default();
    let mesh = setup_test_mesh(5, config).await.unwrap();

    let peer_id = DeviceId("peer_2".to_string());
    let result = mesh.remove_peer(&peer_id).await;
    assert!(result.is_ok(), "Failed to remove peer");

    let metrics = mesh.get_network_metrics().await.unwrap();
    assert_eq!(metrics.peer_count, 4);

    // Verify topology update
    let path = mesh.get_optimal_path(
        &DeviceId("peer_1".to_string()),
        &DeviceId("peer_3".to_string())
    ).await;
    assert!(path.is_ok(), "Failed to calculate path after peer removal");
}

#[tokio::test]
async fn test_network_resilience() {
    let config = NetworkTestConfig {
        base_latency: 15,
        jitter_ms: 8,
        packet_loss_rate: 0.05,
        topology_update_interval: Duration::from_millis(200),
    };

    let mesh = setup_test_mesh(8, config).await.unwrap();

    // Simulate network issues
    for i in 0..3 {
        let peer_id = DeviceId(format!("peer_{}", i));
        let mut mock_conn = MockPeerConnection::new();
        mock_conn.expect_simulate_network_condition()
            .with(eq(200))
            .times(1)
            .returning(|_| Ok(()));
    }

    // Let the network adapt
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify network stability
    let metrics = mesh.get_network_metrics().await.unwrap();
    assert!(metrics.average_latency <= MAX_LATENCY_MS as f64,
        "Network failed to maintain stability under stress");
    assert!(metrics.packet_loss_rate <= 0.1,
        "Excessive packet loss under network stress");
}