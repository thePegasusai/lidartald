use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use petgraph::{
    algo::{dijkstra, min_spanning_tree},
    graph::{Graph, NodeIndex},
    Directed,
};
use tokio::{sync::RwLock, time};
use tracing::{debug, error, info, instrument, warn};
use webrtc::{
    api::APIBuilder,
    data_channel::RTCDataChannel,
    peer_connection::{
        configuration::RTCConfiguration,
        RTCPeerConnection,
        RTCPeerConnectionState,
    },
};

use crate::discovery::{DeviceDiscovery, PeerInfo};
use crate::error::{FleetError, FleetResult};

// Constants from globals
const MAX_PEERS: usize = 32;
const MESH_OPTIMIZATION_INTERVAL_MS: u64 = 1000;
const CONNECTION_TIMEOUT_MS: u64 = 5000;
const MAX_LATENCY_MS: u64 = 50;

/// Represents a device identifier in the mesh network
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DeviceId(String);

/// Represents a peer connection with associated metrics
#[derive(Debug)]
struct PeerConnection {
    connection: Arc<RTCPeerConnection>,
    data_channel: Arc<RTCDataChannel>,
    latency: Arc<RwLock<u64>>,
    last_seen: Arc<RwLock<Instant>>,
    state: Arc<RwLock<RTCPeerConnectionState>>,
}

/// Monitors connection latency and quality
#[derive(Debug)]
struct LatencyMonitor {
    measurements: HashMap<DeviceId, Vec<u64>>,
    window_size: usize,
}

/// Manages connection pooling and reuse
#[derive(Debug)]
struct ConnectionPool {
    available: Vec<RTCPeerConnection>,
    capacity: usize,
}

/// Advanced P2P mesh network manager
#[derive(Debug)]
pub struct MeshNetwork {
    local_device_id: DeviceId,
    peer_connections: DashMap<DeviceId, PeerConnection>,
    topology: Arc<RwLock<Graph<DeviceId, f64, Directed>>>,
    discovery_service: Arc<DeviceDiscovery>,
    latency_monitor: Arc<Mutex<LatencyMonitor>>,
    connection_pool: ConnectionPool,
}

impl MeshNetwork {
    /// Creates a new mesh network manager instance
    #[instrument(skip(discovery))]
    pub async fn new(device_id: String, discovery: DeviceDiscovery) -> FleetResult<Self> {
        let network = Self {
            local_device_id: DeviceId(device_id),
            peer_connections: DashMap::new(),
            topology: Arc::new(RwLock::new(Graph::new())),
            discovery_service: Arc::new(discovery),
            latency_monitor: Arc::new(Mutex::new(LatencyMonitor::new(100))),
            connection_pool: ConnectionPool::new(MAX_PEERS),
        };

        network.start_optimization_task();
        Ok(network)
    }

    /// Adds a new peer to the mesh network
    #[instrument(skip(self))]
    pub async fn add_peer(&self, peer_info: PeerInfo) -> FleetResult<()> {
        if self.peer_connections.len() >= MAX_PEERS {
            return Err(FleetError::MeshError {
                code: 3400,
                message: format!("Maximum peer limit ({}) reached", MAX_PEERS),
                source: None,
            });
        }

        let peer_id = DeviceId(peer_info.device_id.clone());
        let connection = self.establish_connection(&peer_info).await?;
        
        // Update topology
        let mut topology = self.topology.write().await;
        let node = topology.add_node(peer_id.clone());
        let local_node = topology.node_indices()
            .find(|i| topology[*i] == self.local_device_id)
            .unwrap_or_else(|| topology.add_node(self.local_device_id.clone()));
        topology.add_edge(local_node, node, 1.0);

        self.peer_connections.insert(peer_id, connection);
        self.monitor_connection(&peer_id);
        
        info!("Added peer {} to mesh network", peer_info.device_id);
        Ok(())
    }

    /// Removes a peer from the mesh network
    #[instrument(skip(self))]
    pub async fn remove_peer(&self, peer_id: &DeviceId) -> FleetResult<()> {
        if let Some((_, connection)) = self.peer_connections.remove(peer_id) {
            connection.connection.close().await?;
            
            let mut topology = self.topology.write().await;
            let node = topology.node_indices()
                .find(|i| topology[*i] == *peer_id)
                .ok_or(FleetError::MeshError {
                    code: 3401,
                    message: "Peer not found in topology".to_string(),
                    source: None,
                })?;
            topology.remove_node(node);
        }
        
        Ok(())
    }

    /// Returns the optimal path between two peers
    #[instrument(skip(self))]
    pub async fn get_optimal_path(&self, from: &DeviceId, to: &DeviceId) -> FleetResult<Vec<DeviceId>> {
        let topology = self.topology.read().await;
        let start = topology.node_indices()
            .find(|i| topology[*i] == *from)
            .ok_or(FleetError::MeshError {
                code: 3402,
                message: "Source peer not found".to_string(),
                source: None,
            })?;
        
        let end = topology.node_indices()
            .find(|i| topology[*i] == *to)
            .ok_or(FleetError::MeshError {
                code: 3403,
                message: "Destination peer not found".to_string(),
                source: None,
            })?;

        let paths = dijkstra(&topology, start, Some(end), |e| *e.weight());
        let mut path = Vec::new();
        let mut current = end;
        
        while let Some(prev) = paths.get(&current) {
            path.push(topology[current].clone());
            if current == start {
                break;
            }
            current = *prev;
        }
        
        path.reverse();
        Ok(path)
    }

    /// Optimizes the mesh network topology
    #[instrument(skip(self))]
    pub async fn optimize_topology(&self) -> FleetResult<()> {
        let mut topology = self.topology.write().await;
        let latencies = self.collect_latency_metrics().await?;
        
        // Update edge weights based on latency
        for (peer_id, latency) in latencies {
            if let Some(node) = topology.node_indices()
                .find(|i| topology[*i] == peer_id) {
                let local_node = topology.node_indices()
                    .find(|i| topology[*i] == self.local_device_id)
                    .unwrap();
                let edge = topology.find_edge(local_node, node)
                    .unwrap();
                topology[edge] = latency as f64;
            }
        }

        // Generate minimum spanning tree
        let mst = min_spanning_tree(&topology);
        let mut new_topology = Graph::new();
        
        // Rebuild topology with MST edges
        for edge in mst {
            let (source, target) = topology.edge_endpoints(edge).unwrap();
            let source_id = topology[source].clone();
            let target_id = topology[target].clone();
            let weight = topology[edge];
            
            let source_node = new_topology.add_node(source_id);
            let target_node = new_topology.add_node(target_id);
            new_topology.add_edge(source_node, target_node, weight);
        }

        *topology = new_topology;
        Ok(())
    }

    // Private helper methods
    async fn establish_connection(&self, peer_info: &PeerInfo) -> FleetResult<PeerConnection> {
        let connection = match self.connection_pool.get_connection() {
            Some(conn) => conn,
            None => {
                let config = peer_info.connection_params.clone();
                let api = APIBuilder::new().build();
                api.new_peer_connection(config).await?
            }
        };

        let data_channel = connection.create_data_channel("mesh", None).await?;
        
        Ok(PeerConnection {
            connection: Arc::new(connection),
            data_channel: Arc::new(data_channel),
            latency: Arc::new(RwLock::new(0)),
            last_seen: Arc::new(RwLock::new(Instant::now())),
            state: Arc::new(RwLock::new(RTCPeerConnectionState::New)),
        })
    }

    fn start_optimization_task(&self) {
        let topology = self.topology.clone();
        let latency_monitor = self.latency_monitor.clone();
        
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_millis(MESH_OPTIMIZATION_INTERVAL_MS));
            
            loop {
                interval.tick().await;
                if let Err(e) = Self::run_optimization(&topology, &latency_monitor).await {
                    error!("Topology optimization failed: {:?}", e);
                }
            }
        });
    }

    fn monitor_connection(&self, peer_id: &DeviceId) {
        let peer_id = peer_id.clone();
        let connections = self.peer_connections.clone();
        let latency_monitor = self.latency_monitor.clone();
        
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                if let Some(connection) = connections.get(&peer_id) {
                    let start = Instant::now();
                    if connection.data_channel.send_text("ping").await.is_ok() {
                        let latency = start.elapsed().as_millis() as u64;
                        if let Ok(mut monitor) = latency_monitor.lock() {
                            monitor.add_measurement(&peer_id, latency);
                        }
                    }
                } else {
                    break;
                }
            }
        });
    }

    async fn collect_latency_metrics(&self) -> FleetResult<HashMap<DeviceId, u64>> {
        let mut metrics = HashMap::new();
        let monitor = self.latency_monitor.lock().map_err(|_| FleetError::MeshError {
            code: 3404,
            message: "Failed to acquire latency monitor lock".to_string(),
            source: None,
        })?;
        
        for (peer_id, measurements) in &monitor.measurements {
            if !measurements.is_empty() {
                let avg_latency = measurements.iter().sum::<u64>() / measurements.len() as u64;
                metrics.insert(peer_id.clone(), avg_latency);
            }
        }
        
        Ok(metrics)
    }

    async fn run_optimization(
        topology: &RwLock<Graph<DeviceId, f64, Directed>>,
        latency_monitor: &Mutex<LatencyMonitor>,
    ) -> FleetResult<()> {
        let mut topology = topology.write().await;
        let monitor = latency_monitor.lock().map_err(|_| FleetError::MeshError {
            code: 3405,
            message: "Failed to acquire latency monitor lock".to_string(),
            source: None,
        })?;

        // Optimization logic here
        Ok(())
    }
}

// Helper implementations
impl LatencyMonitor {
    fn new(window_size: usize) -> Self {
        Self {
            measurements: HashMap::new(),
            window_size,
        }
    }

    fn add_measurement(&mut self, peer_id: &DeviceId, latency: u64) {
        let measurements = self.measurements
            .entry(peer_id.clone())
            .or_insert_with(Vec::new);
            
        if measurements.len() >= self.window_size {
            measurements.remove(0);
        }
        measurements.push(latency);
    }
}

impl ConnectionPool {
    fn new(capacity: usize) -> Self {
        Self {
            available: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_connection(&self) -> Option<RTCPeerConnection> {
        None // Implement connection pooling logic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Add comprehensive tests here
}