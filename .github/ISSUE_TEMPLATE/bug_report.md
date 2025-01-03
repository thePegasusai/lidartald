---
name: Bug Report
about: Create a detailed bug report to help improve TALD UNIA
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
**Summary of the issue:**
<!-- Provide a clear and concise description of the bug -->

**Expected behavior:**
<!-- Describe what you expected to happen -->

**Actual behavior:**
<!-- Describe what actually happened -->

**Impact Severity:**
- [ ] Critical (System crash, data loss, security vulnerability, LiDAR malfunction)
- [ ] High (Major feature broken, significant performance degradation, fleet synchronization failure)
- [ ] Medium (Feature partially broken, moderate user impact, performance below targets)
- [ ] Low (Minor issue, minimal user impact, cosmetic issues)

## Environment Details
**TALD UNIA Device Information:**
- Device ID: 
- Firmware Version: 
- LiDAR Resolution Setting: <!-- Target: 0.01cm -->
- LiDAR Range Setting: <!-- Target: 5m -->
- Fleet Size (if applicable): <!-- Max 32 devices -->
- Game Session Type (if applicable): 
- Operating Environment:
  - [ ] Indoor
  - [ ] Outdoor

## Reproduction Steps
**Preconditions:**
<!-- List any necessary setup or conditions required -->

**Steps to reproduce:**
1. 
2. 
3. 

**Frequency of occurrence:**
- [ ] Always
- [ ] Sometimes
- [ ] Rarely
- [ ] Once

**Fleet configuration (if applicable):**
<!-- Describe the fleet setup when the bug occurred -->

## Performance Impact
**Metrics Affected:**
- LiDAR scan rate: <!-- Target: 30Hz -->
- Point cloud processing time: <!-- Target: <50ms -->
- Network latency: <!-- Target: <50ms -->
- UI responsiveness: <!-- Target: 60 FPS -->
- Memory usage: <!-- Specify component and usage -->
- Battery impact: <!-- % drain rate -->
- Fleet sync delay: <!-- If applicable -->

## Error Information
**Technical Details:**
- Error Code: <!-- Reference range: 1000-5999 -->
- Component: <!-- LiDAR/Fleet/Game/Social/UI -->

**Stack Trace:**
```
<!-- Insert stack trace if available -->
```

**Log Snippets:**
```
<!-- Insert relevant log entries -->
```

**Fleet Network Logs (if applicable):**
```
<!-- Insert fleet-specific logs -->
```

## Additional Context
**Screenshots:**
<!-- Attach relevant screenshots -->

**Video Recordings:**
<!-- Link to video recordings if available -->

**Point Cloud Data Samples:**
<!-- Attach or link to relevant point cloud data -->

**Network Traces:**
<!-- Attach relevant network capture data -->

**Environment Scan Data:**
<!-- Attach relevant environment scan data -->

**Fleet Topology Diagram (if applicable):**
<!-- Insert or attach fleet topology visualization -->

---
<!-- 
Auto-assignment based on component:
LiDAR issues (1000-1999): @lidar-core-team
Point Cloud Processing (2000-2999): @lidar-core-team
Network Communication (3000-3999): @fleet-manager-team
Fleet Coordination (4000-4999): @fleet-manager-team
Game Integration (5000-5999): @game-engine-team
-->