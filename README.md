# Plane Detection & Box Size Estimation

This project implements and evaluates three plane detection algorithms — **RANSAC**, **MLESAC**, and **Preemptive RANSAC** — on 3D point cloud data from a Kinect sensor to estimate the dimensions of a box placed on the floor.

---

## Features

### Custom Implementations
- **RANSAC**: Standard random sample consensus for plane fitting.
- **MLESAC**: Maximum Likelihood Estimation Sample Consensus, which minimizes a likelihood-based cost instead of only counting inliers.
- **Preemptive RANSAC**: Nistér's method for efficient hypothesis evaluation under time constraints.

### 3D Plane Detection
- Detects the **floor plane** and then the **top surface of a box** from Kinect `.mat` data.
- Applies morphological operations to refine binary masks of detected planes.

### Box Size Estimation
- Calculates:
  - **Height**: Distance between floor and box top.
  - **Length** & **Width**: Derived from bounding box of top surface points.

### Performance Evaluation
- Compares algorithms over different thresholds and parameters:
  - `M` and `B` for Preemptive RANSAC.
- Records computation times and dimension accuracy.
- Optionally exports results to CSV.

### Visualization
- Displays:
  - Amplitude and distance images.
  - Floor masks and detected box components.
- Plots performance metrics for algorithm comparison.

---

## Usage

The script includes an example call:

```python
evaluate_preemptive_vs_others(
    file_path='./data/example1kinect.mat',
    amplitude_index='amplitudes1',
    distance_index='distances1',
    cloud_index='cloud1',
    thresholds=[0.005, 0.01, 0.015],
    preemptive_configs=[(100, 10), (150, 20), (200, 30)],
    export_csv=True
)
