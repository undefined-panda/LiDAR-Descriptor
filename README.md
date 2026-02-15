# LiDAR-Descriptor
Place recognition pipeline using LiDAR data from a publicly available dataset 

The goal is to design a descriptor for LiDAR scans, implement a matching algorithm, and evaluate the system using standard performance metrics.

- descriptor: https://ieeexplore.ieee.org/document/7759060
- LiDAR scans: https://robots.engin.umich.edu/nclt/
- similarity measure as matching algorithm: https://dl.acm.org/doi/10.1109/IROS.2015.7353454




# LiDAR-Based Place Recognition using M2DP

## Overview
This project is part of exercise 3 of the Autonomous Mobile Robots Lecture (Winter 25/26) at TU Darmstadt. It implements a place recognition pipeline based on LiDAR data from the publicly available [**NCLT Dataset**](https://robots.engin.umich.edu/nclt/).

The goal is to:
*   Design and implement a LiDAR scan descriptor.
*   Implement a matching and retrieval algorithm.
*   Detect loop closures.
*   Evaluate performance using standard metrics (Precision-Recall, ROC, F1, Recall).

The descriptor used in this project is **M2DP (Multi-view 2D Projection Descriptor)**.

## Dataset
We use LiDAR data from the first three days of the **NCLT dataset**.

Each day consists of:
*   A sequence of LiDAR scans (`.bin` files).
*   Corresponding timestamps (extracted from filenames).
*   Ground truth pose data (CSV file) providing timestamped robot positions $(x, y, z)$.

## Methodology

### 1. M2DP Preprocessing
For each LiDAR scan:
1.  **Extract 3D points** and remove invalid data.
2.  **Centroid Shift:** Shift point cloud to its centroid.
3.  **PCA Alignment:** Apply Principal Component Analysis to align the x-axis with the first and the y-axis with the second principal component.
This ensures a consistent local coordinate frame.

### 2. M2DP Descriptor
For each scan:
*   **Projections:** Generate $p \times q$ projection planes defined by azimuth $\theta$ and elevation $\phi$.
*   **2D Histograms:** Project points onto each plane, convert to polar coordinates, and build a histogram with $l$ radial and $t$ angular bins.
*   **SVD:** Stack histograms into a matrix $A \in \mathbb{R}^{(l \cdot t) \times (p \cdot q)}$. Apply Singular Value Decomposition: $A = U\Sigma V^T$.
*   **Final Descriptor:** Constructed as $d = [U[:,0], U[:,1]]$.
Descriptors are $L_2$-normalized.

### 3. Loop Candidate Retrieval
For each scan $i$, compare its descriptor to all previous descriptors $j$:
*   **Time Separation:** $t_i - t_j \geq \Delta t$
*   **Matching:** Compute Euclidean distance. A loop candidate is defined if $\|d_i - d_j\| < \tau$.

### 4. Ground Truth Loop Definition
Defined independently of descriptors:
*   **Spatial Constraint:** $\|p_i - p_j\| \leq r$
*   **Time Constraint:** $t_i - t_j \geq \Delta t$
If a scan $j$ fulfills both, scan $i$ is marked as a Ground Truth (GT) loop.

## Evaluation
We evaluate the system using:
1.  **Precision-Recall Curve**
2.  **Average Precision (AP)**
3.  **ROC Curve & AUC**
4.  **F1 Score:** 

## Visualization
The trajectory is visualized in 2D:
*   **Grey line:** Robot trajectory.
*   **Red lines:** Predicted loop closures.
*   **Green lines:** Ground truth loop closures.

## Key Parameters

| Parameter | Description | Values |
| :--- | :--- | :--- |
| **l** | Radial bins | 8 |
| **t** | Angular bins | 16 |
| **p, q** | Projection angles | 4, 4 |
| **τ** | Descriptor threshold | 0.025, 0.03, 0.035 |
| **Δt** | Min. time separation | 30 s |
| **r** | Spatial loop radius | 1, 3, 5 m |

## Conclusion
This project demonstrates a complete LiDAR-based place recognition pipeline from geometric descriptor design to quantitative validation. It can be extended with **Geometric Verification (ICP)** or **KD-tree acceleration** for larger datasets.
