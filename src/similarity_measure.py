from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def retrieve_candidates(descriptors, window, tau, topk=1):
    """
    Returns: np.ndarray of shape (m, 3) with rows (i, j, dist)
    """

    tau2 = tau * tau
    # squared norms of each vector
    norms = np.einsum("ij,ij->i", descriptors, descriptors)

    out = []  # list of (i, j, dist)

    for i in tqdm(range(descriptors.shape[0]), desc="Computing loop candidates"):
        end = i - window
        if end < 0:
            continue  # no eligible previous vectors

        # candidates are j in [0, end]
        A = descriptors[:end + 1]  # shape (end+1, d)

        # squared euclidean distances: ||A - x||^2 = ||A||^2 + ||x||^2 - 2 A·x
        d2 = norms[:end + 1] + norms[i] - 2.0 * (A @ descriptors[i])
        # numerical safety
        d2 = np.maximum(d2, 0.0)

        mask = d2 < tau2
        if not np.any(mask):
            continue

        j_all = np.flatnonzero(mask)   # original j indices that pass tau
        d2_all = d2[mask]

        kk = min(topk, d2_all.size)
        # pick kk smallest (unordered), then sort them
        sel = np.argpartition(d2_all, kk - 1)[:kk]
        sel = sel[np.argsort(d2_all[sel])]

        js = j_all[sel]
        ds = np.sqrt(d2_all[sel])  # convert to true distances (optional)

        # store (i, j, dist)
        out.extend((i, int(j), float(dist)) for j, dist in zip(js, ds))

    out = np.array(out, dtype=float)
    print(f"Candidates found: {len(out)}")

    return out # columns: i, j, dist

def retrieve_loop_pos(loop_candidates, timestamps, gt, time_window, r):
    gt_sorted = gt.sort_values("timestamp")
    ts_gt = gt_sorted["timestamp"].to_numpy()
    pos_gt = gt_sorted[["x", "y"]].to_numpy()

    interp_pos = interp1d(
        ts_gt,
        pos_gt,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    candidates_loop_pos = []
    gt_loop_pos = []
    for i, j, _ in tqdm(loop_candidates, desc="Computing loop candidate positions & Extracting ground truth loop positions"):
        t_i = timestamps[int(i)]
        t_j = timestamps[int(j)]

        p_i = interp_pos(t_i)
        p_j = interp_pos(t_j)

        if not (np.any(np.isnan(p_i)) or np.any(np.isnan(p_j))):
            candidates_loop_pos.append((p_i, p_j))
        
        time_diff = np.abs(t_i.astype(float) - t_j.astype(float)) * 1e-6
        if time_diff > time_window:
            if np.linalg.norm(p_i - p_j) < r:
                gt_loop_pos.append((p_i, p_j))
    
    return candidates_loop_pos, gt_loop_pos

def visualize_trajectory(trajectory, candidates_loop_pos, gt_loop_pos, day):
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # left with lines, right with dots
    for i, (p_i, p_j) in enumerate(candidates_loop_pos):
        if i == 0:
            axes[0].plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], color="red", markersize=15, alpha=0.3, zorder=2, label="Descriptor loops")
            axes[1].scatter(p_i[0], p_i[1], color="red", s=10, alpha=0.3, zorder=2, label="Descriptor loops")
        else:
            axes[0].plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], markersize=15, color="red", alpha=0.3, zorder=2)
            axes[1].scatter(p_i[0], p_i[1], color="red", s=10, alpha=0.3, zorder=2)

    for i, (p_i, p_j) in enumerate(gt_loop_pos):
        if i == 0:
            axes[0].plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], markersize=15, color="green", alpha=0.3, zorder=3, label="GT loops")
            axes[1].scatter(p_i[0], p_i[1], color="green", s=8, alpha=0.3, zorder=3, label="GT loops")
        else:
            axes[0].plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], markersize=15, color="green", alpha=0.3, zorder=3)
            axes[1].scatter(p_i[0], p_i[1], color="green", s=8, alpha=0.3, zorder=3)    

    for i in range(len(axes)):
        axes[i].scatter(x, y, color='lightgray', s=15, label='Robot Trajectory', zorder=1)
        axes[i].set_title(f"Trajectory Visualization of {day}")
        axes[i].set_xlabel("x (m)")
        axes[i].set_ylabel("y (m)")
        axes[i].axis('equal')
        axes[i].legend()
        axes[i].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def get_gt_trajectory(gt, timestamps):
    gt_sorted = gt.sort_values("timestamp")

    ts_gt = gt_sorted["timestamp"].to_numpy()
    x_gt = gt_sorted["x"].to_numpy()
    y_gt = gt_sorted["y"].to_numpy()

    timestamps = np.asarray(timestamps).reshape(-1)
    mask = (timestamps >= ts_gt[0]) & (timestamps <= ts_gt[-1])
    timestamps = timestamps[mask]

    interpld_x = interp1d(ts_gt, x_gt, kind="linear")
    interpld_y = interp1d(ts_gt, y_gt, kind="linear")

    x = interpld_x(timestamps)
    y = interpld_y(timestamps)

    return np.column_stack((x, y)), mask
