from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

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
    y_true = []
    y_score = []
    for i, j, score in tqdm(loop_candidates, desc="Computing loop candidate positions & Extracting ground truth loop positions"):
        t_i = timestamps[int(i)]
        t_j = timestamps[int(j)]

        p_i = interp_pos(t_i)
        p_j = interp_pos(t_j)

        if not (np.any(np.isnan(p_i)) or np.any(np.isnan(p_j))):
            candidates_loop_pos.append((p_i, p_j))
            time_diff = (t_i.astype(np.int64) - t_j.astype(np.int64)) * 1e-6
            dist = np.linalg.norm(p_i - p_j)
            if (time_diff > time_window) and (dist < r):
                y_true.append(1)
            else:
                y_true.append(0)
            
            y_score.append(score)
    
    return candidates_loop_pos, y_true, y_score

def compute_gt_loops_pos(timestamps, gt, time_window, r):
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

    gt_loops = []
    for i in range(len(timestamps)):
        for j in range(i):
            t_i = timestamps[int(i)]
            t_j = timestamps[int(j)]

            p_i = interp_pos(t_i)
            p_j = interp_pos(t_j)

            if not (np.any(np.isnan(p_i)) or np.any(np.isnan(p_j))):
                # time diff check
                if (np.abs(t_i - t_j) / np.timedelta64(1, 's')) > time_window:
                    # pos diff check
                    if np.linalg.norm(p_i - p_j) < r:
                        gt_loops.append((i, j))
    
    return gt_loops
