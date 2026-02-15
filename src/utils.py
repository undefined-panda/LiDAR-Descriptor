from tqdm import tqdm
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.read_vel_sync import read_vel_sync
from src.m2dp_descriptor import compute_m2dp_signature_vectors
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from scipy.interpolate import interp1d

def remove_nan(df):
    for column in df.columns:
        df[column] = df[column].astype("float64")
    
    df = df.dropna()

    return df

def process_file(file_path, l, t, p, q):
    timestamp = file_path.stem

    pc = read_vel_sync(file_path)
    
    m2dp_descriptors = compute_m2dp_signature_vectors(pc, l, t, p, q)

    return timestamp, m2dp_descriptors

def calculate_m2dp_descriptor(folder, output_dir, l=8, t=16, p=4, q=4, save=True, parallelize=True):
    session_day = folder.parent.name
    bin_files = sorted(folder.glob("*.bin"))
    print(f"Found {len(bin_files)} scans in {folder}.")

    if parallelize:
        print("Parallelizing process with multi-threading.")
        fn = partial(process_file, l=l, t=t, p=p, q=q)
        
        with ProcessPoolExecutor() as ex:
            results = list(tqdm(
                ex.map(fn, bin_files),
                total=len(bin_files),
                desc="Processing files"
            ))
    
        timestamps, descriptors = map(list, zip(*results))
    else:
        # read bin files and calculate M2DP descriptor for each point cloud
        timestamps = []
        descriptors = []
        for file_path in tqdm(bin_files, desc=f"Processing bin files from {session_day}"):
            timestamp = file_path.stem

            pc = read_vel_sync(file_path, True, f"data/velodyne_csv/{session_day}.csv")

            m2dp_descriptors = compute_m2dp_signature_vectors(pc, l, t, p, q)

            timestamps.append(timestamp)
            descriptors.append(m2dp_descriptors)
    
    timestamps_np = np.array(timestamps)
    descriptors_np = np.array(descriptors)

    # save descriptors
    if save:
        os.makedirs(output_dir, exist_ok=True)

        descriptors_output_filename = f"m2dp_{session_day}.npy"
        timestamps_output_filename = f"timestamps_{session_day}.npy"
        np.save(output_dir+"/descriptors/"+descriptors_output_filename, descriptors_np)
        np.save(output_dir+"/timestamps/"+timestamps_output_filename, timestamps_np)

        print(f"Done. Signature vector matrix shape: {descriptors_np.shape}")
        print(f"Descriptors saved in {output_dir+"/descriptors/"+descriptors_output_filename}")
        print(f"Timestamps saved in {output_dir+"/timestamps/"+timestamps_output_filename}")

def run_evaluations(y_true, y_score):
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)
    
    fpr, tpr, _ = roc_curve(y_true, -np.array(y_score))
    roc_auc = auc(fpr, tpr)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1 = np.max(f1_scores)

    return recall, precision, ap_score, fpr, tpr, roc_auc, pr_thresholds

def visualize_evaluations(recall, precision, ap_score, fpr, tpr, roc_auc, pr_thresholds):
    # --- PR Curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP = {ap_score:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.show()

    # --- ROC Curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    # --- F1 vs Threshold (PR thresholds) ---
    # Achtung: pr_thresholds hat Länge len(precision)-1 (und len(recall)-1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_thr = f1[1:]  # zu pr_thresholds passend

    best_idx = int(np.argmax(f1_thr))
    best_thr = pr_thresholds[best_idx]
    best_f1 = f1_thr[best_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pr_thresholds, f1_thr)
    ax.axvline(best_thr, linestyle="--")
    ax.scatter([best_thr], [best_f1])
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 vs Threshold (best F1={best_f1:.3f} @ thr={best_thr:.3f})")
    ax.grid(True, alpha=0.3)
    plt.show()

def visualize_trajectory(trajectory, candidates_loop_pos, y_true, day):
    plt.figure(figsize=(12, 12))
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    for i, label in enumerate(tqdm((y_true), desc="Visualizing trajectory")):
        p_i, p_j = candidates_loop_pos[i]

        # plot candidate loop
        if i == 0:
            plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], 'r-', alpha=0.5, label="Descriptor loops")
        else:
            plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], 'r-', alpha=0.5)
        plt.scatter(p_i[0], p_i[1], color="red", s=10, alpha=0.3, zorder=2)
        plt.scatter(p_j[0], p_j[1], color="red", s=10, alpha=0.3, zorder=2)

        # visualize if its also a gt loop
        if label == 1:
            if i == 0:
                plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], 'g-', alpha=0.5, label="Ground truth loops")
            else:
                plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], 'g-', alpha=0.5)
            plt.scatter(p_i[0], p_i[1], color="green", s=4, alpha=0.3, zorder=3)
            plt.scatter(p_j[0], p_j[1], color="green", s=4, alpha=0.3, zorder=3)

    plt.scatter(x, y, color='lightgray', s=15, label='Robot Trajectory', zorder=1)
    plt.title(f"Trajectory Visualization of {day}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def get_gt_trajectory(gt, timestamps):
    gt_sorted = gt.sort_values("timestamp")

    ts_gt = gt_sorted["timestamp"].to_numpy()
    x_gt = gt_sorted["x"].to_numpy()
    y_gt = gt_sorted["y"].to_numpy()

    timestamps = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    mask = (timestamps >= ts_gt[0]) & (timestamps <= ts_gt[-1])
    timestamps = timestamps[mask]

    interpld_x = interp1d(ts_gt, x_gt, kind="linear")
    interpld_y = interp1d(ts_gt, y_gt, kind="linear")

    x = interpld_x(timestamps)
    y = interpld_y(timestamps)

    return np.column_stack((x, y)), mask
