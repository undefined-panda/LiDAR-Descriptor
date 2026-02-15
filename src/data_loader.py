import numpy as np
from src.utils import remove_nan
import pandas as pd
from tqdm import tqdm
import os

def save_timestamps(folder, output_folder):
    bin_files = folder.glob("*.bin")
    output_dir = f"{output_folder}/{folder.parent.name}.csv"
    os.makedirs(output_folder, exist_ok=True)

    file_names = []

    for file_path in tqdm(bin_files, desc=f"Writing timestamps in {output_dir}"):
        file_name = file_path.name
        file_names.append(file_name.removesuffix(".bin"))

    df = pd.DataFrame(file_names, columns=["timestamps"])
    df.to_csv(output_dir, index=False)
    print("CSV saved.\n")

def load_descriptors(descriptors_folder, day):
    descriptors_path = f"{descriptors_folder}/descriptors/m2dp_{day}.npy"
    descriptors = np.load(descriptors_path)
    print(f"Descriptor: {descriptors_path} | Shape: {descriptors.shape}")
    
    return descriptors

def load_descriptors_timestamps(descriptors_folder, day):
    timestamps_path = f"{descriptors_folder}/timestamps/timestamps_{day}.npy"
    timestamps = np.load(timestamps_path)
    print(f"Descriptors timestamps: {timestamps_path}")

    return timestamps

def load_gt_data(gt_pose_folder, day):
    gt_csv_path = f"{gt_pose_folder}/groundtruth_{day}.csv"
    gt = pd.read_csv(gt_csv_path, 
                     header=None, 
                     names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"]
                     )
    gt = remove_nan(gt)
    print(f"Ground Truth data: {gt_csv_path}")

    return gt

def load_place_recognition_data(p, descriptors_folder, gt_pose_folder):
    day = p.name.split("_")[1].removesuffix(".npy")

    descriptors = load_descriptors(descriptors_folder, day)
    timestamps = load_descriptors_timestamps(descriptors_folder, day)
    gt = load_gt_data(gt_pose_folder, day)

    print()

    return descriptors, timestamps, gt
