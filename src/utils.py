from src.read_vel_hits import read_lidar_scans
from src.descriptor import M2DP_Vectorized
import numpy as np
from tqdm import tqdm
import os
import glob

def read_lidar_scans(path):
    try:
        with open(path, 'rb') as f:
            content = f.read()
        num_points = len(content) // 8
        dt = np.dtype([('x', np.uint16), ('y', np.uint16), ('z', np.uint16), ('i', np.uint8), ('l', np.uint8)])
        raw = np.frombuffer(content, dtype=dt)
        
        points = np.empty((num_points, 3), dtype=np.float32)
        # NCLT scaling: (value * 0.005) - 100.0
        points[:, 0] = (raw['x'].astype(np.float32) * 0.005) - 100.0
        points[:, 1] = (raw['y'].astype(np.float32) * 0.005) - 100.0
        points[:, 2] = (raw['z'].astype(np.float32) * 0.005) - 100.0
        return points
    except Exception as e:
        return np.zeros((0, 3))

def calculate_m2dp_descriptor(folder, output_dir, l=8, t=16, p=4, q=4, save=True):
    session_day = folder.parent.name
    descriptors = []
    bin_files = sorted(folder.glob("*.bin"))
    print(f"Found {len(bin_files)} scans in {folder}.")

    # read bin files and calculate M2DP descriptor for each point cloud
    for file_path in tqdm(bin_files, desc=f"Processing bin files from {session_day}"):
        pc = read_lidar_scans(file_path)
        m2dp = M2DP_Vectorized(pc, l, t, p, q)
        descriptor = m2dp.get_signature_vector()
        descriptors.append(descriptor)
    
    descriptors_np = np.array(descriptors)

    # save descriptors
    if save:
        output_filename = f"m2dp_{session_day}.npy"
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir+"/"+output_filename, descriptors_np)
        print(f"Done. Matrix Shape: {descriptors_np.shape}")
        print(f"Saved to {output_dir+"/"+output_filename}\n")
