from tqdm import tqdm
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.read_vel_sync import read_vel_sync
from src.m2dp_descriptor import compute_m2dp_signature_vectors

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
