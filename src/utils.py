from src.read_vel_hits import read_lidar_scans
from src.descriptor import M2DP_Vectorized
import numpy as np

def read_scan_session(path, m2dp):
    f_bin = open(path, "rb")

    total_hits = 0
    first_utime = -1
    last_utime = -1

    descriptors = []

    while True:
        scan = read_lidar_scans(f_bin)
        if scan["magic"] == "eof":
            break

        total_hits += scan["num_hits"]
        if first_utime == -1:
            first_utime = scan["utime"]
        last_utime = scan["utime"]

        try:
            d = m2dp.compute(scan["pc"])
            descriptors.append(d)
        except ValueError:
            continue # skip scans with few points

    f_bin.close()
    print ("Read %d total hits from %ld to %ld" % (total_hits, first_utime, last_utime))

    return descriptors


def calculate_m2dp_descriptor(folder, output_dir, l=8, t=16, p=4, q=4, save=True):
    path = folder.as_posix() + "/velodyne_hits.bin"
    m2dp = M2DP_Vectorized(l, t, p, q)

    print(f"Starting M2DP algorithm on {folder.name}...")
    descriptors = read_scan_session(path, m2dp)
    descriptors = np.array(descriptors)

    if save:
        output_filename = f"m2dp_{folder.name}.npy"
        np.save(output_dir+"/"+output_filename, descriptors)
        print(f"Done. Matrix Shape: {descriptors.shape}")
        print(f"Saved to {output_dir+"/"+output_filename}")
