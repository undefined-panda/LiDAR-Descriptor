"""
This script was taken from https://robots.engin.umich.edu/nclt/ (read_vel_sync.py) and is slightly modified.

This script reads the LiDAR scans from the velodyne_sync folder and returns each as a np.array
"""

import struct
import matplotlib.pyplot as plt
import numpy as np

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def read_vel_sync(file_path, csv_output=False, csv_output_dir=None, show_point_cloud=False):

    f_bin = open(file_path, 'rb')

    if csv_output:
        f_csv = open(csv_output_dir, "w")
    else:
        f_csv = None

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        if f_csv:
            f_csv.write('%s\n' % s)

        hits += [[x, y, z]]

    f_bin.close()

    if f_csv:
        f_csv.close()

    hits = np.asarray(hits)

    if show_point_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hits[:, 0], hits[:, 1], -hits[:, 2], c=-hits[:, 2], s=5, linewidths=0)
        plt.show()

    return hits
