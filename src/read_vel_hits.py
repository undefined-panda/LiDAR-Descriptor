# !/usr/bin/python
#
# Example code to go through the velodyne_hits.bin
# file and read timestamps, number of hits, and the
# hits in each packet.
# 
# Source: https://robots.engin.umich.edu/nclt/
# Rewrote it to use it as a library
#
# To call:
#
#   python read_vel_hits.py velodyne.bin
#

import sys
import struct
import numpy as np

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def read_lidar_scans(f_bin):
    pc = []

    magic = f_bin.read(8)
    if magic == '': # eof
        return {"magic": "eof"}

    if not verify_magic(magic):
        print ("Could not verify magic")

    num_hits = struct.unpack('<I', f_bin.read(4))[0]
    utime = struct.unpack('<Q', f_bin.read(8))[0]
        
    padding = f_bin.read(4) # padding

    for i in range(num_hits):

        x = struct.unpack('<H', f_bin.read(2))[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)
        s = [x, y, z, i, l]
        pc.append(s)
    
    scan = {"pc": np.array(pc), "num_hits": num_hits, "utime": utime, "magic": magic}
    return scan
