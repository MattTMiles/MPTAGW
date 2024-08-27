import numpy as np
import enterprise
from enterprise.pulsar import Pulsar
import os

def binner(pos1, pos2):

    if np.all(pos1 == pos2):
        return 1
    else:
        # bins in angsep space
        #bins = np.array([1e-3, 30.0, 50.0, 80.0, 100.0,
        #        120.0, 150.0, 180.0]) * np.pi/180.0
        bins = np.array([1e-3, 25.714, 51.429, 77.143, 102.85,
               128.571, 154.289, 180.0]) * np.pi/180.0
        # bins = np.array([1e-3, 30, 60, 90, 120,
        #         150, 180]) * np.pi/180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        idx = np.digitize(angsep, bins)
        return idx

#os.remove("/fred/oz002/users/mmiles/MPTA_GW/pair_bins.txt")

done = []

f = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
psrs = list(f.readlines())
f2 = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_psr_position.list", "r")
positions = list(f2.readlines())
positions = [ pos.split()[1:] for pos in positions ]
positions = list(filter(None, positions))
pos_array = np.array(positions)
for psr1 in psrs:
    psr1 = psr1.strip("\n")
    index1 = np.where(pos_array == psr1)[0][0]
    psr1_ra = float(pos_array[index1][1])
    psr1_raj = psr1_ra*np.pi/180
    psr1_dec = float(pos_array[index1][2])
    psr1_decj = psr1_dec*np.pi/180
    psr1_pos = np.array([np.cos(psr1_raj) * np.cos(psr1_decj), np.sin(psr1_raj) * np.cos(psr1_decj), np.sin(psr1_decj)])
    for psr2 in psrs:
        psr2 = psr2.strip("\n")
        if psr1 != psr2:
            if psr1+"_"+psr2 not in done:
                if psr2+"_"+psr1 not in done:
                    
                    index2 = np.where(pos_array == psr2)[0][0]
                    psr2_ra = float(pos_array[index2][1])
                    psr2_raj = psr2_ra*np.pi/180
                    psr2_dec = float(pos_array[index2][2])
                    psr2_decj = psr2_dec*np.pi/180
                    psr2_pos = np.array([np.cos(psr2_raj) * np.cos(psr2_decj), np.sin(psr2_raj) * np.cos(psr2_decj), np.sin(psr2_decj)])
                    angbin = binner(psr1_pos,psr2_pos)
                    pair_bins = open("/fred/oz002/users/mmiles/MPTA_GW/pair_bins_7.txt", "a")
                    pair_bins.write(psr1+"_"+psr2+" "+str(angbin)+"\n")
                    pair_bins.close()
                    print(psr1+"_"+psr2+" "+str(angbin))
                    done.append(psr1+"_"+psr2)