import numpy as np
import matplotlib.pyplot as plt
import bilby
import random

import seaborn as sns
import pandas as pd
import json
import os
import glob

psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"

cross_corr_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/corrAmp_runs/"

pulsar_pair_list = "/fred/oz002/users/mmiles/MPTA_GW/pair_bins_7.txt"

pp_list = list(open(pulsar_pair_list).readlines())

pp_list = [ x.strip("\n") for x in pp_list ]

def reject_outliers(data, m = 1000):
    d = np.abs(data[:,-3] - np.median(data[:,-3]))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0

bin_number = 20

scale_count = 0

for pair in pp_list:
    pairname, pairbin = pair.split()
    # if "J0437" not in pairname or "J0613" not in pairname or "J0900" not in pairname or "J1017" not in pairname or "J1431" not in pairname or \
    # "J1525" not in pairname or "J1643" not in pairname or "J1652" not in pairname or "J1747" not in pairname or "J1802" not in pairname or "J1804-2858" not in pairname \
    # or "J1825" not in pairname or "J1911" not in pairname:
    if "J0000000" not in pairname:
    #if "J0101-6422_J1327-0755" in pairname:
    #if int(pairbin) == 4: 
    #if not "J1024" in pairname or "J1216" in pairname or "J1327" in pairname:
    #if "J1909"  in pairname or "J2241" in pairname or "J1902" in pairname:
    #if "J1327" not in pairname and "J1902-5105" not in pairname and "J2322-2650" not in pairname and "J1036-8317" not in pairname and "J1024-0719" not in pairname:


        print(pairname, pairbin)
        #res_pair = pd.read_json(json.load(open(cross_corr_dir + "/" + pairname + "/" + pairname + "_final_res.json")))
        #res_pair = np.load(cross_corr_dir + "/" + pairname + "/" + pairname + "_burnt_chain.npy")
        try:
            lenchain = os.popen("cat "+cross_corr_dir + "/" + pairname + "/master_chain.txt | wc -l").read().strip("\n")
            burn = int(0.8*int(lenchain))
            res_pair = np.loadtxt(cross_corr_dir + "/" + pairname + "/master_chain.txt", skiprows=burn)
        except:
            lenchain = os.popen("cat "+cross_corr_dir + "/" + pairname + "/master_chain.txt | wc -l").read().strip("\n")
            burn = int(0.8*int(lenchain))
            trialpd = pd.read_csv(cross_corr_dir + "/" + pairname + "/master_chain.txt", sep="\t", header=None, on_bad_lines="skip", skiprows=burn+1)
            cols = trialpd.columns
            trialpd[cols] = trialpd[cols].apply(pd.to_numeric, errors="coerce")
            res_pair = trialpd.values
            res_pair = res_pair[~np.isnan(res_pair).any(axis=1)]
        if len(res_pair) > 500:
            #res_pair = bilby.result.read_in_result(cross_corr_dir + "/" + pairname + "/" + pairname + "_result.json")
            par = glob.glob(cross_corr_dir + "/" + pairname + "/" + pairname + "*/pars.txt")[0]
            pars = list(open(par).readlines())
            if len(pars) != 0:
                pars = [ p.rstrip("\n") for p in pars ]
            else:
                par = glob.glob(cross_corr_dir + "/" + pairname + "/" + pairname + "*/pars.txt")[1]
                pars = list(open(par).readlines())
                pars = [ p.rstrip("\n") for p in pars ]

            corridx = pars.index("gw_single_orf_bin_0")
            ampidx = pars.index("gw_bins_log10_A")
            res_pair = res_pair[::5,:]
            res_pair = reject_outliers(res_pair)
            res_pair = reject_outliers(res_pair)
            res_pair = res_pair[:,[corridx,ampidx]]


            np.save("/fred/oz002/users/mmiles/MPTA_GW/PAIRWISE_NEW/DJR_scripts/fixedNoiseAmpCorr/chains/"+pairname, res_pair)
