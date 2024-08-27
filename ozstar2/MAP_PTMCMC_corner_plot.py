import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
import gc
from scipy.signal import correlate
import pandas as pd


def chain_corner_from_dir(pulsar, parstxt):
    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/" + pulsar

    f = open(pulsar_dir+"/"+parstxt, "r")
    pars = list(f.readlines())
    f.close()

    file = sorted(glob.glob(pulsar_dir+"/*thinned_chain_combined.npy"))[0]

    chain = np.load(file)
    #pulsar = dirname.split("_")[0]

    #pars.remove("nmodel")
    ptrim = [ p.lstrip(pulsar+"_").rstrip("\n") for p in pars ]
    ptrim.remove("nmodel")
    ptrimwn = [ "EFAC" if p == "KAT_MKBF_efac" else "EQUAD" if p == "KAT_MKBF_log10_tnequad" else "ECORR" if p == "basis_ecorr_KAT_MKBF_log10_ecorr" else p for p in ptrim ]
    #pfinal = [ p if p != "nmodel" for p in ptrimwn ]

    map_vals = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/OS_runs/hierarch_MAP_SGWB_vals.json"))

    truths = [ map_vals[pulsar+"_"+pt] for pt in ptrim[:-1] ]
    truths = truths + [-14.25]
    #temp_df = pd.read_json(json.load(open(file)))

    fig = corner.corner(chain,bins=30,labels=list(ptrimwn),quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], smooth=False, smooth1d=False, plot_contours=True, plot_datapoints=False, plot_density=True, levels=[0.393, 0.864, 0.989], fill_contours=False, color="teal", label_kwargs={"fontsize":14}, figsize=(19.20,10.80), truths=truths, truth_color="orange")

    fig.savefig(pulsar_dir+"/SMBHB_WN_corner_w_MAP.png")
    fig.clf()
    plt.clf()
    print(pulsar+" done")
        
    
chain_corner_from_dir(sys.argv[1], "pars.txt")
