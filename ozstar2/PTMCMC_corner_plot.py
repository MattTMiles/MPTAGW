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


def chain_corner_from_dir(dirname, parstxt):
    pulsar_dir = dirname

    f = open(pulsar_dir+"/"+parstxt, "r")
    pars = list(f.readlines())
    f.close()

    #file = sorted(glob.glob(pulsar_dir+"/*thinned_chain_combined.npy"))[0]
    chain = np.loadtxt(pulsar_dir+"/chain_1.txt")
    chain = chain[:,:-4]
    pulsar = dirname.split("_")[0]

    #pars.remove("nmodel")
    ptrim = [ p.lstrip(pulsar+"_").rstrip("\n") for p in pars ]
    #ptrim.remove("nmodel")
    ptrimwn = [ "EFAC" if p == "KAT_MKBF_efac" else "EQUAD" if p == "KAT_MKBF_log10_tnequad" else "ECORR" if p == "basis_ecorr_KAT_MKBF_log10_ecorr" else p for p in ptrim ]

    fig = corner.corner(chain,bins=30,labels=list(ptrimwn),quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], smooth=False, smooth1d=False, plot_contours=True, plot_datapoints=False, plot_density=True, levels=[0.393, 0.864, 0.989], fill_contours=False, color="teal", label_kwargs={"fontsize":14}, figsize=(19.20,10.80))

    fig.savefig(pulsar_dir+"/"+pulsar+"_"+"corner.png")
    fig.clf()
    plt.clf()
    print("Made: "+pulsar_dir+"/"+pulsar+"_"+"corner.png")
        
    
chain_corner_from_dir(sys.argv[1], "/pars.txt")
