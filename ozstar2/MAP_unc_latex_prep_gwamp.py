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


mpta_overall = {}
for dirname in sorted(glob.glob("J*SGWB")):

    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/" + dirname

    f = open(pulsar_dir+"/pars.txt", "r")
    pars = list(f.readlines())
    f.close()

    file = sorted(glob.glob(pulsar_dir+"/*json"))[0]

    pulsar = dirname.split("_")[0]

    ptrim = [ p.lstrip(pulsar+"_").rstrip("\n") for p in pars ]
    ptrimwn = [ "EFAC" if p == "KAT_MKBF_efac" else "EQUAD" if p == "KAT_MKBF_log10_tnequad" else "ECORR" if p == "basis_ecorr_KAT_MKBF_log10_ecorr" else "ECORR" if p == "KAT_MKBF_log10_ecorr" else p for p in ptrim ]
    #pfinal = [ p if p != "nmodel" for p in ptrimwn ]

    map_vals = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_GW_DUMMY_VALS.json"))
    psr_maps = [map_vals[mvkey] for mvkey in map_vals.keys() if pulsar in mvkey]
    temp_df = pd.read_json(json.load(open(file)))

    quant_df = temp_df.quantile([0.16,0.84])
    #quant_df = quant_df.drop("log10_A_gw", axis=1)

    #mpta_overall[pulsar+"_gwamp"] = quant_df["log10_A_gw"]

    #for i, col in enumerate(quant_df.columns):

        #if "log10_A_gw" in col:
    col = "log10_A_gw"
    colmap = psr_maps[0]
    col_low = float(quant_df[col].values[0]) - float(colmap)
    col_up = float(quant_df[col].values[1]) - float(colmap)

    mpta_overall[pulsar+"_"+col] = "${{{:.2f}}}_{{{:+.2f}}}^{{{:+.2f}}}$".format(colmap, col_low, col_up)

    # for i, col in enumerate(quant_df.columns):
    #     if "chrom_bump_t0" in col:
    #         colmap = psr_maps[i]/86400
    #         col_low = (float(quant_df[col].values[0])/86400 - float(colmap))
    #         col_up = (float(quant_df[col].values[1])/86400 - float(colmap))
    #     elif "chrom_bump_sigma" in col:
    #         colmap = psr_maps[i]/86400
    #         col_low = (float(quant_df[col].values[0])/86400 - float(colmap))
    #         col_up = (float(quant_df[col].values[1])/86400 - float(colmap))
    #     else:
    #         colmap = psr_maps[i]
    #         col_low = float(quant_df[col].values[0]) - float(colmap)
    #         col_up = float(quant_df[col].values[1]) - float(colmap)

    #     if "chrom_bump_sign_param" not in col:
    #         mpta_overall[col] = "${{{:.2f}}}_{{{:+.2f}}}^{{{:+.2f}}}$".format(colmap, col_low, col_up)
    #     else:
    #         if colmap > 0:
    #             mpta_overall[col] = "$+$"
    #         elif colmap < 0:
    #             mpta_overall[col] = "$-$"

    print(pulsar+" done")

with open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_overall_val_unc_gwAmp.json","a+") as outFile:
    json.dump(mpta_overall,outFile,indent=4)
        
    

