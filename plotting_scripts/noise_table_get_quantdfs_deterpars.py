import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math
import json


noisefile = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_PMWN_values.json"


pulsar_list_txt = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"
pulsar_list = list(open(pulsar_list_txt,"r"))

noisejson = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_overall_val_unc.json"))

latex_file_position = "/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_MAP_and_unc_new_deter_params.txt"

for pulsar in pulsar_list:
    print(pulsar)
    pulsar = pulsar.strip("\n")
    psrmodels = [ nkey for nkey in noisejson.keys() if pulsar in nkey ]

    chrombump_amp = ""
    chrombump_idx = ""
    chrombump_t0 = ""
    chrombump_sigma = ""
    chrombump_sign = ""

    chromannual_amp = ""
    chromannual_idx = ""
    chromannual_phase = ""

    for n in psrmodels:
        print(n)
        if n == pulsar+"_chrom_bump_log10_Amp":
            chrombump_amp = noisejson[n]
        if n == pulsar+"_chrom_bump_idx":
            chrombump_idx = noisejson[n]
        if n == pulsar+"_chrom_bump_t0":
            chrombump_t0 = noisejson[n]
        if n == pulsar+"_chrom_bump_sigma":
            chrombump_sigma = noisejson[n]
        if n == pulsar+"_chrom_bump_sign_param":
            chrombump_sign = noisejson[n]

        if n == pulsar+"_chrom1yr_log10_Amp":
            chromannual_amp = noisejson[n]
        if n == pulsar+"_chrom1yr_idx":
            chromannual_idx = noisejson[n]
        if n == pulsar+"_chrom1yr_phase":
            chromannual_phase = noisejson[n]

    print(pulsar,chrombump_amp, chrombump_idx, chrombump_t0, chrombump_sigma,chrombump_sign,chromannual_amp,chromannual_idx,chromannual_phase)
    os.system(r"echo '{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} \\' >> ".format(pulsar,chrombump_amp, chrombump_idx, chrombump_t0, chrombump_sigma,chrombump_sign,chromannual_amp,chromannual_idx,chromannual_phase)+latex_file_position)
