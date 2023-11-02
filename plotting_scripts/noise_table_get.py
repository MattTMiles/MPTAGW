import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math
import json


noisefile = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_values_allpulsars_redone.json"


pulsar_list_txt = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
pulsar_list = list(open(pulsar_list_txt,"r"))

noisejson = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_values_allpulsars_redone.json"))

latex_file_position = "/fred/oz002/users/mmiles/MPTA_GW//fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_1Dmed_and_unc_new.txt"

for pulsar in pulsar_list:
    print(pulsar)
    pulsar = pulsar.strip("\n")
    psrmodels = [ nkey for nkey in noisejson.keys() if pulsar in nkey ]

    efac = "-"
    ecorr = "-"
    equad = "-"
    dm_gamma = "-"
    dm_amp = "-"
    red_amp = "-"
    red_gamma = "-"
    sw_gamma = "-"
    sw_amp = "-"
    n_earth = "4"
    chrom_gamma = "-"
    chrom_amp = "-"
    chrom_idx = "-"

    for n in psrmodels:
        print(n)
        if n == pulsar+"_KAT_MKBF_efac":
            efac = noisejson[n]
        if n == pulsar+"_KAT_MKBF_log10_ecorr":
            ecorr = noisejson[n]
        if n == pulsar+"_KAT_MKBF_log10_tnequad":
            equad = noisejson[n]
        if n == pulsar+"_dm_gp_gamma":
            dm_gamma = noisejson[n]
        if n == pulsar+"_dm_gp_log10_A":
            dm_amp = noisejson[n]
        if n == pulsar+"_red_noise_log10_A":
            red_amp = noisejson[n]
        if n == pulsar+"_red_noise_gamma":
            red_gamma = noisejson[n]
        if n == pulsar+"_gp_sw_gamma":
            sw_gamma = noisejson[n]
        if n == pulsar+"_gp_sw_log10_A":
            sw_amp = noisejson[n]
            print(sw_amp)
        if n == pulsar+"_n_earth_n_earth":
            n_earth = noisejson[n]
        if n == pulsar+"_chrom_gp_gamma":
            chrom_gamma = noisejson[n]
        if n == pulsar+"_chrom_gp_log10_A":
            chrom_amp = noisejson[n]
            #for n in psrmodels:
            if pulsar+"_chrom_gp_idx" in psrmodels:
                chrom_idx = noisejson[pulsar+"_chrom_gp_idx"]
            else:
                chrom_idx = 4
        if n == pulsar+"_chrom_wide_gp_gamma":
            chrom_gamma = noisejson[n]
        if n == pulsar+"_chrom_wide_gp_log10_A":
            chrom_amp = noisejson[n]
            #for n in psrmodels:
            if pulsar+"_chrom_wide_gp_idx" in psrmodels:
                chrom_idx = noisejson[pulsar+"_chrom_wide_gp_idx"]
            else:
                chrom_idx = 4

    print(pulsar,efac, equad, ecorr, red_amp,red_gamma,dm_amp,dm_gamma,chrom_amp,chrom_gamma,chrom_idx,sw_amp,sw_gamma,n_earth)
    os.system(r"echo '{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} & {9} & {10} & {11} & {12} & {13} \\' >> ".format(pulsar,efac, equad, ecorr, red_amp,red_gamma,dm_amp,dm_gamma,chrom_amp,chrom_gamma,chrom_idx,sw_amp,sw_gamma,n_earth)+latex_file_position)
