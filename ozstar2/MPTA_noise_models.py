import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse


active_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models"
noise_json = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_red_noise_models.json"

psrlist = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt"
psrlist = list(open(psrlist).readlines())
outfile = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"

model_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/"

mpta_models = {}

for pulsar in psrlist:
    pulsarmodel = pulsar.strip("\n")
    psrname = pulsarmodel.split("_")[0]
    
    ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_red_noise_models.json"))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if psrname in psr_model ]
    models = []
    for model in psrmodels:
        noise = model.replace(psrname+"_","")
        models.append(noise)

    noise_read = psrname
    
    if "dm_gp_gamma" in models:
        noise_read += "_DM"
    if "gp_sw_gamma" in models and "n_earth_n_earth" in models:
        noise_read += "_SW"
    if "n_earth_n_earth" in models and "gp_sw_gamma" not in models:
        noise_read += "_SWDET"
    if "chrom_gp_gamma" in models and "chrom_gp_idx" in models:
        noise_read += "_CHROM"
    if "chrom_gp_gamma" in models and "chrom_gp_idx" not in models:
        noise_read += "_CHROMCIDX"
    if "red_noise_gamma" in models:
        noise_read += "_RN"
    if "chrom_wide_gp_gamma" in models and "chrom_wide_gp_idx" in models:
        noise_read += "_CHROM_WIDE"
    if "chrom_wide_gp_gamma" in models and "chrom_wide_gp_idx" not in models:
        noise_read += "_CHROMCIDX_WIDE"

    print(noise_read)

    res = bilby.result.read_in_result(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/"+psrname+"*/*json")[0])
    ev = res.log_evidence
    
    mpta_models[noise_read] = ev


with open(outfile,"a+") as outFile:
    json.dump(mpta_models,outFile,indent=4)

