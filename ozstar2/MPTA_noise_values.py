import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse


active_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models"
noise_json = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"

psrlist = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"
psrlist = list(open(psrlist).readlines())
outfile = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_values.json"

model_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/"

mpta_models = {}

for pulsar in psrlist:
    psrname = pulsar.strip("\n")
    #psrname = pulsarmodel.split("_")[0]
    
    ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if psrname in psr_model ]
    psrmodels = psrmodels[0].split("_")[1:]
    #models = []
    #for model in psrmodels:
    #    noise = model.replace(psrname+"_","")
    #    models.append(noise)

    
    if "SW" in psrmodels:
        res = bilby.result.read_in_result(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/"+psrname+"*/"+psrname+"_PM_WN_SW_HC/*json")[0])
        for parlab in res.parameter_labels:
            mpta_models[parlab] = res.posterior.iloc[res.posterior.log_likelihood.idxmax()][parlab]
    elif "SWDET" in psrmodels:
        res = bilby.result.read_in_result(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/"+psrname+"*/"+psrname+"_PM_WN_SW_HC/*json")[0])
        for parlab in res.parameter_labels:
            if "gp_sw" not in parlab:
                mpta_models[parlab] = res.posterior.iloc[res.posterior.log_likelihood.idxmax()][parlab]
    else:
        res = bilby.result.read_in_result(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN_HC/"+psrname+"*/"+psrname+"_PM_WN_HC/*json")[0])
        for parlab in res.parameter_labels:
            mpta_models[parlab] = res.posterior.iloc[res.posterior.log_likelihood.idxmax()][parlab]



with open(outfile,"a+") as outFile:
    json.dump(mpta_models,outFile,indent=4)

