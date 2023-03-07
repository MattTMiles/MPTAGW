import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/SW_vs_PM.txt"

wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/"
pm_wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/"

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    pulsar = pulsar.strip("\n")
    models = pulsar.split("_")[1:]
    pulsar = pulsar.split("_")[0]
    #if pulsar not in altpar_psrs:
    os.chdir(pm_wn_dir)
    print(pulsar)


    #sw_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_SW_models.json"))
    ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_noise_models.json"))
    keys = list(ev_json.keys())
    #swkeys = list(sw_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]

    for i, pm in enumerate(psrmodels):
        if pm == "SWDET":
            res_SW = bilby.result.read_in_result(glob.glob(pm_wn_dir+"/"+pulsar+"/"+pulsar+"_WN_SW/*.json")[0])
            
            for parlab in res_SW.parameter_labels:
                if "n_earth" in parlab:
                    total_dict[parlab] = res_SW.posterior.iloc[res_SW.posterior.log_likelihood.idxmax()][parlab]

with open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_SW_models.json","a+") as outfile:
#with open("MPTA_WN_models_check.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
    


"/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_SW_models.json"