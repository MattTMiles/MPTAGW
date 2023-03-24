import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt"
chosen_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/"

total_dict = {}

for pulsar_model in open(pulsar_list,"r").readlines():
    os.chdir("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models")
    pulsar = pulsar_model.strip("\n").split("_")[0]
    psrnoise = pulsar_model.strip("\n").split("_")[1:]
    try:
        result = bilby.result.read_in_result(chosen_dir+"/"+pulsar+"_PM_WN/PM_WN_result.json")
        ev = result.log_evidence
        print(chosen_dir+"/"+pulsar+"_PM_WN/PM_WN_result.json")
    except:
        pass

    try:
        result = bilby.result.read_in_result(chosen_dir+"/"+pulsar+"_WN/WN_result.json")
        ev = result.log_evidence
        print(chosen_dir+"/"+pulsar+"_WN/WN_result.json")
    except:
        pass
    try:
        result = bilby.result.read_in_result(chosen_dir+"/"+pulsar+"_WN_SW/WN_SW_result.json")
        ev = result.log_evidence
        print(chosen_dir+"/"+pulsar+"_WN_SW/WN_SW_result.json")
    except:
        pass
    try:
        result = bilby.result.read_in_result(chosen_dir+"/"+pulsar+"_PM_WN_SW/PM_WN_SW_result.json")
        ev = result.log_evidence
        print(chosen_dir+"/"+pulsar+"_PM_WN_SW/PM_WN_SW_result.json")
    except:
        pass


    if "SWDET" not in psrnoise:
        for parlab in result.parameter_labels:
            if "efac" not in parlab and "equad" not in parlab and "ecorr" not in parlab:
                total_dict[parlab] = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][parlab]
    else:
        for parlab in result.parameter_labels:
            if "gp_sw" not in parlab and "efac" not in parlab and "equad" not in parlab and "ecorr" not in parlab:
                total_dict[parlab] = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][parlab]

    

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models")

with open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_red_noise_models.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
