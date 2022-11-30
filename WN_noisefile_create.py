import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/"

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    os.chdir(enterprise_dir)
    pulsar = pulsar.strip("\n")
    print(pulsar)
    try:
        result = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"*/*.json")[0])
    except:
        print("Can't collect white noise for: "+pulsar)
        continue
    ev = result.log_evidence

    '''
    #if pulsar != "J1231-1411" and pulsar != "J1455-3330" and pulsar != 
    if ev > ev_no_ecorr + 4:
        chosen = result
    else:
        chosen = result_no_ecorr
    '''

    for parlab in result.parameter_labels:
        if "efac" in parlab or "equad" in parlab or "ecorr" in parlab:
            total_dict[parlab] = result.get_one_dimensional_median_and_error_bar(parlab).median

    #if pulsar+"_KAT_MKBF_log10_ecorr" not in chosen.parameter_labels:
    #    total_dict[pulsar+"_KAT_MKBF_log10_ecorr"] = -9

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/")

with open("MPTA_WN_models.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
