import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out"

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    os.chdir(enterprise_dir)
    pulsar = pulsar.strip("\n")
    try:
        result = bilby.result.read_in_result(enterprise_dir+"/"+pulsar+"_white_noise/white_noise_result.json")
        result_no_ecorr = bilby.result.read_in_result(enterprise_dir+"/"+pulsar+"_white_noise_no_ecorr/white_noise_no_ecorr_result.json")
    except:
        continue
    ev = result.log_evidence
    ev_no_ecorr = result_no_ecorr.log_evidence

    if ev > ev_no_ecorr + 4:
        chosen = result
    else:
        chosen = result_no_ecorr

    for parlab in chosen.parameter_labels:
        total_dict[parlab] = result.get_one_dimensional_median_and_error_bar(parlab).median
    
    if pulsar+"_KAT_MKBF_log10_ecorr" not in chosen.parameter_labels:
        total_dict[pulsar+"_KAT_MKBF_log10_ecorr"] = -9

    redresult = bilby.result.read_in_result(enterprise_dir+"/"+pulsar+"_dm_red_nlive1000/dm_red_nlive1000_result.json")
    for redpar in redresult.parameter_labels:
        total_dict[redpar] = redresult.get_one_dimensional_median_and_error_bar(redpar).median

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise")

with open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/noisefile_incRed.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
