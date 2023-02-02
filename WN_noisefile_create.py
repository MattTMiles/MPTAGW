import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/"

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    pulsar = pulsar.strip("\n")
    #if pulsar not in altpar_psrs:
    os.chdir(enterprise_dir)
    print(pulsar)
    try:
        res_WN = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN/*.json")[0])
        res_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN_NO_ECORR/*.json")[0])
        res_WN_NO_EQUAD = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN_NO_EQUAD/*.json")[0])
        res_WN_NO_EQUAD_NO_ECORR = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN_NO_EQUAD_NO_ECORR/*.json")[0])
        ev_WN = res_WN.log_evidence
        ev_WN_NO_ECORR = res_WN_NO_ECORR.log_evidence
        ev_WN_NO_EQUAD = res_WN_NO_EQUAD.log_evidence
        ev_WN_NO_EQUAD_NO_ECORR = res_WN_NO_EQUAD_NO_ECORR.log_evidence

    
    except:
        print("Can't collect white noise for: "+pulsar)
        continue

    if ev_WN > ev_WN_NO_ECORR and ev_WN > ev_WN_NO_EQUAD and ev_WN > ev_WN_NO_EQUAD_NO_ECORR:
        chosen = res_WN
    elif ev_WN_NO_ECORR > ev_WN and ev_WN_NO_ECORR > ev_WN_NO_EQUAD and ev_WN_NO_ECORR > ev_WN_NO_EQUAD_NO_ECORR:
        chosen = res_WN_NO_ECORR
    elif ev_WN_NO_EQUAD > ev_WN and ev_WN_NO_EQUAD > ev_WN_NO_ECORR and ev_WN_NO_EQUAD > ev_WN_NO_EQUAD_NO_ECORR:
        chosen = res_WN_NO_EQUAD
    else:
        chosen = res_WN_NO_EQUAD_NO_ECORR

    for parlab in chosen.parameter_labels:
        if "efac" in parlab or "equad" in parlab or "ecorr" in parlab:
            total_dict[parlab] = chosen.posterior.iloc[chosen.posterior.log_likelihood.idxmax()][parlab]


os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/")

with open("MPTA_WN_models.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
