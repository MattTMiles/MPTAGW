# At the moment this code compares the requirements of including EQUAD in the models.

import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
#pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/trusted_noise"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_400"
active_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/"
ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_active_noise.json"))
equad_check = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/equad_check/"
keys = list(ev_json.keys())

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    os.chdir(enterprise_dir)
    pulsar = pulsar.strip("\n")
    print(pulsar)
    #psrmodel = [ psr_model for psr_model in keys if pulsar in psr_model ][0]
    try:
        model_result = bilby.result.read_in_result(glob.glob(active_dir+"/"+pulsar+"*/*json")[0])
        #wn_result = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+psrmodel+"/*json"))
    except:
        print("Can't collect model noise for: "+pulsar)
        continue

    try:
        result_no_equad = bilby.result.read_in_result(enterprise_dir+"/"+pulsar+"_PM_WN_NO_EQUAD/PM_WN_NO_EQUAD_result.json")
        #result_no_ecorr = bilby.result.read_in_result(enterprise_dir+"/"+pulsar+"_WN_NO_ECORR/WN_NO_ECORR_result.json")
    except:
        print("Can't collect noise for: "+pulsar)
        continue
    model_ev = model_result.log_evidence
    
    ev_no_equad = result_no_equad.log_evidence
    #ev_no_ecorr = result_no_ecorr.log_evidence

    if model_ev > ev_no_equad:
        chosen = model_result
        print("PM_WN is chosen")
        total_dict[pulsar+"_PM_WN"] = model_ev
        pref_dir = glob.glob(active_dir+"/"+pulsar+"*")[0]
        os.system("cp -r "+pref_dir+" "+equad_check)
    else:
        chosen = result_no_equad
        print("PM_WN_NO_EQUAD is chosen")
        total_dict[pulsar+"_PM_WN_NO_EQUAD"] = ev_no_equad
        pref_dir = glob.glob(enterprise_dir+"/"+pulsar+"_PM_WN_NO_EQUAD")[0]
        os.system("cp -r "+pref_dir+" "+equad_check)

    #max_L = chosen.log_likelihood_evaluations.max()

    #if model_ev >= max_L:
    #    print(pulsar+": Model is preferred")
    #    os.system("echo "+pulsar+": Model is preferred >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/WN_fairness_check_0711.txt")
    #else:
    #    print(pulsar+": WN is preferred")
    #    os.system("echo "+pulsar+": WN is preferred >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/WN_fairness_check_0711.txt")

#os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise")

#os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models")
#with open("WN_EQUAD_comparison.json","a+") as outfile:
#    json.dump(total_dict,outfile,indent=4)
