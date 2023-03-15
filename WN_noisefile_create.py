import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/"

wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/"
pm_wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/"

total_dict = {}

for pulsar in open(pulsar_list,"r").readlines():
    pulsar = pulsar.strip("\n")
    models = pulsar.split("_")[1:]
    pulsar = pulsar.split("_")[0]
    #if pulsar not in altpar_psrs:
    os.chdir(enterprise_dir)
    print(pulsar)
    '''
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
    '''
    try:

        if models[0] == "WN":
            if len(models) > 1 and "SW" in models[1]:
                res_WN = bilby.result.read_in_result(glob.glob(pm_wn_dir+"/"+pulsar+"/"+pulsar+"_WN_SW/*.json")[0])
            else:
                res_WN = bilby.result.read_in_result(glob.glob(wn_dir+"/"+pulsar+"/"+pulsar+"_WN/*.json")[0])
        
        if models[0] == "PM":
            if len(models) > 1 and "SW" in models[-1]:
                res_WN = bilby.result.read_in_result(glob.glob(pm_wn_dir+"/"+pulsar+"/"+pulsar+"_PM_WN_SW/*.json")[0])
            else:
                res_WN = bilby.result.read_in_result(glob.glob(pm_wn_dir+"/"+pulsar+"/"+pulsar+"_PM_WN/*.json")[0])
        '''
        if ev_WN > ev_WN_NO_ECORR and ev_WN > ev_WN_NO_EQUAD and ev_WN > ev_WN_NO_EQUAD_NO_ECORR:
            chosen = res_WN
        elif ev_WN_NO_ECORR > ev_WN and ev_WN_NO_ECORR > ev_WN_NO_EQUAD and ev_WN_NO_ECORR > ev_WN_NO_EQUAD_NO_ECORR:
            chosen = res_WN_NO_ECORR
        elif ev_WN_NO_EQUAD > ev_WN and ev_WN_NO_EQUAD > ev_WN_NO_ECORR and ev_WN_NO_EQUAD > ev_WN_NO_EQUAD_NO_ECORR:
            chosen = res_WN_NO_EQUAD
        else:
            chosen = res_WN_NO_EQUAD_NO_ECORR
        '''
        
        for parlab in res_WN.parameter_labels:
            if "efac" in parlab or "equad" in parlab or "ecorr" in parlab:
                total_dict[parlab] = res_WN.posterior.iloc[res_WN.posterior.log_likelihood.idxmax()][parlab]
    
    except:
        print(pulsar + " is not done yet")


os.chdir("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/")

with open("MPTA_WN_models_new.json","a+") as outfile:
#with open("MPTA_WN_models_check.json","a+") as outfile:
    json.dump(total_dict,outfile,indent=4)
