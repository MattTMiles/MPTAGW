import numpy as np
import os
import glob
import json
import bilby

pm_wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_400"
wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc"
wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/WN_params_inc_extra_ecorr.json"))

wnkeys = list(wn_json.keys())

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise/")

for psr in open(pulsar_list,"r").readlines():
    psr = psr.strip("\n")

    wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if psr in wn_model ]
    
    if "ecorr" in wnmodels:
        wn_chosen = "WN"
    else:
        wn_chosen = "WN_NO_ECORR"

    wn_psr_dir = wn_dir + "/" + psr + "_" + wn_chosen
    pmwn_psr_dir = pm_wn_dir + "/" + psr + "_PM_WN"

    res_wn = bilby.result.read_in_result(glob.glob(wn_psr_dir+"/*json")[0]).log_evidence
    res_pmwn = bilby.result.read_in_result(glob.glob(pmwn_psr_dir+"/*json")[0]).log_evidence

    if res_pmwn > res_wn + 2:
        print(psr + ": Full model is preferred")
        os.system("echo "+psr+": Full model is preferred >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/PMWN_vs_WN_BF2.txt")
    else:
        if "ecorr" in wnmodels:
            print(psr + ": WN model is preferred")
            os.system("echo "+psr+": WN model is preferred >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/PMWN_vs_WN_BF2.txt")
        else:
            print(psr + ": WN_NO_ECORR model is preferred")
            os.system("echo "+psr+": WN_NO_ECORR model is preferred >> /fred/oz002/users/mmiles/MPTA_GW/enterprise/PMWN_vs_WN_BF2.txt")


    





