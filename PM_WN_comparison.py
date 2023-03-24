import numpy as np
import os
import glob
import json
import bilby

pm_wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/"
wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/WN/"
wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json"))

wnkeys = list(wn_json.keys())

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/")

for psr in open(pulsar_list,"r").readlines():
    psr = psr.strip("\n")

    #wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if psr in wn_model ]
    
    #if "ecorr" in wnmodels:
    wn_chosen = "WN"
    #else:
    #    wn_chosen = "WN_NO_ECORR"

    wn_psr_dir = wn_dir + "/" + psr + "/" + psr + "_" + wn_chosen
    pmwn_psr_dir = pm_wn_dir + "/" + psr + "/" + psr + "_PM_WN"

    res_wn = bilby.result.read_in_result(glob.glob(wn_psr_dir+"/*json")[0]).log_evidence
    try:
        res_pmwn = bilby.result.read_in_result(glob.glob(pmwn_psr_dir+"/*json")[0]).log_evidence
    except:
        print(psr + " is not yet complete.")

    if res_pmwn > res_wn + 2:
        print(psr + ": Full model is preferred")
        os.system("echo "+psr+" PMWN >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/PMWN_vs_WN_BF2.txt")
        os.system("cp -r "+pmwn_psr_dir+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")
    else:
        #if "ecorr" in wnmodels:
        print(psr + ": WN model is preferred")
        os.system("echo "+psr+" WN >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/PMWN_vs_WN_BF2.txt")
        os.system("cp -r "+wn_psr_dir+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")



    





