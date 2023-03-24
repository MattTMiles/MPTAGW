import numpy as np
import os
import glob
import json
import bilby

pm_wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/"
wn_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/WN/"
wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json"))

wnkeys = list(wn_json.keys())

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt"

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/")

for psrmodels in open(pulsar_list,"r").readlines():
    psrmodel = psrmodels.strip("\n")
    psrname = psrmodel.split("_")[0]
    psrnoise = psrmodel.split("_")[1:]

    pm_wn_sw_dir = pm_wn_dir+"/"+psrname+"/"+psrname+"_PM_WN_SW"
    pm_wn_dir_p = pm_wn_dir+"/"+psrname+"/"+psrname+"_PM_WN"
    wn_sw_dir = pm_wn_dir+"/"+psrname+"/"+psrname+"_WN_SW"
    wn_dir_p = wn_dir+"/"+psrname+"/"+psrname+"_WN"

    if psrmodel == psrname+"_PM_WN_SW" or psrmodel == psrname+"_PM_WN_SWDET":
        os.system("cp -r "+pm_wn_sw_dir+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")

    elif psrmodel == psrname+"_PM_WN":
        os.system("cp -r "+pm_wn_dir_p+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")

    elif psrmodel == psrname+"_WN_SW" or psrmodel == psrname+"_WN_SWDET":
        os.system("cp -r "+wn_sw_dir+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")

    elif psrmodel == psrname+"_WN":
        os.system("cp -r "+wn_dir_p+" /fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/")






    





