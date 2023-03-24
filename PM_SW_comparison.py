import numpy as np
import os
import glob
import json
import bilby

sw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/PM_WN/"
pm_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM"
#wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json"))

#wnkeys = list(wn_json.keys())

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"

os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/")

for psr in open(pulsar_list,"r").readlines():
    psr = psr.strip("\n")

    chosenpmwn = pm_dir + "/" + psr + "_PM_WN"
    chosenwn = pm_dir + "/" + psr + "_WN"

    if os.path.exists(chosenpmwn):
        chosen = chosenpmwn
        chosen_short = psr + "_PM_WN"
        chosensw = psr + "_PM_WN_SW"
    elif os.path.exists(chosenwn):
        chosen = chosenwn
        chosen_short = psr + "_WN"
        chosensw = psr + "_WN_SW"
    else:
        print("Not there")
        continue

    sw_psr_dir = sw_dir + "/" + psr + "/" + chosensw

    try:
        res_sw = bilby.result.read_in_result(glob.glob(sw_psr_dir+"/*json")[0]).log_evidence
    except:
        print("Not done yet")
        continue

    res_pm = bilby.result.read_in_result(glob.glob(chosen+"/*json")[0]).log_evidence


    if res_sw > res_pm + 2.3:
        print(psr + ": Solar wind model is preferred")
        os.system("echo "+chosensw+" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt")
    else:
        #if "ecorr" in wnmodels:
        print(psr + ": Solar wind not preferred")
        os.system("echo "+chosen_short+" >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/SW_vs_PM.txt")



    





