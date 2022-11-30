import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import json

#gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_400"
gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400"
#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_trusted_noise_281022.txt"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
#noise_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/noise_evidence_27_10_2022.json"
#noise_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/noise_vals_0711_WN_adjusted.json"
noise_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/noise_0911_wide_and_ecorr.json"
data = json.load(open(noise_list))

noise_ev = 0
gw_ev = 0

filename =  "/fred/oz002/users/mmiles/MPTA_GW/enterprise/SP_WN_BF_18.txt"
with open(filename, "a") as f:

    for psr in open(psr_list).readlines():
        psrname = psr.strip("\n")
        print(psrname)

        try:
            psr_SPGWC = gw_dir + "/" + psrname + "_SPGWC_WN"
            result_SPGWC = bilby.result.read_in_result(psr_SPGWC+"/SPGWC_WN_result.json")
            gw_ev = result_SPGWC.log_evidence

            psr_SPGWWC_18 = gw_dir + "/" + psrname + "_SPGWC_18_WN"
            result_SPGWC_18 = bilby.result.read_in_result(psr_SPGWWC_18+"/SPGWC_18_WN_result.json")
            gw_ev_18 = result_SPGWC_18.log_evidence
        
        except:
            continue
        """
        for key in data.keys():
            if psrname in key:
                print(key)
                noise_ev = data[key]
        """

        print("Noise log-evidence: {}".format(gw_ev_18))
        print("GW log-evidence: {}".format(gw_ev))

        bf = gw_ev - gw_ev_18
        f.write(psrname + " " + str(bf) + "\n" )

        print("Log BF: {}".format(bf))

