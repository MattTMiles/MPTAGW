import numpy
import os
import bilby
import glob
import json
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"
partim = "/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/partim_noise_removed/gaussian_ecorr"

noise_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/"
#noise_dir
os.chdir(partim)

#gw_amp = -14.287
#gw_gamma = 4.33
#gw_c = 30

noise_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_SPGW_noise_values.json"))
keys = list(noise_json.keys())

white_noise_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models.json"))
white_noise_keys = list(white_noise_json.keys())


for pulsar in open(pulsar_list).readlines():
#for pulsar in ["J1909-3744"]:
    pulsar = pulsar.strip("\n")
    print(pulsar)
    psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ]

    wnmodels = [ wn_model for wn_model in white_noise_keys if pulsar in wn_model ]

    openpar = open(partim+"/"+pulsar+"_tdb.par", "a")
    
    for wnkey in wnmodels:
        if "efac" in wnkey:
            openpar.write("EFAC -f KAT_MKBF {} \n".format(white_noise_json[wnkey]))
        if "equad" in wnkey:
            openpar.write("TNEQ -f KAT_MKBF {} \n".format(white_noise_json[wnkey]))
        if "ecorr" in wnkey:
            openpar.write("TNECORR -f KAT_MKBF {} \n".format(white_noise_json[wnkey]))

    for key in psrmodels:
        #val = noise_json[key]
        if "earth" in key:
            openpar.write("SWNEARTH {} \n".format(noise_json[key]))
        if "gp_sw_gamma" in key:
            openpar.write("SWGAM {} \n".format(noise_json[key]))
        if "gp_sw_log10_A" in key:
            openpar.write("SWAMP {} \n".format(noise_json[key]))
            openpar.write("SWC {} \n".format(120))
        if "dm_gp_gamma" in key:
            openpar.write("TNDMGam {} \n".format(noise_json[key]))
        if "dm_gp_log10_A" in key:
            openpar.write("TNDMAmp {} \n".format(noise_json[key]))
            openpar.write("TNDMC {} \n".format(120))
        if "chrom_gp_gamma" in key:
            openpar.write("TNChromGam {} \n".format(noise_json[key]))
        if "chrom_gp_log10_A" in key:
            openpar.write("TNChromAmp {} \n".format(noise_json[key]))
            openpar.write("TNChromC {} \n".format(120))
        if "chrom_gp_idx" in key:
            openpar.write("TNChromIdx {} \n".format(noise_json[key]))
        if "red_noise_gamma" in key:
            openpar.write("TNRedGam {} \n".format(noise_json[key]))
        if "red_noise_log10_A" in key:
            openpar.write("TNRedAmp {} \n".format(noise_json[key]))
            openpar.write("TNRedC {} \n".format(120))
    
    if len([ psm for psm in psrmodels if "n_earth" in psm ]) < 1:
        openpar.write("SWNEARTH {} \n".format(0))
        openpar.write("SWGAM {} \n".format(1))
        openpar.write("SWAMP {} \n".format(-20))
        openpar.write("SWC {} \n".format(120))
    if len([ psm for psm in psrmodels if "gp_sw_gamma" in psm ]) < 1:
        if len([ psm for psm in psrmodels if "n_earth" in psm ]) > 0:
            openpar.write("SWGAM {} \n".format(1))
            openpar.write("SWAMP {} \n".format(-20))
            openpar.write("SWC {} \n".format(120))
    if len([ psm for psm in psrmodels if "chrom_gp_idx" in psm ]) < 1:
        if len([ psm for psm in psrmodels if "chrom" in psm ]) > 0:
            openpar.write("TNChromIdx {} \n".format(4))
    
    openpar.write("TNGWAMP {} \n".format(-14.15))
    openpar.write("TNGWGAM {} \n".format(4.33))
    openpar.write("TNGWC {} \n".format(120))



        


        

