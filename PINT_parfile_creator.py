import numpy
import os
import bilby
import glob
import json
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
partim = "/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/tdb_partim_w_noise"
noise_dir1 = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/PM_WN/"
noise_dir2 = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/"
#noise_dir
os.chdir(partim)

gw_amp = -14.287
gw_gamma = 4.33
gw_c = 30

noise_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_noise_models.json"))
keys = list(noise_json.keys())

for pulsar in open(pulsar_list).readlines():
#for pulsar in ["J1909-3744"]:
    pulsar = pulsar.strip("\n")
    print(pulsar)
    psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]

    openpar = open(partim+"/"+pulsar+"_tdb.par", "a")
    try:

        result_WN = bilby.result.read_in_result(glob.glob(noise_dir1+"/"+pulsar+"/"+pulsar+"*PM_WN_SW/*json")[0])
        result_RN = bilby.result.read_in_result(glob.glob(noise_dir2+"/"+pulsar+"/"+pulsar+"*SPGWC1000/*json")[0])
        
        for par in result_WN.parameter_labels:
            print(par)
            if pulsar+"_KAT_MKBF_efac" in par:
                openpar.write("TNGlobalEF {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
            
            if pulsar+"_KAT_MKBF_log10_ecorr" in par:
                ent_ecorr = result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]
                #tn_ecorr = (10**(ent_ecorr))*1e6
                openpar.write("TNECORR -f KAT_MKBF {} \n".format(ent_ecorr))
            
            if pulsar+"_KAT_MKBF_log10_tnequad" in par:
                openpar.write("TNGlobalEQ {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
            
            if pulsar+"_n_earth_n_earth" in par:
                if "SW" not in psrmodels and "SWDET" not in psrmodels:
                    openpar.write("SWNEARTH {} \n".format(4))
                elif "SWDET" in psrmodels:
                    openpar.write("SWNEARTH {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
                else:
                    openpar.write("SWNEARTH {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
                    #for par in result_WN.parameter_labels:
            if pulsar+"_gp_sw_gamma" in par:
                if "SW" in psrmodels and "SWDET" not in psrmodels:
                    openpar.write("SWGAM {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
                else:
                    openpar.write("SWGAM {} \n".format(0))
                #for par in result_WN.parameter_labels:
            if pulsar+"_gp_sw_log10_A" in par:
                if "SW" in psrmodels and "SWDET" not in psrmodels:
                    openpar.write("SWAMP {} \n".format(result_WN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
                    openpar.write("SWC {} \n".format(30))
                else:
                    openpar.write("SWAMP {} \n".format(0))
                    openpar.write("SWC {} \n".format(30))

        
        for par in result_RN.parameter_labels:
            if pulsar+"_dm_gp_log10_A" in par:
                ent_dm_amp = result_RN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]
                #tn_dm_amp = ent_dm_amp
                #tn_dm_amp = ent_dm_amp + np.log10((2.41e-16*(1.4e9**2))/np.sqrt(12*(np.pi**2)))
                openpar.write("TNDMAmp {} \n".format(ent_dm_amp))
                for par in result_RN.parameter_labels:
                    if pulsar+"_dm_gp_gamma" in par:
                        openpar.write("TNDMGam {} \n".format(result_RN.posterior.iloc[result_WN.posterior.log_likelihood.idxmax()][par]))
                openpar.write("TNDMC {} \n".format(30))
                    
            
            if pulsar+"_low_band_noise_low_gamma" in par:
                bl_gamma = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                for par in result_RN.parameter_labels:
                    if pulsar+"_low_band_noise_low_log10_A" in par:
                        bl_amp = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                openpar.write("TNBANDAMP {} \n".format(bl_amp))
                openpar.write("TNBANDGAM {} \n".format(bl_gamma))
                openpar.write("TNBANDC {} \n".format(30))
                openpar.write("TNBANDLOW {} \n".format(800))
                openpar.write("TNBANDHIGH {} \n".format(1284))

            if pulsar+"_high_band_noise_high_gamma" in par:
                bh_gamma = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                for par in result_RN.parameter_labels:
                    if pulsar+"_high_band_noise_high_log10_A" in par:
                        bh_amp = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                openpar.write("TNBANDAMP {} \n".format(bh_amp))
                openpar.write("TNBANDGAM {} \n".format(bh_gamma))
                openpar.write("TNBANDC {} \n".format(30))
                openpar.write("TNBANDLOW {} \n".format(1284))
                openpar.write("TNBANDHIGH {} \n".format(1700))

            if pulsar+"_chrom_gp_log10_A" in par:
                chrom_amp = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                chrom_idx = 4
                for par in result_RN.parameter_labels:
                    if pulsar+"_chrom_gp_gamma" in par:
                        chrom_gamma = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                    if pulsar+"_chrom_gp_idx" in par:
                        chrom_idx = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                
                openpar.write("TNChromAmp {} \n".format(chrom_amp))
                openpar.write("TNChromGam {} \n".format(chrom_gamma))
                openpar.write("TNChromIdx {} \n".format(chrom_idx))
                openpar.write("TNChromC {} \n".format(30))
            
            if pulsar+"_red_noise_log10_A" in par:
                red_amp = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                for par in result_RN.parameter_labels:
                    if pulsar+"_red_noise_gamma" in par:
                        red_gamma = result_RN.posterior.iloc[result_RN.posterior.log_likelihood.idxmax()][par]
                
                print("writing red noise")
                openpar.write("TNRedAmp {} \n".format(red_amp))
                openpar.write("TNRedGam {} \n".format(red_gamma))
                openpar.write("TNRedC {} \n".format(30))
        
        print("writing ML GW signal for entire PTA")
        openpar.write("TNGWAMP {} \n".format(gw_amp))
        openpar.write("TNGWGAM {} \n".format(gw_gamma))
        openpar.write("TNGWC {} \n".format(gw_c))

        openpar.seek(0)
    except:
        print(pulsar+" isn't there right now")
        



        


        

