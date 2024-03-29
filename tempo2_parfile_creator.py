import numpy
import os
import bilby
import glob
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
partim = "/fred/oz002/users/mmiles/MPTA_GW/partim_noise_input"
noise_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM/"
#noise_dir
os.chdir(partim)

for pulsar in open(pulsar_list).readlines():
#for pulsar in ["J1909-3744"]:
    pulsar = pulsar.strip("\n")
    print(pulsar)

    openpar = open(partim+"/"+pulsar+".par", "a")
    try:

        result = bilby.result.read_in_result(glob.glob(noise_dir+"/"+pulsar+"*/*json")[0])
        #result = bilby.result.read_in_result(glob.glob(noise_dir+"/"+pulsar+"/"+pulsar+"_DM/*json")[0])
        
        for par in result.parameter_labels:
            print(par)
            if pulsar+"_KAT_MKBF_efac" in par:
                openpar.write("TNGlobalEF {} \n".format(result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]))
            
            if pulsar+"_KAT_MKBF_log10_ecorr" in par:
                ent_ecorr = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                tn_ecorr = (10**(ent_ecorr))*1e6
                openpar.write("TNECORR -f KAT_MKBF {} \n".format(tn_ecorr))
            
            if pulsar+"_KAT_MKBF_log10_tnequad" in par:
                openpar.write("TNGlobalEQ {} \n".format(result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]))
            
            if pulsar+"_dm_gp_log10_A" in par:
                ent_dm_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                #tn_dm_amp = ent_dm_amp
                tn_dm_amp = ent_dm_amp + np.log10((2.41e-16*(1.4e9**2))/np.sqrt(12*(np.pi**2)))
                openpar.write("TNDMAmp {} \n".format(tn_dm_amp))
                for par in result.parameter_labels:
                    if pulsar+"_dm_gp_gamma" in par:
                        openpar.write("TNDMGam {} \n".format(result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]))
                openpar.write("TNDMC {} \n".format(30))
                    
            
            if pulsar+"_low_band_noise_low_gamma" in par:
                bl_gamma = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                for par in result.parameter_labels:
                    if pulsar+"_low_band_noise_low_log10_A" in par:
                        bl_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                openpar.write("TNBandNoise {0} {1} {2} {3} {4} \n".format(800, 1284, bl_amp, bl_gamma, 30))
            
            if pulsar+"_high_band_noise_high_gamma" in par:
                bh_gamma = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                for par in result.parameter_labels:
                    if pulsar+"_high_band_noise_high_log10_A" in par:
                        bh_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                openpar.write("TNBandNoise {0} {1} {2} {3} {4} \n".format(1284, 1700, bh_amp, bh_gamma, 30))

            if pulsar+"_chrom_gp_log10_A" in par:
                chrom_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                chrom_idx = 4
                for par in result.parameter_labels:
                    if pulsar+"_chrom_gp_gamma" in par:
                        chrom_gamma = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                    if pulsar+"_chrom_gp_idx" in par:
                        chrom_idx = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                openpar.write("TNChromAmp {} \n".format(chrom_amp))
                openpar.write("TNChromGam {} \n".format(chrom_gamma))
                openpar.write("TNChromIdx {} \n".format(chrom_idx))
                openpar.write("TNChromC {} \n".format(30))
                
            if pulsar+"_chrom_wide_gp_log10_A" in par:
                chrom_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                chrom_idx = 4
                for par in result.parameter_labels:
                    if pulsar+"_chrom_wide_gp_gamma" in par:
                        chrom_gamma = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                    if pulsar+"_chrom_wide_gp_idx" in par:
                        chrom_idx = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                
                openpar.write("TNChromAmp {} \n".format(chrom_amp))
                openpar.write("TNChromGam {} \n".format(chrom_gamma))
                openpar.write("TNChromIdx {} \n".format(chrom_idx))
                openpar.write("TNChromC {} \n".format(30))
            
            if pulsar+"_red_noise_log10_A" in par:
                red_amp = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                for par in result.parameter_labels:
                    if pulsar+"_red_noise_gamma" in par:
                        red_gamma = result.posterior.iloc[result.posterior.log_likelihood.idxmax()][par]
                
                print("writing red noise")
                openpar.write("TNRedAmp {} \n".format(red_amp))
                openpar.write("TNRedGam {} \n".format(red_gamma))
                openpar.write("TNRedC {} \n".format(30))

        openpar.seek(0)
    except:
        print(pulsar+" isn't there right now")
        



        


        

