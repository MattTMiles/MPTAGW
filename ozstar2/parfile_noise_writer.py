import numpy
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")

parser.add_argument("-parfile", dest="parfile")
parser.add_argument("-noisefile", dest="noisefile")
args = parser.parse_args()

parfile = args.parfile
noisefile = args.noisefile

dest_parfile = parfile.rstrip(".par")+"_noise.par"
psrname = parfile.rstrip("_tdb.par")

noise_json = json.load(open(noisefile))
noise_keys = list(noise_json.keys())

psr_model = [ noise_model for noise_model in noise_keys if psrname in noise_model ]
#psr_noise = noise_json[psr_model]

os.system("cp "+parfile+" "+dest_parfile)

openpar = open(dest_parfile, "a")


for n in psr_model:
    #WN properties
    if "efac" in n:
        openpar.write("EFAC -f KAT_MKBF {} \n".format(noise_json[n]))
    if "equad" in n:
        openpar.write("TNEQ -f KAT_MKBF {} \n".format(noise_json[n]))
    if "ecorr_all" in n:
        openpar.write("TNECORR -f KAT_MKBF {} \n".format(noise_json[n]))
    if "ecorr_low" in n:
        openpar.write("TNECORRLOW -f KAT_MKBF {} \n".format(noise_json[n]))
    if "ecorr_high" in n:
        openpar.write("TNECORRHIGH -f KAT_MKBF {} \n".format(noise_json[n]))

    #red noise properties
    if "earth" in n:
        openpar.write("SWNEARTH {} \n".format(noise_json[n]))
        if psrname+"_gp_sw_gamma" not in psr_model:
            openpar.write("SWGAM {} \n".format(1))
            openpar.write("SWAMP {} \n".format(-20))      
            openpar.write("SWC {} \n".format(120))
    if "gp_sw_gamma" in n:
        openpar.write("SWGAM {} \n".format(noise_json[n]))
    if "gp_sw_log10_A" in n:
        openpar.write("SWAMP {} \n".format(noise_json[n]))
        openpar.write("SWC {} \n".format(120))
    
    if "dm_gp_gamma" in n:
        openpar.write("DMGam {} \n".format(noise_json[n]))
    if "dm_gp_log10_A" in n:
        openpar.write("DMAmp {} \n".format(noise_json[n]))
        openpar.write("DMC {} \n".format(120))
    
    if "chrom_gp_gamma" in n or "chrom_wide_gp_gamma" in n or "chromcidx_gp_gamma" in n:
        openpar.write("ChromGam {} \n".format(noise_json[n]))
    if "chrom_gp_log10_A" in n or "chrom_wide_gp_log10_A" in n or "chromcidx_gp_log10_A" in n:
        openpar.write("ChromAmp {} \n".format(noise_json[n]))
        openpar.write("ChromC {} \n".format(120))
        if psrname+"_chrom_gp_idx" not in psr_model and psrname+"_chrom_wide_gp_idx" not in psr_model:
            openpar.write("ChromIdx {} \n".format(4))
    if "chrom_gp_idx" in n or "chrom_wide_gp_idx" in n:
        openpar.write("ChromIdx {} \n".format(noise_json[n]))

    if "red_noise_gamma" in n:
        openpar.write("RedGam {} \n".format(noise_json[n]))
    if "red_noise_log10_A" in n:
        openpar.write("RedAmp {} \n".format(noise_json[n]))
        openpar.write("RedC {} \n".format(120))

    #deterministic properties
    if "chrom1yr_idx" in n:
        openpar.write("CHROMANNUALIDX {} \n".format(noise_json[n]))
    if "chrom1yr_log10_Amp" in n:
        openpar.write("CHROMANNUALAMP {} \n".format(noise_json[n]))
    if "chrom1yr_phase" in n:
        openpar.write("CHROMANNUALPHASE {} \n".format(noise_json[n]))
    
    if "chrom_bump_idx" in n:
        openpar.write("CHROMBUMPIDX {} \n".format(noise_json[n]))
    if "chrom_bump_log10_Amp" in n:
        openpar.write("CHROMBUMPAMP {} \n".format(noise_json[n]))
    if "chrom_bump_sigma" in n:
        openpar.write("CHROMBUMPSIGMA {} \n".format(noise_json[n]))
    if "chrom_bump_sign_param" in n:
        openpar.write("CHROMBUMPSIGN {} \n".format(noise_json[n]))                
    if "chrom_bump_t0" in n:
        openpar.write("CHROMBUMPT {} \n".format(noise_json[n]))        

openpar.write("TNGWAMP {} \n".format(-14.515))
openpar.write("TNGWGAM {} \n".format(4.333))
openpar.write("TNGWC {} \n".format(30))
    
