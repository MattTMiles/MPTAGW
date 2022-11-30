import bilby
import os
import glob
import json
import sys
import numpy as np


pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc"

all_evidence= {}

chosen_evidence = {}

for pulsar in open(pulsar_list,"r").readlines():
    os.chdir(enterprise_dir)
    pulsar = pulsar.strip("\n")

    psr_ev_dict = {}

    psrdirs = glob.glob("*"+pulsar+"*")
    temp_bin = {}

    full = pulsar+"_white_noise"
    full_no_ecorr = pulsar+"_white_noise_no_ecorr"
    bm_full = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+full+"/*json")[0]).log_evidence
    bm_no_ecorr = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+full_no_ecorr+"/*json")[0]).log_evidence
    
    choice = {}
    
    if bm_full > bm_no_ecorr + 4:
        choice[full] = bm_full
    else:
        choice[full_no_ecorr] = bm_no_ecorr
    
    dm = pulsar+"_dm"
    red = pulsar+"_red"
    
    try:
        dm_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+dm+"/*json")[0]).log_evidence
    except:
        dm_ev = 0
    
    try:
        red_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+red+"/*json")[0]).log_evidence
    except:
        red_ev = 0
    

    if dm_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[dm] = dm_ev

        if red_ev > dm_ev:
            choice.popitem()
            choice[red] = red_ev
    
    elif red_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[red] = red_ev
    
    chrom_dm = pulsar+"_chrom_dm"
    chrom_red = pulsar+"_chrom_red"
    dm_red = pulsar+"_dm_red_nlive1000"

    try:
        chrom_dm_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+chrom_dm+"/*json")[0]).log_evidence
    except:
        chrom_dm_ev = 0

    try:
        chrom_red_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+chrom_red+"/*json")[0]).log_evidence
    except:
        chrom_red_ev = 0

    try:
        dm_red_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+dm_red+"/*json")[0]).log_evidence
    except:
        dm_red_ev = 0

    if chrom_dm_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[chrom_dm] = chrom_dm_ev

        if chrom_red_ev > chrom_dm_ev:
            choice.popitem()
            choice[chrom_red] = chrom_red_ev

            if dm_red_ev > chrom_red_ev:
                choice.popitem()
                choice[dm_red] = dm_red
        
        elif dm_red_ev > chrom_dm_ev:
            choice.popitem()
            choice[dm_red] = dm_red_ev

    elif chrom_red_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[chrom_red] = chrom_red_ev

        if dm_red_ev > chrom_red_ev:
            choice.popitem()
            choice[dm_red] = dm_red_ev

    elif dm_red_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[dm_red] = dm_red_ev

    chrom_dm_red = pulsar+"chrom_red_dm"
    try:
        chrom_dm_red_ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+chrom_dm_red+"/*json")[0]).log_evidence
    except:
        chrom_dm_red_ev = 0

    if chrom_dm_red_ev > list(choice.values())[0] + 4:
        choice.popitem()
        choice[chrom_dm_red] = chrom_dm_red_ev
        

    for psrdir in psrdirs:
        try:
            # Try to include into the list in a smart way
            ev = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+psrdir+"/*json")[0]).log_evidence
            temp_bin[psrdir] = ev
            #if len(psr_ev_dict) > 0:
            #    first = list(psr_ev_dict.values())[0]
                # If the new evidence is greater than the threshold it'll remove the winning value and put this in place
            #    if ev > first + 4:
                    # If it beats it this pops the current value out
            #        (k := next(iter(psr_ev_dict)), psr_ev_dict.pop(k))
            #        psr_ev_dict[psrdir] = ev
            #else:
            #    psr_ev_dict[psrdir] = ev
        except:
            continue
    
    # For posterity print out all the evidences of each pulsar
    for key in temp_bin:
        print(key, ":", temp_bin[key])

    all_evidence.update(temp_bin)
    
    # Need to also check this against the basic models.
    #full = pulsar+"_white_noise"
    #full_no_ecorr = pulsar+"_white_noise_no_ecorr"
    #bm_full = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+full+"/*json")[0]).log_evidence
    #bm_no_ecorr = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+full_no_ecorr+"/*json")[0]).log_evidence
    
    #choice = {}
    
    #if bm_full > bm_no_ecorr + 4:
    #    choice[full] = bm_full
    #else:
    #    choice[full_no_ecorr] = bm_no_ecorr

    #if list(choice.values())[0] > list(psr_ev_dict.values())[0]:
    #    chosen = choice
    #else:
    #    chosen = psr_ev_dict

    # Then add chosen to the overall dictionary
    chosen_evidence.update(choice)
    print("The winner is: \n{}".format(choice))










