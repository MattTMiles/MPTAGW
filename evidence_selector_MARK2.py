import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse

# Run with:
# python evidence_selector_MARK2.py -pulsars {pulsar_list.txt} -outfile {path_to_file.json}

parser = argparse.ArgumentParser(description="Evidence comparer and decider.")
parser.add_argument("-pulsars", dest="pulsars", nargs="+", help="List of pulsars to do evidence comparison on.",required = True)
parser.add_argument("-outfile", dest="outfile", help="Path and name of the outfile (.json extension)",required = True)
parser.add_argument("-alt_dir", dest="alt_dir", help="Provide an alternate directory path to check the evidence in. Make sure that all the information is kept in here otherwise it just won't work.", required = False)
args = parser.parse_args()

pulsar_list = args.pulsars
outfile = args.outfile
alt_dir = str(args.alt_dir)

#pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list.txt"
top_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200"
#live_200_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200"
ecorr_maybe = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/extra_ecorr"

# This is a list of pulsars that are identified as having a wider preferrred spectral or chromatic indices
#wide_checks = ["J1012-4235","J1125-6014","J1435-6100","J1705-1903","J1708-3506","J1730-2304","J1747-4036","J1802-2124","J1911-1114","J2145-0750","J2234+0944"]
wide_checks = []
# This is a list of pulsars that probably should be favoured for ECORR
#ecorr_checks = ["J1231-1411","J1327-0755","J1455-3330","J1514-4946","J1804-2717","J2124-3358"]
ecorr_checks = []
# This is a list of pulsars that have been reassessed in the live_200 directory
#redo_200 = ["J1614-2230"]
redo_200 = []
# This is a list of pulsars that have been reassessed in the live_200 directory including the WN parameters
#redo_200_wn = ["J1614-2230"]
redo_200_wn = []
all_evidence= {}

chosen_evidence = {}

bf_cutoff = 2.3

os.chdir(top_dir)
try:
    os.remove(outfile)
except:
    FileNotFoundError()

if os.path.exists(pulsar_list[0]):
    psrlist = open(pulsar_list[0]).readlines()
else:
    psrlist = list(pulsar_list)

if alt_dir == "None" or alt_dir == "":

    for pulsar in psrlist:
        os.chdir(enterprise_dir)
        pulsar = pulsar.strip("\n")
        print(pulsar)
        psr_ev_dict = {}

        psrdirs = glob.glob("*"+pulsar+"*")

        #Evaluate the top tier

        try:
            res_RN_DM_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROM")
            res_RN_DM_BL_BH_CHROM = 0

        try:
            res_RN_DM_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROM_WIDE")
            res_RN_DM_BL_BH_CHROM_WIDE = 0

        if pulsar in ecorr_checks:
            try:
                res_RN_DM_BL_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROM_ecorr")
                res_RN_DM_BL_BH_CHROM_ecorr = 0

        T_top = {}
        #T_top[pulsar+"_RN_DM_BL_BH_CHROM"] = res_RN_DM_BL_BH_CHROM
        #T_top[pulsar+"_RN_DM_BL_BH_CHROM_WIDE"] = res_RN_DM_BL_BH_CHROM_WIDE

        T_top[pulsar+"_RN_DM_BL_BH_CHROM"] = 0
        T_top[pulsar+"_RN_DM_BL_BH_CHROM_WIDE"] = 0
        
        if pulsar in ecorr_checks:
            T_top[pulsar+"_RN_DM_BL_BH_CHROM_ecorr"] = res_RN_DM_BL_BH_CHROM_ecorr
        
        T_top_sort = dict(sorted(T_top.items(), key=lambda item: item[1],reverse=True))


        #Initialise the chosen directory    
        choice = {}
        choice[list(T_top_sort.keys())[0]] = list(T_top_sort.values())[0]


        #Evaluate Tier 5
        ev_V = {}
        
        try:
            res_RN_DM_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
         
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROMCIDX")
            res_RN_DM_BL_BH_CHROMCIDX = 0


        ev_V[pulsar+"_RN_DM_BL_BH_CHROMCIDX"] = res_RN_DM_BL_BH_CHROMCIDX
        

        if pulsar in ecorr_checks:
            try:
                res_RN_DM_BL_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROMCIDX_ecorr")
                res_RN_DM_BL_BH_CHROMCIDX_ecorr = 0

            ev_V[pulsar+"_RN_DM_BL_BH_CHROMCIDX_ecorr"] = res_RN_DM_BL_BH_CHROMCIDX_ecorr
            ev_V[pulsar+"_RN_DM_BL_BH_CHROM_ecorr"] = res_RN_DM_BL_BH_CHROM_ecorr

        T_five = {}
        #T_five[pulsar+"_RN_DM_BL_BH_CHROMCIDX"] = res_RN_DM_BL_BH_CHROMCIDX
        T_five[pulsar+"_RN_DM_BL_BH_CHROMCIDX"] = 0
        
        if pulsar in ecorr_checks:
            T_five[pulsar+"_RN_DM_BL_BH_CHROMCIDX_ecorr"] = res_RN_DM_BL_BH_CHROMCIDX_ecorr
        
        T_five_sort = dict(sorted(T_five.items(), key=lambda item: item[1],reverse=True))

        if list(T_five_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_five_sort.keys())[0]] = list(T_five_sort.values())[0]


        # Go through Tier 4 and choose the best
        try:
            res_RN_DM_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH")
            res_RN_DM_BL_BH = 0
        if pulsar not in redo_200:
            try: 
                res_RN_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM")
                res_RN_BL_BH_CHROM = 0
            try: 
                res_RN_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM_WIDE")
                res_RN_BL_BH_CHROM_WIDE = 0
            try:
                res_RN_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROMCIDX")
                res_RN_BL_BH_CHROMCIDX = 0
        else:
            try: 
                res_RN_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM")
                res_RN_BL_BH_CHROM = 0
            try:
                res_RN_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROMCIDX")
                res_RN_BL_BH_CHROMCIDX = 0
        try:
            res_DM_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROM")
            res_DM_BL_BH_CHROM = 0
        try:
            res_DM_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROM_WIDE")
            res_DM_BL_BH_CHROM_WIDE = 0
        try:
            res_RN_DM_BL_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROM")
            res_RN_DM_BL_CHROM = 0
        try:
            res_RN_DM_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROM_WIDE")
            res_RN_DM_BL_CHROM_WIDE = 0
        try:
            res_RN_DM_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROM")
            res_RN_DM_BH_CHROM = 0
        try:
            res_RN_DM_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROM_WIDE")
            res_RN_DM_BH_CHROM_WIDE = 0
        try:
            res_DM_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROMCIDX")
            res_DM_BL_BH_CHROMCIDX = 0
        try:        
            res_RN_DM_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROMCIDX")
            res_RN_DM_BL_CHROMCIDX = 0
        try:    
            res_RN_DM_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROMCIDX")
            res_RN_DM_BH_CHROMCIDX = 0
        
        if pulsar in ecorr_checks:
            try:
                res_RN_DM_BL_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_ecorr")
                res_RN_DM_BL_BH_ecorr = 0
            try: 
                res_RN_BL_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM_ecorr")
                res_RN_BL_BH_CHROM_ecorr = 0
            try:
                res_DM_BL_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROM_ecorr")
                res_DM_BL_BH_CHROM_ecorr = 0
            try:
                res_RN_DM_BL_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROM_ecorr")
                res_RN_DM_BL_CHROM_ecorr = 0
            try:
                res_RN_DM_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROM_ecorr")
                res_RN_DM_BH_CHROM_ecorr = 0
            try:
                res_RN_BL_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROMCIDX_ecorr")
                res_RN_BL_BH_CHROMCIDX_ecorr = 0
            try:
                res_DM_BL_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROMCIDX_ecorr")
                res_DM_BL_BH_CHROMCIDX_ecorr = 0
            try:        
                res_RN_DM_BL_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROMCIDX_ecorr")
                res_RN_DM_BL_CHROMCIDX_ecorr = 0
            try:    
                res_RN_DM_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROMCIDX_ecorr")
                res_RN_DM_BH_CHROMCIDX_ecorr = 0

        T_four = {}
        # Putting in half measures to account for the extra term that chromatic noise has which arbitrarily increases the likelihood
        T_four_half = {}
        '''
        T_four[pulsar+"_RN_DM_BL_BH"] = res_RN_DM_BL_BH
        T_four_half[pulsar+"_RN_BL_BH_CHROM"] = res_RN_BL_BH_CHROM
        T_four_half[pulsar+"_DM_BL_BH_CHROM"] = res_DM_BL_BH_CHROM
        T_four_half[pulsar+"_RN_DM_BL_CHROM"] = res_RN_DM_BL_CHROM
        T_four_half[pulsar+"_RN_DM_BH_CHROM"] = res_RN_DM_BH_CHROM
        T_four_half[pulsar+"_RN_BL_BH_CHROM_WIDE"] = res_RN_BL_BH_CHROM_WIDE
        T_four_half[pulsar+"_DM_BL_BH_CHROM_WIDE"] = res_DM_BL_BH_CHROM_WIDE
        T_four_half[pulsar+"_RN_DM_BL_CHROM_WIDE"] = res_RN_DM_BL_CHROM_WIDE
        T_four_half[pulsar+"_RN_DM_BH_CHROM_WIDE"] = res_RN_DM_BH_CHROM_WIDE
        T_four[pulsar+"_RN_BL_BH_CHROMCIDX"] = res_RN_BL_BH_CHROMCIDX
        T_four[pulsar+"_DM_BL_BH_CHROMCIDX"] = res_DM_BL_BH_CHROMCIDX
        T_four[pulsar+"_RN_DM_BL_CHROMCIDX"] = res_RN_DM_BL_CHROMCIDX
        T_four[pulsar+"_RN_DM_BH_CHROMCIDX"] = res_RN_DM_BH_CHROMCIDX
        '''
        T_four[pulsar+"_RN_DM_BL_BH"] = 0
        T_four_half[pulsar+"_RN_BL_BH_CHROM"] = 0
        T_four_half[pulsar+"_DM_BL_BH_CHROM"] = 0
        T_four_half[pulsar+"_RN_DM_BL_CHROM"] = 0
        T_four_half[pulsar+"_RN_DM_BH_CHROM"] = 0
        T_four_half[pulsar+"_RN_BL_BH_CHROM_WIDE"] = 0
        T_four_half[pulsar+"_DM_BL_BH_CHROM_WIDE"] = 0
        T_four_half[pulsar+"_RN_DM_BL_CHROM_WIDE"] = 0
        T_four_half[pulsar+"_RN_DM_BH_CHROM_WIDE"] = 0
        T_four[pulsar+"_RN_BL_BH_CHROMCIDX"] = 0
        T_four[pulsar+"_DM_BL_BH_CHROMCIDX"] = 0
        T_four[pulsar+"_RN_DM_BL_CHROMCIDX"] = 0
        T_four[pulsar+"_RN_DM_BH_CHROMCIDX"] = 0

        if pulsar in ecorr_checks:
            T_four[pulsar+"_RN_DM_BL_BH_ecorr"] = res_RN_DM_BL_BH_ecorr
            T_four_half[pulsar+"_RN_BL_BH_CHROM_ecorr"] = res_RN_BL_BH_CHROM_ecorr
            T_four_half[pulsar+"_DM_BL_BH_CHROM_ecorr"] = res_DM_BL_BH_CHROM_ecorr
            T_four_half[pulsar+"_RN_DM_BL_CHROM_ecorr"] = res_RN_DM_BL_CHROM_ecorr
            T_four_half[pulsar+"_RN_DM_BH_CHROM_ecorr"] = res_RN_DM_BH_CHROM_ecorr
            T_four[pulsar+"_RN_BL_BH_CHROMCIDX_ecorr"] = res_RN_BL_BH_CHROMCIDX_ecorr
            T_four[pulsar+"_DM_BL_BH_CHROMCIDX_ecorr"] = res_DM_BL_BH_CHROMCIDX_ecorr
            T_four[pulsar+"_RN_DM_BL_CHROMCIDX_ecorr"] = res_RN_DM_BL_CHROMCIDX_ecorr
            T_four[pulsar+"_RN_DM_BH_CHROMCIDX_ecorr"] = res_RN_BL_BH_CHROM_ecorr

        T_four_half_sort = dict(sorted(T_four_half.items(), key=lambda item: item[1],reverse=True))
        T_four_sort = dict(sorted(T_four.items(), key=lambda item: item[1],reverse=True))
        
        # Select out condition to maintain probable model
        if list(T_four_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_four_half_sort.keys())[0]] = list(T_four_half_sort.values())[0]

        if list(T_four_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_four_sort.keys())[0]] = list(T_four_sort.values())[0]
        
        # Go through Tier 3 and choose the best
        try:
            res_RN_DM_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROM")
            res_RN_DM_CHROM = 0
        try:
            res_RN_DM_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROM_WIDE")
            res_RN_DM_CHROM_WIDE = 0
        if pulsar not in redo_200:
            try:
                res_RN_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH")
                res_RN_BL_BH = 0
            try:
                res_RN_BL_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM")
                res_RN_BL_CHROM = 0
            try:
                res_RN_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM_WIDE")
                res_RN_BL_CHROM_WIDE = 0
            try:
                res_RN_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM")
                res_RN_BH_CHROM = 0
            try:
                res_RN_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BH_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM_WIDE")
                res_RN_BH_CHROM_WIDE = 0
            try:
                res_RN_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROMCIDX")
                res_RN_BL_CHROMCIDX = 0
            try:
                res_RN_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROMCIDX")
                res_RN_BH_CHROMCIDX = 0
            try:
                res_CHROM_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL_BH")
                res_CHROM_BL_BH = 0
            try:
                res_CHROM_WIDE_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_WIDE_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BL_BH")
                res_CHROM_WIDE_BL_BH = 0
            try:
                res_CHROMCIDX_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROMCIDX_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL_BH")
                res_CHROMCIDX_BL_BH = 0
        else:
            try:
                res_RN_BL_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"/"+pulsar+"_RN_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH")
                res_RN_BL_BH = 0
            try:
                res_RN_BL_CHROM = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BL_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM")
                res_RN_BL_CHROM = 0
            try:
                res_RN_BH_CHROM = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM")
                res_RN_BH_CHROM = 0
            try:
                res_RN_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BL_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROMCIDX")
                res_RN_BL_CHROMCIDX = 0
            try:
                res_RN_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROMCIDX")
                res_RN_BH_CHROMCIDX = 0
            try:
                res_CHROM_BL_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROM_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL_BH")
                res_CHROM_BL_BH = 0
            try:
                res_CHROMCIDX_BL_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROMCIDX_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL_BH")
                res_CHROMCIDX_BL_BH = 0
        try:   
            res_DM_BL_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH")
            res_DM_BL_BH = 0
        try:
            res_RN_DM_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL")
            res_RN_DM_BL = 0
        try:
            res_RN_DM_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH")
            res_RN_DM_BH = 0
        try:
            res_DM_BL_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROM")
            res_DM_BL_CHROM = 0
        try:
            res_DM_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROM_WIDE")
            res_DM_BL_CHROM_WIDE = 0
        try:
            res_DM_BH_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROM")
            res_DM_BH_CHROM = 0
        try:
            res_DM_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROM_WIDE")
            res_DM_BH_CHROM_WIDE = 0
        try:
            res_RN_DM_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROMCIDX")
            res_RN_DM_CHROMCIDX = 0
        try:
            res_DM_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROMCIDX")
            res_DM_BL_CHROMCIDX = 0
        try:
            res_DM_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROMCIDX")
            res_DM_BH_CHROMCIDX = 0
        
        if pulsar in ecorr_checks:
            try:
                res_RN_DM_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROM_ecorr")
                res_RN_DM_CHROM_ecorr = 0
            try:
                res_RN_BL_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_ecorr")
                res_RN_BL_BH_ecorr = 0
            try:   
                res_DM_BL_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_ecorr")
                res_DM_BL_BH_ecorr = 0
            try:
                res_RN_DM_BL_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_ecorr")
                res_RN_DM_BL_ecorr = 0
            try:
                res_RN_DM_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_ecorr")
                res_RN_DM_BH_ecorr = 0
            try:
                res_RN_BL_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM_ecorr")
                res_RN_BL_CHROM_ecorr = 0
            try:
                res_RN_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM_ecorr")
                res_RN_BH_CHROM_ecorr = 0
            try:
                res_DM_BL_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROM_ecorr")
                res_DM_BL_CHROM_ecorr = 0
            try:
                res_DM_BH_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BH_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROM_ecorr")
                res_DM_BH_CHROM_ecorr = 0
            try:
                res_RN_DM_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROMCIDX_ecorr")
                res_RN_DM_CHROMCIDX_ecorr = 0
            try:
                res_RN_BL_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROMCIDX_ecorr")
                res_RN_BL_CHROMCIDX_ecorr = 0
            try:
                res_RN_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROMCIDX_ecorr")
                res_RN_BH_CHROMCIDX_ecorr = 0
            try:
                res_DM_BL_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROMCIDX_ecorr")
                res_DM_BL_CHROMCIDX_ecorr = 0
            try:
                res_DM_BH_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BH_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROMCIDX_ecorr")
                res_DM_BH_CHROMCIDX_ecorr = 0
            try:
                res_CHROM_BL_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROM_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL_BH_ecorr")
                res_CHROM_BL_BH_ecorr = 0
            try:
                res_CHROMCIDX_BL_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROMCIDX_BL_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL_BH_ecorr")
                res_CHROMCIDX_BL_BH_ecorr = 0


        T_three = {}
        T_three_half = {}
        T_three_half[pulsar+"_RN_DM_CHROM"] = res_RN_DM_CHROM
        T_three_half[pulsar+"_RN_DM_CHROM_WIDE"] = res_RN_DM_CHROM_WIDE
        '''
        T_three[pulsar+"_RN_BL_BH"] = res_RN_BL_BH
        T_three[pulsar+"_DM_BL_BH"] = res_DM_BL_BH
        T_three[pulsar+"_RN_DM_BL"] = res_RN_DM_BL
        T_three[pulsar+"_RN_DM_BH"] = res_RN_DM_BH
        T_three_half[pulsar+"_RN_BL_CHROM"] = res_RN_BL_CHROM
        T_three_half[pulsar+"_RN_BH_CHROM"] = res_RN_BH_CHROM
        T_three_half[pulsar+"_DM_BL_CHROM"] = res_DM_BL_CHROM
        T_three_half[pulsar+"_DM_BH_CHROM"] = res_DM_BH_CHROM
        T_three_half[pulsar+"_RN_BL_CHROM_WIDE"] = res_RN_BL_CHROM_WIDE
        T_three_half[pulsar+"_RN_BH_CHROM_WIDE"] = res_RN_BH_CHROM_WIDE
        T_three_half[pulsar+"_DM_BL_CHROM_WIDE"] = res_DM_BL_CHROM_WIDE
        T_three_half[pulsar+"_DM_BH_CHROM_WIDE"] = res_DM_BH_CHROM_WIDE
        '''
        T_three[pulsar+"_RN_BL_BH"] = 0
        T_three[pulsar+"_DM_BL_BH"] = 0
        T_three[pulsar+"_RN_DM_BL"] = 0
        T_three[pulsar+"_RN_DM_BH"] = 0
        T_three_half[pulsar+"_RN_BL_CHROM"] = 0
        T_three_half[pulsar+"_RN_BH_CHROM"] = 0
        T_three_half[pulsar+"_DM_BL_CHROM"] = 0
        T_three_half[pulsar+"_DM_BH_CHROM"] = 0
        T_three_half[pulsar+"_RN_BL_CHROM_WIDE"] = 0
        T_three_half[pulsar+"_RN_BH_CHROM_WIDE"] = 0
        T_three_half[pulsar+"_DM_BL_CHROM_WIDE"] = 0
        T_three_half[pulsar+"_DM_BH_CHROM_WIDE"] = 0

        T_three[pulsar+"_RN_DM_CHROMCIDX"] = res_RN_DM_CHROMCIDX
        '''
        T_three[pulsar+"_RN_BL_CHROMCIDX"] = res_RN_BL_CHROMCIDX
        T_three[pulsar+"_RN_BH_CHROMCIDX"] = res_RN_BH_CHROMCIDX
        T_three[pulsar+"_DM_BL_CHROMCIDX"] = res_DM_BL_CHROMCIDX
        T_three[pulsar+"_DM_BH_CHROMCIDX"] = res_DM_BH_CHROMCIDX
        T_three_half[pulsar+"_CHROM_BL_BH"] = res_CHROM_BL_BH
        T_three_half[pulsar+"_CHROM_WIDE_BL_BH"] = res_CHROM_WIDE_BL_BH
        T_three[pulsar+"_CHROMCIDX_BL_BH"] = res_CHROMCIDX_BL_BH
        '''
        T_three[pulsar+"_RN_BL_CHROMCIDX"] = 0
        T_three[pulsar+"_RN_BH_CHROMCIDX"] = 0
        T_three[pulsar+"_DM_BL_CHROMCIDX"] = 0
        T_three[pulsar+"_DM_BH_CHROMCIDX"] = 0
        T_three_half[pulsar+"_CHROM_BL_BH"] = 0
        T_three_half[pulsar+"_CHROM_WIDE_BL_BH"] = 0
        T_three[pulsar+"_CHROMCIDX_BL_BH"] = 0

        if pulsar in ecorr_checks:
            T_three_half[pulsar+"_RN_DM_CHROM_ecorr"] = res_RN_DM_CHROM_ecorr
            T_three[pulsar+"_RN_BL_BH_ecorr"] = res_RN_BL_BH_ecorr
            T_three[pulsar+"_DM_BL_BH_ecorr"] = res_DM_BL_BH_ecorr
            T_three[pulsar+"_RN_DM_BL_ecorr"] = res_RN_DM_BL_ecorr
            T_three[pulsar+"_RN_DM_BH_ecorr"] = res_RN_DM_BH_ecorr
            T_three_half[pulsar+"_RN_BL_CHROM_ecorr"] = res_RN_BL_CHROM_ecorr
            T_three_half[pulsar+"_RN_BH_CHROM_ecorr"] = res_RN_BH_CHROM_ecorr
            T_three_half[pulsar+"_DM_BL_CHROM_ecorr"] = res_DM_BL_CHROM_ecorr
            T_three_half[pulsar+"_DM_BH_CHROM_ecorr"] = res_DM_BH_CHROM_ecorr
            T_three[pulsar+"_RN_DM_CHROMCIDX_ecorr"] = res_RN_DM_CHROMCIDX_ecorr
            T_three[pulsar+"_RN_BL_CHROMCIDX_ecorr"] = res_RN_BL_CHROMCIDX_ecorr
            T_three[pulsar+"_RN_BH_CHROMCIDX_ecorr"] = res_RN_BH_CHROMCIDX_ecorr
            T_three[pulsar+"_DM_BL_CHROMCIDX_ecorr"] = res_DM_BL_CHROMCIDX_ecorr
            T_three[pulsar+"_DM_BH_CHROMCIDX_ecorr"] = res_DM_BH_CHROMCIDX_ecorr
            T_three_half[pulsar+"_CHROM_BL_BH_ecorr"] = res_CHROM_BL_BH_ecorr
            T_three[pulsar+"_CHROMCIDX_BL_BH_ecorr"] = res_CHROMCIDX_BL_BH_ecorr

        T_three_half_sort = dict(sorted(T_three_half.items(), key=lambda item: item[1],reverse=True))
        T_three_sort = dict(sorted(T_three.items(), key=lambda item: item[1],reverse=True))

        if list(T_three_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_three_half_sort.keys())[0]] = list(T_three_half_sort.values())[0]

        if list(T_three_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_three_sort.keys())[0]] = list(T_three_sort.values())[0]

        # Same for Tier 2

        try:
            res_RN_DM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_DM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM")
            res_RN_DM = 0
        if pulsar not in redo_200:
            try:
                res_RN_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROM")
                res_RN_CHROM = 0
            try:
                res_RN_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROM_WIDE")
                res_RN_CHROM_WIDE = 0
            try:
                res_RN_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL")
                res_RN_BL = 0  
            try:
                res_RN_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH")
                res_RN_BH = 0
            try:
                res_RN_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROMCIDX")
                res_RN_CHROMCIDX = 0 
            try:
                res_CHROM_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL")
                res_CHROM_BL = 0 
            try:
                res_CHROM_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BH")
                res_CHROM_BH = 0 
            try:
                res_CHROM_WIDE_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_WIDE_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BL")
                res_CHROM_WIDE_BL = 0 
            try:
                res_CHROM_WIDE_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_WIDE_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BH")
                res_CHROM_WIDE_BH = 0 
            try:
                res_CHROMCIDX_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROMCIDX_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL")
                res_CHROMCIDX_BL = 0 
            try:
                res_CHROMCIDX_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROMCIDX_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BH")
                res_CHROMCIDX_BH = 0 
        else:
            try:
                res_RN_CHROM = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROM")
                res_RN_CHROM = 0
            try:
                res_RN_BL = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL")
                res_RN_BL = 0  
            try:
                res_RN_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH")
                res_RN_BH = 0 
            try:
                res_RN_CHROMCIDX = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROMCIDX")
                res_RN_CHROMCIDX = 0 
            try:
                res_CHROM_BL = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROM_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL")
                res_CHROM_BL = 0 
            try:
                res_CHROM_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROM_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BH")
                res_CHROM_BH = 0 
            try:
                res_CHROMCIDX_BL = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROMCIDX_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL")
                res_CHROMCIDX_BL = 0 
            try:
                res_CHROMCIDX_BH = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROMCIDX_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BH")
                res_CHROMCIDX_BH = 0 
        try:
            res_DM_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROM")
            res_DM_CHROM = 0  
        try:
            res_DM_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROM_WIDE")
            res_DM_CHROM_WIDE = 0  
        try:
            res_DM_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL")
            res_DM_BL = 0 
        try:
            res_DM_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH")
            res_DM_BH = 0 
        try:
            res_DM_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROMCIDX")
            res_DM_CHROMCIDX = 0 
        try:
            res_BH_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_BH_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BH_BL")
            res_BH_BL = 0
        
        if pulsar in ecorr_checks:
            try:
                res_RN_DM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_DM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_DM_ecorr")
                res_RN_DM_ecorr = 0
            try:
                res_RN_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROM_ecorr")
                res_RN_CHROM_ecorr = 0
            try:
                res_DM_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_CHROM_ecorr")
                res_DM_CHROM_ecorr = 0  
            try:
                res_RN_BL_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BL_ecorr")
                res_RN_BL_ecorr = 0  
            try:
                res_RN_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_ecorr")
                res_RN_BH_ecorr = 0 
            try:
                res_DM_BL_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BL_ecorr")
                res_DM_BL_ecorr = 0 
            try:
                res_DM_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_BH_ecorr")
                res_DM_BH_ecorr = 0 
            try:
                res_RN_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_CHROMCIDX_ecorr")
                res_RN_CHROMCIDX_ecorr = 0 
            try:
                res_DM_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_CHROMCIDX_ecorr")
                res_DM_CHROMCIDX_ecorr = 0 
            try:
                res_CHROM_BL_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROM_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BL_ecorr")
                res_CHROM_BL_ecorr = 0 
            try:
                res_CHROM_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROM_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_BH_ecorr")
                res_CHROM_BH_ecorr = 0 
            try:
                res_CHROMCIDX_BL_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROMCIDX_BL/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL_ecorr")
                res_CHROMCIDX_BL_ecorr = 0 
            try:
                res_CHROMCIDX_BH_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROMCIDX_BH/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BH_ecorr")
                res_CHROMCIDX_BH_ecorr = 0 

        if pulsar in wide_checks:
            try:
                res_DM_WIDE_CHROM_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_DM_WIDE_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_WIDE_CHROM_WIDE")
                res_DM_WIDE_CHROM_WIDE = 0
            try:
                res_DM_WIDE_CHROM = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_DM_WIDE_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_WIDE_CHROM")
                res_DM_WIDE_CHROM = 0
            try:
                res_DM_CHROM_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_DM_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_CHROM_WIDE")
                res_DM_CHROM_WIDE = 0
            try:
                res_RN_BH_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_BH_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_BH_WIDE")
                res_RN_BH_WIDE = 0
            

        T_two = {}
        T_two_half = {}
        T_two[pulsar+"_RN_DM"] = res_RN_DM
        T_two_half[pulsar+"_RN_CHROM"] = res_RN_CHROM
        T_two_half[pulsar+"_DM_CHROM"] = res_DM_CHROM
        T_two_half[pulsar+"_RN_CHROM_WIDE"] = res_RN_CHROM_WIDE
        T_two_half[pulsar+"_DM_CHROM_WIDE"] = res_DM_CHROM_WIDE
        '''
        T_two[pulsar+"_RN_BL"] = res_RN_BL
        T_two[pulsar+"_RN_BH"] = res_RN_BH
        T_two[pulsar+"_DM_BL"] = res_DM_BL
        T_two[pulsar+"_DM_BH"] = res_DM_BH
        T_two[pulsar+"_BH_BL"] = res_BH_BL
        '''
        T_two[pulsar+"_RN_BL"] = 0
        T_two[pulsar+"_RN_BH"] = 0
        T_two[pulsar+"_DM_BL"] = 0
        T_two[pulsar+"_DM_BH"] = 0
        T_two[pulsar+"_BH_BL"] = 0

        T_two[pulsar+"_RN_CHROMCIDX"] = res_RN_CHROMCIDX
        T_two[pulsar+"_DM_CHROMCIDX"] = res_DM_CHROMCIDX
        '''
        T_two_half[pulsar+"_CHROM_BL"] = res_CHROM_BL
        T_two_half[pulsar+"_CHROM_BH"] = res_CHROM_BH
        T_two_half[pulsar+"_CHROM_WIDE_BL"] = res_CHROM_WIDE_BL
        T_two_half[pulsar+"_CHROM_WIDE_BH"] = res_CHROM_WIDE_BH
        T_two[pulsar+"_CHROMCIDX_BL"] = res_CHROMCIDX_BL
        T_two[pulsar+"_CHROMCIDX_BH"] = res_CHROMCIDX_BH
        '''
        
        T_two_half[pulsar+"_CHROM_BL"] = 0
        T_two_half[pulsar+"_CHROM_BH"] = 0
        T_two_half[pulsar+"_CHROM_WIDE_BL"] = 0
        T_two_half[pulsar+"_CHROM_WIDE_BH"] = 0
        T_two[pulsar+"_CHROMCIDX_BL"] = 0
        T_two[pulsar+"_CHROMCIDX_BH"] = 0

        if pulsar in ecorr_checks:
            T_two[pulsar+"_RN_DM_ecorr"] = res_RN_DM_ecorr
            T_two_half[pulsar+"_RN_CHROM_ecorr"] = res_RN_CHROM_ecorr
            T_two_half[pulsar+"_DM_CHROM_ecorr"] = res_DM_CHROM_ecorr
            T_two[pulsar+"_RN_BL_ecorr"] = res_RN_BL_ecorr
            T_two[pulsar+"_RN_BH_ecorr"] = res_RN_BH_ecorr
            T_two[pulsar+"_DM_BL_ecorr"] = res_DM_BL_ecorr
            T_two[pulsar+"_DM_BH_ecorr"] = res_DM_BH_ecorr
            T_two[pulsar+"_RN_CHROMCIDX_ecorr"] = res_RN_CHROMCIDX_ecorr
            T_two[pulsar+"_DM_CHROMCIDX_ecorr"] = res_DM_CHROMCIDX_ecorr
            T_two_half[pulsar+"_CHROM_BL_ecorr"] = res_CHROM_BL_ecorr
            T_two_half[pulsar+"_CHROM_BH_ecorr"] = res_CHROM_BH_ecorr
            T_two[pulsar+"_CHROMCIDX_BL_ecorr"] = res_CHROMCIDX_BL_ecorr
            T_two[pulsar+"_CHROMCIDX_BH_ecorr"] = res_CHROMCIDX_BH_ecorr

        if pulsar in wide_checks:
            T_two_half[pulsar+"_DM_WIDE_CHROM_WIDE"] = res_DM_WIDE_CHROM_WIDE
            T_two_half[pulsar+"_DM_WIDE_CHROM"] = res_DM_WIDE_CHROM
            T_two_half[pulsar+"_DM_CHROM_WIDE"] = res_DM_CHROM_WIDE
            T_two[pulsar+"_RN_BH_WIDE"] = res_RN_BH_WIDE

        T_two_half_sort = dict(sorted(T_two_half.items(), key=lambda item: item[1],reverse=True))
        T_two_sort = dict(sorted(T_two.items(), key=lambda item: item[1],reverse=True))

        if list(T_two_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_two_half_sort.keys())[0]] = list(T_two_half_sort.values())[0]

        if list(T_two_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_two_sort.keys())[0]] = list(T_two_sort.values())[0]
        

        # Same for Tier 1
        if pulsar not in redo_200:
            try:
                res_RN = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_RN/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN")
                res_RN = 0
            try:
                res_CHROM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM")
                res_CHROM = 0
            try:
                res_CHROM_WIDE = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE")
                res_CHROM_WIDE = 0
            try:
                res_CHROMCIDX = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX")
                res_CHROMCIDX = 0
        else:
            try:
                res_RN = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN")
                res_RN = 0
            try:
                res_CHROM = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM")
                res_CHROM = 0
            try:
                res_CHROMCIDX = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX")
                res_CHROMCIDX = 0
        try:
            res_DM = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_DM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM")
            res_DM = 0
        try:
            res_BH = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BH")
            res_BH = 0
        try:
            res_BL = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BL")
            res_BL = 0


        if pulsar in ecorr_checks:
            try:
                res_RN_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_RN/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_ecorr")
                res_RN_ecorr = 0
            try:
                res_DM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_DM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_ecorr")
                res_DM_ecorr = 0
            try:
                res_CHROM_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROM/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_ecorr")
                res_CHROM_ecorr = 0
            try:
                res_CHROMCIDX_ecorr = bilby.result.read_in_result(glob.glob(ecorr_maybe+"/"+pulsar+"_CHROMCIDX/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_ecorr")
                res_CHROMCIDX_ecorr = 0

        if pulsar in wide_checks:
            try:
                res_CHROMCIDX_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROMCIDX_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_WIDE")
                res_CHROMCIDX_WIDE = 0
            try:
                res_RN_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_RN_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_RN_WIDE")
                res_RN_WIDE = 0
            try:
                res_CHROM_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_CHROM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE")
                res_CHROM_WIDE = 0
            try:
                res_DM_WIDE = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_DM_WIDE/*json")[0]).log_evidence
            except:
                print(pulsar+" does not have: "+pulsar+"_DM_WIDE")
                res_DM_WIDE = 0
            

        T_one = {}
        T_one_half = {}
        T_one[pulsar+"_RN"] = res_RN
        T_one[pulsar+"_DM"] = res_DM
        T_one_half[pulsar+"_CHROM"] = res_CHROM
        T_one_half[pulsar+"_CHROM_WIDE"] = res_CHROM_WIDE
        T_one[pulsar+"_CHROMCIDX"] = res_CHROMCIDX
        '''
        T_one[pulsar+"_BH"] = res_BH
        T_one[pulsar+"_BL"] = res_BL
        '''
        T_one[pulsar+"_BH"] = 0
        T_one[pulsar+"_BL"] = 0
        if pulsar in ecorr_checks:
            T_one[pulsar+"_RN_ecorr"] = res_RN_ecorr
            T_one[pulsar+"_DM_ecorr"] = res_DM_ecorr
            T_one_half[pulsar+"_CHROM_ecorr"] = res_CHROM_ecorr
            T_one[pulsar+"_CHROMCIDX_ecorr"] = res_CHROMCIDX_ecorr

        if pulsar in wide_checks:
            T_one[pulsar+"_CHROMCIDX_WIDE"] = res_CHROMCIDX_WIDE
            T_one[pulsar+"_RN_WIDE"] = res_RN_WIDE
            T_one_half[pulsar+"_CHROM_WIDE"] = res_CHROM_WIDE
            T_one[pulsar+"_DM_WIDE"] = res_DM_WIDE
        
        T_one_half_sort = dict(sorted(T_one_half.items(), key=lambda item: item[1],reverse=True))
        T_one_sort = dict(sorted(T_one.items(), key=lambda item: item[1],reverse=True))

        if list(T_one_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_one_half_sort.keys())[0]] = list(T_one_half_sort.values())[0]

        if list(T_one_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_one_sort.keys())[0]] = list(T_one_sort.values())[0]

        if pulsar not in redo_200_wn:

            try:
                res_WN = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN/*json")[0]).log_evidence
                ml_WN = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN/*json")[0]).log_likelihood_evaluations.max()
            except:
                print(pulsar+" does not have: "+pulsar+"_WN")
                res_WN = 0
            try:
                res_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_evidence
                ml_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+pulsar+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_likelihood_evaluations.max()
            except:
                print(pulsar+" does not have: "+pulsar+"_WN_NO_ECORR")
                res_WN_NO_ECORR = 0
        else:
            try:
                res_WN = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_WN/*json")[0]).log_evidence
                ml_WN = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_WN/*json")[0]).log_likelihood_evaluations.max()
            except:
                print(pulsar+" does not have: "+pulsar+"_WN")
                res_WN = 0
            try:
                res_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_evidence
                ml_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(live_200_dir+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_likelihood_evaluations.max()
            except:
                print(pulsar+" does not have: "+pulsar+"_WN_NO_ECORR")
                res_WN_NO_ECORR = 0
        '''
        if pulsar not in ecorr_checks:
            if res_WN > res_WN_NO_ECORR +2:
                if ml_WN > list(choice.values())[0]:
                    choice.popitem()
                    choice[pulsar+"_WN"] = res_WN
            else:
                if ml_WN_NO_ECORR > list(choice.values())[0]:
                    choice.popitem()
                    choice[pulsar+"_WN_NO_ECORR"] = res_WN_NO_ECORR
        
            chosen_evidence.update(choice)
        else:
            if ml_WN > list(choice.values())[0]:
                choice.popitem()
                choice[pulsar+"_WN"] = res_WN
        '''
        chosen_evidence.update(choice)

    os.chdir("/fred/oz002/users/mmiles/MPTA_GW/enterprise")

    with open(outfile,"a+") as outFile:
        json.dump(chosen_evidence,outFile,indent=4)

elif alt_dir is not None and alt_dir != "":
    for pulsar in psrlist:
        os.chdir(alt_dir)
        pulsar = pulsar.strip("\n")
        print(pulsar)
        psr_ev_dict = {}

        psrdirs = glob.glob("*"+pulsar+"*")

        #Evaluate Tier 5, the most complex for the best
        ev_V = {}
        
        try:
            res_RN_DM_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROMCIDX")
            res_RN_DM_BL_BH_CHROMCIDX = 0
        try:
            res_RN_DM_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROM")
            res_RN_DM_BL_BH_CHROM = 0

        try:
            res_RN_DM_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH_CHROM_WIDE")
            res_RN_DM_BL_BH_CHROM_WIDE = 0

        ev_V[pulsar+"_RN_DM_BL_BH_CHROMCIDX"] = res_RN_DM_BL_BH_CHROMCIDX
        ev_V[pulsar+"_RN_DM_BL_BH_CHROM"] = res_RN_DM_BL_BH_CHROM
        ev_V[pulsar+"_RN_DM_BL_BH_CHROM_WIDE"] = res_RN_DM_BL_BH_CHROM_WIDE

        T_top = {}
        # Accounting for the extra chromatic noise term
        T_top_half = {}
        T_top[pulsar+"_RN_DM_BL_BH_CHROMCIDX"] = res_RN_DM_BL_BH_CHROMCIDX
        T_top_half[pulsar+"_RN_DM_BL_BH_CHROM"] = res_RN_DM_BL_BH_CHROM
        T_top_half[pulsar+"_RN_DM_BL_BH_CHROM_WIDE"] = res_RN_DM_BL_BH_CHROM_WIDE
    
        T_top_half_sort = dict(sorted(T_top_half.items(), key=lambda item: item[1],reverse=True))
        T_top_sort = dict(sorted(T_top.items(), key=lambda item: item[1],reverse=True))

        #Initialise the top directory    
        choice = {}
        choice[list(T_top_half_sort.keys())[0]] = list(T_top_half_sort.values())[0]

        if list(T_top_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_top_sort.keys())[0]] = list(T_top_sort.values())[0]


        # Go through Tier 4 and choose the best
        try:
            res_RN_DM_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_BH")
            res_RN_DM_BL_BH = 0

        try: 
            res_RN_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM")
            res_RN_BL_BH_CHROM = 0
        try:
            res_RN_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROMCIDX")
            res_RN_BL_BH_CHROMCIDX = 0
        try:
            res_DM_BL_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROM")
            res_DM_BL_BH_CHROM = 0
        try:
            res_RN_DM_BL_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROM")
            res_RN_DM_BL_CHROM = 0
        try:
            res_RN_DM_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROM")
            res_RN_DM_BH_CHROM = 0
        try:
            res_DM_BL_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROMCIDX")
            res_DM_BL_BH_CHROMCIDX = 0
        try:        
            res_RN_DM_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROMCIDX")
            res_RN_DM_BL_CHROMCIDX = 0
        try:    
            res_RN_DM_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROMCIDX")
            res_RN_DM_BH_CHROMCIDX = 0
        try: 
            res_RN_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_BH_CHROM_WIDE")
            res_RN_BL_BH_CHROM_WIDE = 0
        try:
            res_DM_BL_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH_CHROM_WIDE")
            res_DM_BL_BH_CHROM_WIDE = 0
        try:
            res_RN_DM_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL_CHROM_WIDE")
            res_RN_DM_BL_CHROM_WIDE = 0
        try:
            res_RN_DM_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH_CHROM_WIDE")
            res_RN_DM_BH_CHROM_WIDE = 0
       
        T_four = {}
        T_four_half = {}
        T_four[pulsar+"_RN_DM_BL_BH"] = res_RN_DM_BL_BH
        T_four_half[pulsar+"_RN_BL_BH_CHROM"] = res_RN_BL_BH_CHROM
        T_four_half[pulsar+"_DM_BL_BH_CHROM"] = res_DM_BL_BH_CHROM
        T_four_half[pulsar+"_RN_DM_BL_CHROM"] = res_RN_DM_BL_CHROM
        T_four_half[pulsar+"_RN_DM_BH_CHROM"] = res_RN_DM_BH_CHROM
        T_four[pulsar+"_RN_BL_BH_CHROMCIDX"] = res_RN_BL_BH_CHROMCIDX
        T_four[pulsar+"_DM_BL_BH_CHROMCIDX"] = res_DM_BL_BH_CHROMCIDX
        T_four[pulsar+"_RN_DM_BL_CHROMCIDX"] = res_RN_DM_BL_CHROMCIDX
        T_four[pulsar+"_RN_DM_BH_CHROMCIDX"] = res_RN_DM_BH_CHROMCIDX
        T_four_half[pulsar+"_RN_BL_BH_CHROM_WIDE"] = res_RN_BL_BH_CHROM_WIDE
        T_four_half[pulsar+"_DM_BL_BH_CHROM_WIDE"] = res_DM_BL_BH_CHROM_WIDE
        T_four_half[pulsar+"_RN_DM_BL_CHROM_WIDE"] = res_RN_DM_BL_CHROM_WIDE
        T_four_half[pulsar+"_RN_DM_BH_CHROM_WIDE"] = res_RN_DM_BH_CHROM_WIDE


        T_four_half_sort = dict(sorted(T_four_half.items(), key=lambda item: item[1],reverse=True))
        T_four_sort = dict(sorted(T_four.items(), key=lambda item: item[1],reverse=True))
        
        # Select out condition to maintain probable model

        if list(T_four_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_four_half_sort.keys())[0]] = list(T_four_half_sort.values())[0]

        if list(T_four_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_four_sort.keys())[0]] = list(T_four_sort.values())[0]
        
        # Go through Tier 3 and choose the best
        try:
            res_RN_DM_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROM")
            res_RN_DM_CHROM = 0
        try:
            res_RN_DM_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROM_WIDE")
            res_RN_DM_CHROM_WIDE = 0

        try:
            res_RN_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_BH")
            res_RN_BL_BH = 0

        try:
            res_RN_BL_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM")
            res_RN_BL_CHROM = 0
        try:
            res_RN_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM")
            res_RN_BH_CHROM = 0
        try:
            res_RN_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROM_WIDE")
            res_RN_BL_CHROM_WIDE = 0
        try:
            res_RN_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROM_WIDE")
            res_RN_BH_CHROM_WIDE = 0

        try:
            res_RN_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL_CHROMCIDX")
            res_RN_BL_CHROMCIDX = 0
        try:
            res_RN_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BH_CHROMCIDX")
            res_RN_BH_CHROMCIDX = 0

        try:
            res_CHROM_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_BL_BH")
            res_CHROM_BL_BH = 0
        try:
            res_CHROM_WIDE_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_WIDE_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BL_BH")
            res_CHROM_WIDE_BL_BH = 0

        try:
            res_CHROMCIDX_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROMCIDX_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL_BH")
            res_CHROMCIDX_BL_BH = 0
        try:   
            res_DM_BL_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_BH")
            res_DM_BL_BH = 0
        try:
            res_RN_DM_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BL")
            res_RN_DM_BL = 0
        try:
            res_RN_DM_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_BH")
            res_RN_DM_BH = 0

        try:
            res_DM_BL_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROM")
            res_DM_BL_CHROM = 0
        try:
            res_DM_BH_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BH_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROM")
            res_DM_BH_CHROM = 0
        try:
            res_DM_BL_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROM_WIDE")
            res_DM_BL_CHROM_WIDE = 0
        try:
            res_DM_BH_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BH_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROM_WIDE")
            res_DM_BH_CHROM_WIDE = 0


        try:
            res_RN_DM_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM_CHROMCIDX")
            res_RN_DM_CHROMCIDX = 0
        try:
            res_DM_BL_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL_CHROMCIDX")
            res_DM_BL_CHROMCIDX = 0
        try:
            res_DM_BH_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BH_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH_CHROMCIDX")
            res_DM_BH_CHROMCIDX = 0
        

        T_three = {}
        T_three_half = {}
        T_three_half[pulsar+"_RN_DM_CHROM"] = res_RN_DM_CHROM
        T_three_half[pulsar+"_RN_DM_CHROM_WIDE"] = res_RN_DM_CHROM_WIDE
        T_three[pulsar+"_RN_BL_BH"] = res_RN_BL_BH
        T_three[pulsar+"_DM_BL_BH"] = res_DM_BL_BH
        T_three[pulsar+"_RN_DM_BL"] = res_RN_DM_BL
        T_three[pulsar+"_RN_DM_BH"] = res_RN_DM_BH
        T_three_half[pulsar+"_RN_BL_CHROM"] = res_RN_BL_CHROM
        T_three_half[pulsar+"_RN_BH_CHROM"] = res_RN_BH_CHROM
        T_three_half[pulsar+"_DM_BL_CHROM"] = res_DM_BL_CHROM
        T_three_half[pulsar+"_DM_BH_CHROM"] = res_DM_BH_CHROM
        T_three_half[pulsar+"_RN_BL_CHROM_WIDE"] = res_RN_BL_CHROM_WIDE
        T_three_half[pulsar+"_RN_BH_CHROM_WIDE"] = res_RN_BH_CHROM_WIDE
        T_three_half[pulsar+"_DM_BL_CHROM_WIDE"] = res_DM_BL_CHROM_WIDE
        T_three_half[pulsar+"_DM_BH_CHROM_WIDE"] = res_DM_BH_CHROM_WIDE
        T_three[pulsar+"_RN_DM_CHROMCIDX"] = res_RN_DM_CHROMCIDX
        T_three[pulsar+"_RN_BL_CHROMCIDX"] = res_RN_BL_CHROMCIDX
        T_three[pulsar+"_RN_BH_CHROMCIDX"] = res_RN_BH_CHROMCIDX
        T_three[pulsar+"_DM_BL_CHROMCIDX"] = res_DM_BL_CHROMCIDX
        T_three[pulsar+"_DM_BH_CHROMCIDX"] = res_DM_BH_CHROMCIDX
        T_three_half[pulsar+"_CHROM_BL_BH"] = res_CHROM_BL_BH
        T_three_half[pulsar+"_CHROM_WIDE_BL_BH"] = res_CHROM_WIDE_BL_BH
        T_three[pulsar+"_CHROMCIDX_BL_BH"] = res_CHROMCIDX_BL_BH
        
        T_three_half_sort = dict(sorted(T_three_half.items(), key=lambda item: item[1],reverse=True))
        T_three_sort = dict(sorted(T_three.items(), key=lambda item: item[1],reverse=True))

        if list(T_three_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_three_half_sort.keys())[0]] = list(T_three_half_sort.values())[0]

        if list(T_three_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_three_sort.keys())[0]] = list(T_three_sort.values())[0]

        # Same for Tier 2

        try:
            res_RN_DM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_DM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_DM")
            res_RN_DM = 0

        try:
            res_RN_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_CHROM")
            res_RN_CHROM = 0
        try:
            res_RN_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_CHROM_WIDE")
            res_RN_CHROM_WIDE = 0

        try:
            res_RN_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BL")
            res_RN_BL = 0  
        try:
            res_RN_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_BH")
            res_RN_BH = 0
        try:
            res_RN_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN_CHROMCIDX")
            res_RN_CHROMCIDX = 0 

        try:
            res_CHROM_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_BL")
            res_CHROM_BL = 0 
        try:
            res_CHROM_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_BH")
            res_CHROM_BH = 0 
        try:
            res_CHROM_WIDE_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_WIDE_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BL")
            res_CHROM_WIDE_BL = 0 
        try:
            res_CHROM_WIDE_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_WIDE_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE_BH")
            res_CHROM_WIDE_BH = 0 

        try:
            res_CHROMCIDX_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROMCIDX_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BL")
            res_CHROMCIDX_BL = 0 
        try:
            res_CHROMCIDX_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROMCIDX_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROMCIDX_BH")
            res_CHROMCIDX_BH = 0 

        try:
            res_DM_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROM")
            res_DM_CHROM = 0  
        try:
            res_DM_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROM_WIDE")
            res_DM_CHROM_WIDE = 0  

        try:
            res_DM_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BL")
            res_DM_BL = 0 
        try:
            res_DM_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_BH")
            res_DM_BH = 0 
        try:
            res_DM_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM_CHROMCIDX")
            res_DM_CHROMCIDX = 0 
        try:
            res_BH_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_BH_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BH_BL")
            res_BH_BL = 0
        
        T_two = {}
        T_two_half = {}
        T_two[pulsar+"_RN_DM"] = res_RN_DM
        T_two_half[pulsar+"_RN_CHROM"] = res_RN_CHROM
        T_two_half[pulsar+"_DM_CHROM"] = res_DM_CHROM
        T_two_half[pulsar+"_RN_CHROM_WIDE"] = res_RN_CHROM_WIDE
        T_two_half[pulsar+"_DM_CHROM_WIDE"] = res_DM_CHROM_WIDE
        T_two[pulsar+"_RN_BL"] = res_RN_BL
        T_two[pulsar+"_RN_BH"] = res_RN_BH
        T_two[pulsar+"_DM_BL"] = res_DM_BL
        T_two[pulsar+"_DM_BH"] = res_DM_BH
        T_two[pulsar+"_BH_BL"] = res_BH_BL
        T_two[pulsar+"_RN_CHROMCIDX"] = res_RN_CHROMCIDX
        T_two[pulsar+"_DM_CHROMCIDX"] = res_DM_CHROMCIDX
        T_two_half[pulsar+"_CHROM_BL"] = res_CHROM_BL
        T_two_half[pulsar+"_CHROM_BH"] = res_CHROM_BH
        T_two_half[pulsar+"_CHROM_WIDE_BL"] = res_CHROM_WIDE_BL
        T_two_half[pulsar+"_CHROM_WIDE_BH"] = res_CHROM_WIDE_BH
        T_two[pulsar+"_CHROMCIDX_BL"] = res_CHROMCIDX_BL
        T_two[pulsar+"_CHROMCIDX_BH"] = res_CHROMCIDX_BH

        T_two_half_sort = dict(sorted(T_two_half.items(), key=lambda item: item[1],reverse=True))
        T_two_sort = dict(sorted(T_two.items(), key=lambda item: item[1],reverse=True))

        if list(T_two_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_two_half_sort.keys())[0]] = list(T_two_half_sort.values())[0]

        if list(T_two_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_two_sort.keys())[0]] = list(T_two_sort.values())[0]
        

        # Same for Tier 1
        try:
            res_RN = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_RN/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_RN")
            res_RN = 0

        try:
            res_CHROM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM")
            res_CHROM = 0
        try:
            res_CHROM_WIDE = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROM_WIDE/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROM_WIDE")
            res_CHROM_WIDE = 0

        try:
            res_CHROMCIDX = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_CHROMCIDX/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_CHROMCIDX")
            res_CHROMCIDX = 0
        try:
            res_DM = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_DM/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_DM")
            res_DM = 0
        try:
            res_BH = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_BH/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BH")
            res_BH = 0
        try:
            res_BL = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_BL/*json")[0]).log_evidence
        except:
            print(pulsar+" does not have: "+pulsar+"_BL")
            res_BL = 0
            

        T_one = {}
        T_one_half = {}
        T_one[pulsar+"_RN"] = res_RN
        T_one[pulsar+"_DM"] = res_DM
        T_one_half[pulsar+"_CHROM"] = res_CHROM
        T_one[pulsar+"_CHROMCIDX"] = res_CHROMCIDX
        T_one[pulsar+"_BH"] = res_BH
        T_one[pulsar+"_BL"] = res_BL

        T_one_half_sort = dict(sorted(T_one_half.items(), key=lambda item: item[1],reverse=True))
        T_one_sort = dict(sorted(T_one.items(), key=lambda item: item[1],reverse=True))

        if list(T_one_half_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_one_half_sort.keys())[0]] = list(T_one_half_sort.values())[0]

        if list(T_one_sort.values())[0] + bf_cutoff > list(choice.values())[0]:
            choice.popitem()
            choice[list(T_one_sort.keys())[0]] = list(T_one_sort.values())[0]
        '''
        try:
            res_WN = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_WN/*json")[0]).log_evidence
            ml_WN = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_WN/*json")[0]).log_likelihood_evaluations.max()
        except:
            print(pulsar+" does not have: "+pulsar+"_WN")
            res_WN = 0
        try:
            res_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_evidence
            ml_WN_NO_ECORR = bilby.result.read_in_result(glob.glob(alt_dir+"/"+pulsar+"_WN_NO_ECORR/*json")[0]).log_likelihood_evaluations.max()
        except:
            print(pulsar+" does not have: "+pulsar+"_WN_NO_ECORR")
            res_WN_NO_ECORR = 0

        if res_WN > res_WN_NO_ECORR +2:
            if ml_WN > list(choice.values())[0]:
                choice.popitem()
                choice[pulsar+"_WN"] = res_WN
        
            chosen_evidence.update(choice)
        else:
            if ml_WN > list(choice.values())[0]:
                choice.popitem()
                choice[pulsar+"_WN"] = res_WN
        ''' 
        chosen_evidence.update(choice)

    os.chdir("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models")

    with open(outfile,"a+") as outFile:
        json.dump(chosen_evidence,outFile,indent=4)







    

    













    


