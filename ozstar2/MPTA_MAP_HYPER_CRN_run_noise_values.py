import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="Noise value extractor")
parser.add_argument("-type", dest="type", choices={"full", "white", "red", "inc_ER", "full_433", "full_433_global", "full_global", "median"}, help="The type of nosise to extract", required=True)
parser.add_argument("-post_dir", dest="post_dir", help="Directory containing the posteriors",required = True)
parser.add_argument("-models", dest="models", type=str, help="Json file with models to extract", required=False)
parser.add_argument("-outfile", dest="outfile", help="Path of the output .json file", required=True)

args = parser.parse_args()

extract = args.type
post_dir = str(args.post_dir)
models = str(args.models)
outfile = str(args.outfile)

if extract == "white":
    wn_models ={}

    jsons = sorted(glob.glob(post_dir+"/J*/*WN*/*json"))

    for file in jsons:
        print(file)

        temp_df = pd.read_json(json.load(open(file)))
        psrname = temp_df.columns[0].split("_")[0]
        
        efac_name = psrname+"_KAT_MKBF_efac"
        equad_name = psrname+"_KAT_MKBF_log10_tnequad"
        ecorr_name = psrname+"_basis_ecorr_KAT_MKBF_log10_ecorr"

        temp_efac = temp_df[efac_name].values
        temp_equad = temp_df[equad_name].values
        temp_ecorr = temp_df[ecorr_name].values

        wn_amp_bins = np.log10(np.logspace(-10,-1,200))

        pdf_efac, edges_efac = np.histogram(temp_efac, bins=200, density=True)
        vals_efac = (edges_efac[:-1] + edges_efac[1:]) / 2
        map_efac = vals_efac[pdf_efac.argmax()]

        pdf_equad, edges_equad = np.histogram(temp_equad, bins=wn_amp_bins, density=True)
        vals_equad = (edges_equad[:-1] + edges_equad[1:]) / 2
        map_equad = vals_equad[pdf_equad.argmax()]
        
        pdf_ecorr, edges_ecorr = np.histogram(temp_ecorr, bins=wn_amp_bins, density=True)
        vals_ecorr = (edges_ecorr[:-1] + edges_ecorr[1:]) / 2
        map_ecorr = vals_ecorr[pdf_ecorr.argmax()]

        wn_models[efac_name] = map_efac
        wn_models[equad_name] = map_equad
        wn_models[ecorr_name] = map_ecorr

    with open(outfile,"a+") as outFile:
        json.dump(wn_models,outFile,indent=4)

elif extract == "full":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    f = open("CRN_pars_noER.txt", "r")
    pars = list(f.readlines())
    f.close()
    chainall = np.load("CRN_all_updated_short.npy")

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    fp = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
    psrs = list(fp.readlines())
    fp.close()

    for psr in psrs:
        psrname = psr.strip("\n")
        ind = np.argwhere([psrname in p or "gw" in p for p in pars]).squeeze()
        psr_pars = [ pars[i] for i in ind ]
        psr_pars = [ p.strip("\n") for p in psr_pars ]
        psrchain = chainall[:, ind]

        wn_amp_bins = np.log10(np.logspace(-10,-1,200))
        
        for i, par in enumerate(psr_pars):
            temp_vals = psrchain[:,i]
            
            if "efac" in par:
                pdf_efac, edges_efac = np.histogram(temp_vals, bins=200, density=True)
                vals_efac = (edges_efac[:-1] + edges_efac[1:]) / 2
                temp_map = vals_efac[pdf_efac.argmax()]
                mpta_models[par] = temp_map

            if "equad" in par:
                pdf_equad, edges_equad = np.histogram(temp_vals, bins=wn_amp_bins, density=True)
                vals_equad = (edges_equad[:-1] + edges_equad[1:]) / 2
                temp_map = vals_equad[pdf_equad.argmax()]
                mpta_models[par] = temp_map
            
            if "ecorr" in par:
                pdf_ecorr, edges_ecorr = np.histogram(temp_vals, bins=wn_amp_bins, density=True)
                vals_ecorr = (edges_ecorr[:-1] + edges_ecorr[1:]) / 2
                temp_map = vals_ecorr[pdf_ecorr.argmax()]
                mpta_models[par] = temp_map

            #if ("ecorr" not in par) * ("equad" not in par) * ("efac" not in par) * ("gw" not in par):
            if ("ecorr" not in par) * ("equad" not in par) * ("efac" not in par):
                if "red" in par:
                    if "log10_A" in par:
                        if "chrom1yr" not in par and "bump" not in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                        elif "chrom1yr" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                        elif "bump" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                        elif "sw" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)
                    else:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)   

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map
            
                elif ("log10_A" in par) * ("red" not in par):
                    if "chrom1yr" not in par and "bump" not in par and "sw" not in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                    elif "chrom1yr" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                    elif "bump" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                    elif "sw" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map
                else:
                    temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map



elif extract == "full_433":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    f = open("CRN_pars.txt", "r")
    pars = list(f.readlines())
    f.close()
    chainall = np.load("CRN_ER_all_updated_short.npy")

    indgamma = np.argwhere(["gw_gamma" in p for p in pars]).squeeze()

    chainall = chainall[ (chainall[:, indgamma] < 4.43) * (chainall[:, indgamma] > 4.23) ]

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    fp = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
    psrs = list(fp.readlines())
    fp.close()

    for psr in psrs:
        psrname = psr.strip("\n")
        ind = np.argwhere([psrname in p or "gw" in p for p in pars]).squeeze()
        psr_pars = [ pars[i] for i in ind ]
        psr_pars = [ p.strip("\n") for p in psr_pars ]
        psrchain = chainall[:, ind]

        wn_amp_bins = np.log10(np.logspace(-10,-1,200))
        
        for i, par in enumerate(psr_pars):
            temp_vals = psrchain[:,i]
            
            if "efac" in par:
                pdf_efac, edges_efac = np.histogram(temp_vals, bins=200, density=True)
                vals_efac = (edges_efac[:-1] + edges_efac[1:]) / 2
                temp_map = vals_efac[pdf_efac.argmax()]
                mpta_models[par] = temp_map

            if "equad" in par:
                pdf_equad, edges_equad = np.histogram(temp_vals, bins=wn_amp_bins, density=True)
                vals_equad = (edges_equad[:-1] + edges_equad[1:]) / 2
                temp_map = vals_equad[pdf_equad.argmax()]
                mpta_models[par] = temp_map
            
            if "ecorr" in par:
                pdf_ecorr, edges_ecorr = np.histogram(temp_vals, bins=wn_amp_bins, density=True)
                vals_ecorr = (edges_ecorr[:-1] + edges_ecorr[1:]) / 2
                temp_map = vals_ecorr[pdf_ecorr.argmax()]
                mpta_models[par] = temp_map

            #if ("ecorr" not in par) * ("equad" not in par) * ("efac" not in par) * ("gw" not in par):
            if ("ecorr" not in par) * ("equad" not in par) * ("efac" not in par):
                if "red" in par:
                    if "log10_A" in par:
                        if "chrom1yr" not in par and "bump" not in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                        elif "chrom1yr" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                        elif "bump" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                        elif "sw" in par:
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)
                    else:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)   

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map
            
                elif ("log10_A" in par) * ("red" not in par):
                    if "chrom1yr" not in par and "bump" not in par and "sw" not in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                    elif "chrom1yr" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                    elif "bump" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                    elif "sw" in par:
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map
                else:
                    temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[par] = temp_map

elif extract == "full_433_global":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    f = open("CRN_pars.txt", "r")
    pars = list(f.readlines())
    f.close()
    chainall = np.load("CRN_ER_all_updated_short.npy")

    indgamma = np.argwhere(["gw_gamma" in p for p in pars]).squeeze()

    chainall = chainall[ (chainall[:, indgamma] < 4.43) * (chainall[:, indgamma] > 4.23) ]

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    fp = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
    psrs = list(fp.readlines())
    fp.close()

    likelihoods = chainall[:,-3]
    ml_ind = likelihoods.argmax()
    ml_chain = chainall[ml_ind, :]

    for i, par in enumerate(pars):
        par = par.rstrip("\n")
        mpta_models[par] = ml_chain[i]

elif extract == "full_global":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    f = open("CRN_pars_noER.txt", "r")
    pars = list(f.readlines())
    f.close()
    chainall = np.load("CRN_all_updated_short.npy")

    chainall = np.loadtxt("master_chain_CRN_DATA_new.txt")

    indgamma = np.argwhere(["gw_gamma" in p for p in pars]).squeeze()

    #chainall = chainall[ (chainall[:, indgamma] < 4.43) * (chainall[:, indgamma] > 4.23) ]

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    fp = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
    psrs = list(fp.readlines())
    fp.close()

    likelihoods = chainall[:,-3]
    ml_ind = likelihoods.argmax()
    ml_chain = chainall[ml_ind, :]

    for i, par in enumerate(pars):
        par = par.rstrip("\n")
        mpta_models[par] = ml_chain[i]


elif extract == "red":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    jsons = sorted(glob.glob(post_dir+"/J*/*json"))

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    for file in jsons:
        temp_df = pd.read_json(json.load(open(file)))
        psrname = temp_df.columns[0].split("_")[0]

        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if psrname in psr_model ][0].split("_")[1:]
        
        for col in temp_df.columns:
            if ("ecorr" not in col) * ("equad" not in col) * ("efac" not in col) * ("gw" not in col):
                            
                if "red" in col:
                    if "RN" in psrmodels:
                        if "log10_A" in col:
                            if "chrom1yr" not in col and "bump" not in col and "sw" not in col:
                                temp_vals = temp_df[col].values
                                temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                            elif "chrom1yr" in col:
                                temp_vals = temp_df[col].values
                                temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                            elif "bump" in col:
                                temp_vals = temp_df[col].values
                                temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                            elif "sw" in col:
                                temp_vals = temp_df[col].values
                                temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)
                        else:
                            temp_vals = temp_df[col].values
                            temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)   
                        
                        centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                        temp_map = centers[temp_pdf.argmax()]

                        mpta_models[col] = temp_map

                elif ("log10_A" in col) * ("red" not in col):
                    if "chrom1yr" not in col and "bump" not in col and "sw" not in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                    elif "chrom1yr" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                    elif "bump" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                    elif "sw" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[col] = temp_map
                else:
                    temp_vals = temp_df[col].values
                    temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[col] = temp_map
                
elif extract == "inc_ER":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    jsons = sorted(glob.glob(post_dir+"/J*/*json"))

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    for file in jsons:
        temp_df = pd.read_json(json.load(open(file)))
        psrname = temp_df.columns[0].split("_")[0]

        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if psrname in psr_model ][0].split("_")[1:]
        
        for col in temp_df.columns:
            if ("ecorr" not in col) * ("equad" not in col) * ("efac" not in col) * ("gw" not in col):
                            
                if ("log10_A" in col):
                    if "chrom1yr" not in col and "bump" not in col and "sw" not in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)
                    elif "chrom1yr" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=annual_A_bins, density=True)
                    elif "bump" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=bump10_A_bins, density=True)
                    elif "sw" in col:
                        temp_vals = temp_df[col].values
                        temp_pdf, temp_edges = np.histogram(temp_vals, bins=sw_A_bins, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[col] = temp_map
                else:
                    temp_vals = temp_df[col].values
                    temp_pdf, temp_edges = np.histogram(temp_vals, bins=100, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[col] = temp_map

elif extract == "median":
    
    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    f = open("CRN_pars_noER.txt", "r")
    pars = list(f.readlines())
    f.close()
    chainall = np.load("CRN_all_updated_short.npy")

    indgamma = np.argwhere(["gw_gamma" in p for p in pars]).squeeze()

    chainall = chainall[ (chainall[:, indgamma] < 4.43) * (chainall[:, indgamma] > 4.23) ]

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))

    fp = open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt", "r")
    psrs = list(fp.readlines())
    fp.close()

    for psr in psrs:
        psrname = psr.strip("\n")
        ind = np.argwhere([psrname in p or "gw" in p for p in pars]).squeeze()
        psr_pars = [ pars[i] for i in ind ]
        psr_pars = [ p.strip("\n") for p in psr_pars ]
        psrchain = chainall[:, ind]

        for i, par in enumerate(psr_pars):
            temp_vals = psrchain[:,i]

            mpta_models[par] = np.median(temp_vals)






with open(outfile,"a+") as outFile:
    json.dump(mpta_models,outFile,indent=4)