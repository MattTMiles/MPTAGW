import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="Noise value extractor")
parser.add_argument("-type", dest="type", choices={"full", "white", "red", "inc_ER", "gw_dummy"}, help="The type of nosise to extract", required=True)
parser.add_argument("-post_dir", dest="post_dir", help="Directory containing the posteriors",required = True)
parser.add_argument("-models", dest="models", type=str, help="Json file with models to extract", required=False)
parser.add_argument("-outfile", dest="outfile", help="Path of the output .json file", required=True)

args = parser.parse_args()

extract = args.type
post_dir = str(args.post_dir)
models = str(args.models)
outfile = str(args.outfile)

if extract == "white":
    mpta_models ={}

    jsons = sorted(glob.glob(post_dir+"/J*/*json"))

    for file in jsons:
        print(file)

        temp_df = pd.read_json(json.load(open(file)))
        psrname = temp_df.columns[0].split("_")[0]
        
        efac_name = psrname+"_KAT_MKBF_efac"
        equad_name = psrname+"_KAT_MKBF_log10_tnequad"
        #ecorr_name = psrname+"_basis_ecorr_KAT_MKBF_log10_ecorr"
        ecorr_name = psrname+"_KAT_MKBF_log10_ecorr"

        wn_amp_bins = np.log10(np.logspace(-10,-1,200))

        temp_efac = temp_df[efac_name].values

        pdf_efac, edges_efac = np.histogram(temp_efac, bins=200, density=True)
        vals_efac = (edges_efac[:-1] + edges_efac[1:]) / 2
        map_efac = vals_efac[pdf_efac.argmax()]
        
        mpta_models[efac_name] = map_efac
        
        try:
            temp_equad = temp_df[equad_name].values
            
            pdf_equad, edges_equad = np.histogram(temp_equad, bins=wn_amp_bins, density=True)
            vals_equad = (edges_equad[:-1] + edges_equad[1:]) / 2
            map_equad = vals_equad[pdf_equad.argmax()]

            mpta_models[equad_name] = map_equad
        except:
            KeyError
        try:
            temp_ecorr = temp_df[ecorr_name].values

            pdf_ecorr, edges_ecorr = np.histogram(temp_ecorr, bins=wn_amp_bins, density=True)
            vals_ecorr = (edges_ecorr[:-1] + edges_ecorr[1:]) / 2
            map_ecorr = vals_ecorr[pdf_ecorr.argmax()]

            mpta_models[ecorr_name] = map_ecorr
        except:
            KeyError


elif extract == "full":

    mpta_models = {}
    
    #Assuming the best models have already been moved into this directory
    jsons = sorted(glob.glob(post_dir+"/J*/*json"))

    log10_A_bins = np.log10(np.logspace(-20,-11,200))
    bump10_A_bins = np.log10(np.logspace(-10,-1,200))
    annual_A_bins = np.log10(np.logspace(-20,-5,300))
    sw_A_bins = np.log10(np.logspace(-10,1,200))
    wn_amp_bins = np.log10(np.logspace(-10,-1,200))

    for file in jsons:
        temp_df = pd.read_json(json.load(open(file)))
        psrname = temp_df.columns[0].split("_")[0]

        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if psrname in psr_model ][0].split("_")[1:]
        
        for col in temp_df.columns:
            if "efac" in col:
                temp_vals = temp_df[col].values
                temp_pdf, temp_edges = np.histogram(temp_vals, bins=200, density=True)

            if "equad" in col:
                temp_vals = temp_df[col].values
                temp_pdf, temp_edges = np.histogram(temp_vals, bins=wn_amp_bins, density=True)
            
            if "ecorr" in col:
                temp_vals = temp_df[col].values
                temp_pdf, temp_edges = np.histogram(temp_vals, bins=wn_amp_bins, density=True)

            centers = (temp_edges[:-1] + temp_edges[1:]) / 2
            temp_map = centers[temp_pdf.argmax()]

            mpta_models[col] = temp_map

            if ("ecorr" not in col) * ("equad" not in col) * ("efac" not in col) * ("gw" not in col):
                            
                if "red" in col:
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
        #psrname = temp_df.columns[0].split("_")[0]
        psrname = file.split("/")[1].split("_")[0]

        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/DJR_data/MPTA_noise_models.json"))
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

elif extract == "gw_dummy":

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
            if "gw" in col:
                            
                if ("log10_A" in col):
                    temp_vals = temp_df[col].values
                    temp_pdf, temp_edges = np.histogram(temp_vals, bins=log10_A_bins, density=True)

                    centers = (temp_edges[:-1] + temp_edges[1:]) / 2
                    temp_map = centers[temp_pdf.argmax()]

                    mpta_models[psrname+"_"+col] = temp_map


with open(outfile,"a+") as outFile:
    json.dump(mpta_models,outFile,indent=4)