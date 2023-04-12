import numpy as np
from numpy.random import multivariate_normal

import argparse
import astropy
import os,  os.path, sys
import glob
import chainconsumer
from chainconsumer import ChainConsumer
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

clr = [mcolors.CSS4_COLORS['mediumseagreen'],mcolors.CSS4_COLORS['firebrick'], mcolors.CSS4_COLORS['goldenrod'],mcolors.CSS4_COLORS['steelblue'],mcolors.CSS4_COLORS['palegreen'],mcolors.CSS4_COLORS['palevioletred'],mcolors.CSS4_COLORS['darkorange'],mcolors.CSS4_COLORS['cornflowerblue'],
       mcolors.CSS4_COLORS['salmon'],mcolors.CSS4_COLORS['lavender'],mcolors.CSS4_COLORS['lemonchiffon'],mcolors.CSS4_COLORS['darkcyan'],mcolors.CSS4_COLORS['olive'],mcolors.CSS4_COLORS['slategray'],mcolors.CSS4_COLORS['sienna'],mcolors.CSS4_COLORS['rebeccapurple']]

from astropy import units as u
from astropy.coordinates import SkyCoord

import bilby

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Pulsar pair cross correlation enterprise noise run.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsars to run the plots on", required = True)
parser.add_argument("-data", dest="data", help="Directory to read in.", required = True)
parser.add_argument("-noise", type = str.lower, nargs="+",dest="noise", help="Which noise terms are plotted. Takes arguments as '-noise efac red dm' etc.", \
    choices={"efac", "equad", "ecorr", "dm", "red", "sw", "chrom","n_earth", "all"})
parser.add_argument("-outdir", dest="outdir", help="Directory to write out to.", required = False)
parser.add_argument("-scale", dest="scale", help="Scales the posteriors for potentially better viewing.", required = False)
args = parser.parse_args()

pulsar = args.pulsar
data_dir = args.data
noise = args.noise
outdir = args.outdir
scale = args.scale


os.chdir(data_dir)
#Get all of the pulsar pair directories
pair_dirs = glob.glob(data_dir+"/*"+pulsar+"*")

results = {}

for pair_dir in pair_dirs:
    pair = pair_dir.split("/")[-1]
    try:
        results[pair] = bilby.result.read_in_result(pair_dir+"/"+pair+"_result.json")
    except OSError:
        print(pair+" is not finished")

pars = []
labels = []
for n in noise:
    if n == "efac":
        pars.append(pulsar+"_KAT_MKBF_efac")
        labels.append("EFAC")
    if n == "equad":
        pars.append(pulsar+"_KAT_MKBF_log10_ecorr")
        labels.append("EQUAD")
    if n == "ecorr":
        pars.append(pulsar+"_KAT_MKBF_log10_ecorr")
        labels.append("ECORR")
    if n == "dm":
        pars.append(pulsar+"_dm_gp_gamma")
        labels.append(r"$\gamma_{\mathrm{DM}}$")
        pars.append(pulsar+"_dm_gp_log10_A")
        labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{DM}})$")
    if n == "red":
        pars.append(pulsar+"_red_noise_gamma")
        labels.append(r"$\gamma_{\mathrm{Red}}$")
        pars.append(pulsar+"_red_noise_log10_A")
        labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Red}})$")
    if n == "sw":
        pars.append(pulsar+"_gp_sw_gamma")
        labels.append(r"$\gamma_{\mathrm{SW}}$")
        pars.append(pulsar+"_gp_sw_log10_A")
        labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{SW}})$")
    if n == "n_earth":
        pars.append(pulsar+"_n_earth_n_earth")
        labels.append(r"SW_NEARTH")
    if n == "chrom":
        if pulsar+"_chrom_gp_gamma" in list(results.values())[0].parameter_labels:
            pars.append(pulsar+"_chrom_gp_gamma")
            labels.append(r"$\gamma_{\mathrm{Chrom}}$")
            pars.append(pulsar+"_chrom_gp_log10_A")
            labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Chrom}})$")
        elif pulsar+"_chrom_wide_gp_gamma" in list(results.values())[0].parameter_labels:
            pars.append(pulsar+"_chrom_wide_gp_gamma")
            labels.append(r"$\gamma_{\mathrm{Chrom}}$")
            pars.append(pulsar+"_chrom_wide_gp_log10_A")
            labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Chrom}})$")
        
        if pulsar+"_chrom_gp_idx" in list(results.values())[0].parameter_labels:
            pars.append(pulsar+"_chrom_gp_idx")
            labels.append(r"Chromatic Index")
        if pulsar+"_chrom_wide_gp_idx" in list(results.values())[0].parameter_labels:
            pars.append(pulsar+"_chrom_wide_gp_idx")
            labels.append(r"Chromatic Index")

    if n == "all":
        for proc in list(results.values())[0].parameter_labels:
            if proc == pulsar+"_KAT_MKBF_efac":
                pars.append(proc)
                labels.append("EFAC")
            if proc == pulsar+"_KAT_MKBF_log10_ecorr":
                pars.append(proc)
                labels.append("EQUAD")
            if proc == pulsar+"_KAT_MKBF_log10_ecorr":
                pars.append(proc)
                labels.append("ECORR")
            if proc == pulsar+"_dm_gp_gamma":
                pars.append(proc)
                labels.append(r"$\gamma_{\mathrm{DM}}$")
            if proc == pulsar+"_dm_gp_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{DM}})$")
            if proc == pulsar+"_red_noise_gamma":
                pars.append(proc)
                labels.append(r"$\gamma_{\mathrm{Red}}$")
            if proc == pulsar+"_red_noise_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Red}})$")
            if proc == pulsar+"_gp_sw_gamma":
                pars.append(proc)
                labels.append(r"$\gamma_{\mathrm{SW}}$")
            if proc == pulsar+"_gp_sw_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{SW}})$")
            if proc == pulsar+"_n_earth_n_earth":
                pars.append(proc)
                labels.append(r"SW_NEARTH")
            if proc == pulsar+"_chrom_gp_gamma":
                pars.append(proc)
                labels.append(r"$\gamma_{\mathrm{Chrom}}$")
            if proc == pulsar+"_chrom_gp_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Chrom}})$")
            if proc == pulsar+"_chrom_wide_gp_gamma":
                pars.append(proc)
                labels.append(r"$\gamma_{\mathrm{Chrom}}$")
            if proc == pulsar+"_chrom_wide_gp_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{Chrom}})$")
            if proc == pulsar+"_chrom_gp_idx":
                pars.append(proc)
                labels.append(r"Chromatic Index")
            if proc == pulsar+"_chrom_wide_gp_idx":
                pars.append(proc)
                labels.append(r"Chromatic Index")
            if proc == "gw_bins_log10_A":
                pars.append(proc)
                labels.append(r"$\log_{10}(\mathrm{A}_{\mathrm{GW}})$")




total_label = pulsar+"_"+"_".join(noise)

if outdir:
    _ = bilby.core.result.plot_multiple(
        list(results.values()), labels=[pulsar+" pairs: {}".format(len(results))], parameters=pars, corner_labels = labels, filename=outdir+"/{}".format(total_label)
    )
    plt.suptitle(total_label, fontsize = 20)
    plt.tight_layout()
    #plt.show()
    plt.clf()
else:
    _ = bilby.core.result.plot_multiple(
        list(results.values()), labels=[pulsar+" pairs: {}".format(len(results))], parameters=pars, corner_labels = labels, filename="{}".format(total_label)
    )
    plt.suptitle(total_label, fontsize = 20)
    plt.tight_layout()
    #plt.show()
    plt.clf()


    

    


    

