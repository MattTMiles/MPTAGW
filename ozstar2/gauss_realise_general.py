import numpy as np
import sys
import importlib.util
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.table import QTable, Table, Column
import pandas as pd

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
import enterprise.constants as const
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

import gc
from scipy import stats
from scipy.stats import anderson

import argparse
import os

#sys.path.insert(0,"/home/mmiles/soft/PINT/src")
import pint
from pint.models import *

import pint.fitter
from pint.residuals import Residuals
from pint.toa import get_TOAs
import pint.logging
import pint.config
from scipy.interpolate import interp1d
import astropy.units as u

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = True)
parser.add_argument("-directory", dest="directory", help="Directory where par and tim are found", required = True)
parser.add_argument("-noise", type = str.lower, nargs = "+", dest="noise", help="Noise processes that can be realised", choices = {"red", "gw", "dm", "sw", "ecorr", "chrom"}, required = True)
args = parser.parse_args()
pulsar = str(args.pulsar)
maindir = str(args.directory)
noise = args.noise


def co_coll(directory, pulsar, noise):

    maindir = directory+"/"
    psr = Pulsar(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", ephem="DE440")

    m, t_all = get_model_and_toas(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", allow_name_mixing=True)
    psrname = psr.name
    f = pint.fitter.DownhillGLSFitter(toas=t_all, model=m)
    f.fit_toas(maxiter=3, debug=True)

    noise_dims = f.current_state.model.noise_model_dimensions(f.toas)
    ntmpar = len(f.model.free_params)

    noise_translate = {"red":"pl_red_noise", "gw": "pl_gw_noise", "dm": "pl_DM_noise", "sw": "SW_noise", "ecorr":"ecorr_noise", "chrom": "pl_chrom_noise"}   

    noise_term = noise_translate[noise]
    p0 = noise_dims[noise_term][0] + ntmpar +1
    p1 = noise_dims[noise_term][0] + ntmpar +1 + noise_dims[noise_term][1]

    xhat = f.current_state.xhat
    ab_coeffs = xhat[p0:p1]/f.current_state.norm[p0:p1]

    Tspan = np.max(psr.toas) - np.min(psr.toas)

    f_coeffs = np.linspace(1/Tspan, ((p1-p0)/2)/Tspan, int((p1-p0)/2))

    interp_region = np.linspace(psr.toas.min(), psr.toas.max(), 100000)

    return ab_coeffs, f_coeffs, interp_region


def gauss_realise(directory, pulsar, noise):

    ab_coeff_file = directory+"/"+pulsar+"_"+noise+"_ab_coeffs.npy"
    ab_coeffs = np.load(ab_coeff_file,allow_pickle=True)

    freq_coeff_file = directory+"/"+pulsar+"_"+noise+"_freq_coeffs.npy"
    freq_coeffs = np.load(freq_coeff_file,allow_pickle=True)

    time_file = directory+"/"+pulsar+"_"+noise+"_interp_region.npy"
    times = np.load(time_file, allow_pickle=True)

    gpt = 0
    i=0
    for f in freq_coeffs:
        
        gpt += ab_coeffs[i]*np.sin(2*np.pi*f*times) + ab_coeffs[i+1]*np.cos(2*np.pi*f*times)
        i +=2

    return gpt

def gpt_plotter(directory, pulsar, noise):

    noise_translate = {"red":"pl_red_noise", "gw": "pl_gw_noise", "dm": "pl_DM_noise", "sw": "SW_noise", "ecorr":"ecorr_noise", "chrom": "pl_chrom_noise"}  
    gpt_files = glob.glob(directory+"/"+pulsar+"*gpt.npy")
    #gpt_file = directory+"/"+pulsar+"_"+noise+"_gpt.npy"
    #gpt = np.load(gpt_file, allow_pickle=True)

    maindir = directory+"/"
    psr = Pulsar(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", ephem="DE440")

    m, t_all = get_model_and_toas(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", allow_name_mixing=True)
    psrname = psr.name
    f = pint.fitter.DownhillGLSFitter(toas=t_all, model=m)
    f.fit_toas(maxiter=3, debug=True)

    mjds = f.toas.get_mjds()
    noise_df = pd.DataFrame.from_dict(f.resids.noise_resids)
    noise_df["MJD"] = mjds.value

    time_file = directory+"/"+pulsar+"_"+noise[0]+"_interp_region.npy"
    times = np.load(time_file, allow_pickle=True)

    fig, ax = plt.subplots(2, figsize = (15, 10))
    ax[0].errorbar(x=noise_df["MJD"], y=f.resids.resids.value,yerr=f.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals")
    ax[1].errorbar(x=noise_df["MJD"], y=f.resids.resids.value,yerr=f.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals", alpha=0.15, zorder=2)
    ax[0].set_title(psrname + " residuals")
    ax[0].legend()
    ax[0].set_ylabel("Residuals (s)")

    for noise_gpt in gpt_files:
        noise_key = noise_gpt.split("_")[-2]
        if noise_key != "ecorr":
            n_gpt = np.load(noise_gpt)
            times = np.load(time_file, allow_pickle=True)
            times = times*(u.s)
            times_day = times.to(u.d)
            times_value = times_day.value

            noise_tag = noise_translate[noise_key]
            if noise_key == "red":
                lab = "Achromatic Red Noise"
            elif noise_key == "gw":
                lab = "GW signal"
            elif noise_key == "dm":
                lab = "Dispersion Measure Noise"
            elif noise_key == "sw":
                lab = "Solar Wind"

                noise_df["Rounded MJD"] = noise_df["MJD"].round(decimals=4)
                sw_ave = noise_df.groupby(["Rounded MJD"])["SW_noise"].median()


                f_interp = interp1d(sw_ave.index.values, sw_ave.values)

                times_value = times_value[times_value > sw_ave.index.values.min()]
                sw_interp = f_interp(times_value)
                n_gpt = sw_interp
                times = times_value

            elif noise_key == "ecorr":
                lab = "GP ECORR"
            elif noise_key == "pl_chrom_noise":
                lab = "Chromatic Noise"
            else:
                lab = noise_key

            #ax[0].plot(times, n_gpt, linewidth = 2, alpha=0.25, label=lab)
            ax[1].plot(times_value, n_gpt, linewidth = 2, alpha=0.75, label=lab, zorder=10)


    ax[1].set_title(psrname + " residuals + noise processes")
    ax[1].legend()
    ax[1].set_ylabel("Residuals (s)")
    ax[1].set_xlabel("MJD")

    fig.savefig(maindir+pulsar+"_gauss_realised_noise.png")
    fig.clf()
    plt.clf()


    fig1, ax = plt.subplots(2, figsize = (15, 10))
    ax[0].errorbar(x=noise_df["MJD"], y=f.resids.resids.value,yerr=f.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals")
    ax[1].errorbar(x=noise_df["MJD"], y=f.resids.resids.value,yerr=f.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals", alpha=0.15, zorder=2)
    ax[0].set_title(psrname + " residuals", fontsize=15)
    ax[0].legend(fontsize=15)
    ax[0].set_ylabel("Residuals (s)", fontsize=15)

    for noise_gpt in gpt_files:
        noise_key = noise_gpt.split("_")[-2]
        if noise_key != "ecorr":
            n_gpt = np.load(noise_gpt)
            times = np.load(time_file, allow_pickle=True)
            times = times*(u.s)
            times_day = times.to(u.d)
            times_value = times_day.value

            noise_tag = noise_translate[noise_key]
            if noise_key == "red":
                lab = "Achromatic Red Noise"
            elif noise_key == "gw":
                lab = "GW signal"
            elif noise_key == "dm":
                lab = "Dispersion Measure Noise"
            elif noise_key == "sw":
                lab = "Solar Wind"

                noise_df["Rounded MJD"] = noise_df["MJD"].round(decimals=4)
                sw_ave = noise_df.groupby(["Rounded MJD"])["SW_noise"].median()


                f_interp = interp1d(sw_ave.index.values, sw_ave.values)

                times_value = times_value[times_value > sw_ave.index.values.min()]
                sw_interp = f_interp(times_value)
                n_gpt = sw_interp
                

            elif noise_key == "ecorr":
                lab = "GP ECORR"
            elif noise_key == "pl_chrom_noise":
                lab = "Chromatic Noise"
            else:
                lab = noise_key

            #ax[0].plot(times, n_gpt, linewidth = 2, alpha=0.25, label=lab)
            ax[1].plot(times_value, n_gpt, linewidth = 2, alpha=0.75, label=lab, zorder=10)


    ax[1].set_title(psrname + " residuals + noise processes", fontsize=15)
    ax[1].legend(fontsize=15)
    ax[1].set_ylabel("Residuals (s)", fontsize=15)
    ax[1].set_xlabel("MJD", fontsize=15)
    ax[1].set_ylim(-2e-6, 2e-6)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[0].tick_params(axis='both', which='major', labelsize=12)

    fig1.savefig(maindir+pulsar+"_gauss_realised_noise_zoomed.png")
    fig1.clf()
    plt.clf()

for n in noise:

    ab, f, interp_region = co_coll(maindir, pulsar, n)

    np.save(maindir+"/"+pulsar+"_"+n+"_ab_coeffs.npy", ab)
    np.save(maindir+"/"+pulsar+"_"+n+"_freq_coeffs.npy", f)
    np.save(maindir+"/"+pulsar+"_"+n+"_interp_region.npy", interp_region)


    gpt = gauss_realise(maindir, pulsar, n)

    np.save(maindir+"/"+pulsar+"_"+n+"_gpt.npy", gpt)


gpt_plotter(maindir, pulsar, noise)
