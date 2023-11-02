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

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
args = parser.parse_args()
pulsar = str(args.pulsar)


def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]

    return np.average(val, weights = 1/(np.array(wt)**2))


def uncertainty_scaled(dataframe, value):
    val = dataframe[value]

    return np.sqrt(np.average(val**2, weights = 1/(np.array(val)**2))) /np.sqrt(len(val))


def ecorr_apply(dataframe,value, ecorr):
    val = dataframe[value]
    
    return np.sqrt(val**2 + ecorr**2)


def lazy_noise_reducer_ecorr_gauss(parfile, timfile):
    # Both the pulsar and the fitter object are obscenely heavy, so this code needs to garbage collect everything first otherwise it taps out
    maindir = "/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/tdb_partim_w_noise_ECORRgauss/"
    psr = Pulsar(maindir+parfile, maindir+timfile, ephem="DE440")
    
    m, t_all = get_model_and_toas(maindir+parfile, maindir+timfile, allow_name_mixing=True)
    psrname = m.name.split("/")[-1].replace("_tdb.par","")

    glsfit = pint.fitter.GLSFitter(toas=t_all, model=m)
    glsfit.fit_toas(maxiter=3)
    mjds = glsfit.toas.get_mjds()
    noise_df = pd.DataFrame.from_dict(glsfit.resids.noise_resids)
    noise_df["MJD"] = mjds.value

    
    if "SW_noise" in glsfit.resids.noise_resids.keys():
        deter_sw = solar_wind(n_earth=m.SWNEARTH.quantity)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
        sw_array = mean_sw(psr).get_delay()
        sw_pert_array = noise_df["SW_noise"].values

        sw_total = sw_array + sw_pert_array
        #sw_total = sw_pert_array
        noise_df["SW_noise"] = sw_total

    

    #fig, ax = plt.subplots(figsize=(16,9))
    if "ecorr_noise" in glsfit.resids.noise_resids.keys():
        fig1, axs= plt.subplots(2, figsize=(15,10))
        axs[0].errorbar(x=noise_df["MJD"], y=glsfit.resids.resids.value,yerr=glsfit.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals",alpha=1,zorder=1)
        for col in noise_df.columns:
            if "noise" in col:

                if col == "ecorr_noise":
                        axs[0].plot(noise_df["MJD"], noise_df[col], ".", label = col, zorder=2, alpha=0.25)
                        axs[1].plot(noise_df["MJD"], noise_df[col], ".", label = col, zorder=2, alpha=0.25)
                        mjd_vals = noise_df["MJD"].values
                        ecorr_vals = noise_df[col].values

                        np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_MJD", mjd_vals)
                        np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_ECORR", ecorr_vals)
    
    if "pl_red_noise" in glsfit.resids.noise_resids.keys():
        for col in noise_df.columns:
            if col == "pl_red_noise":
                fig2, axs1= plt.subplots(figsize=(15,5))
                axs1.plot(noise_df["MJD"], noise_df[col], ".", label = col, zorder=2, alpha=0.25)
                red_vals = noise_df[col].values
                mjd_vals = noise_df["MJD"].values
                np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_RED", red_vals)
                np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_MJD", mjd_vals)
                axs1.set_title(psrname + " Red noise")
                axs1.set_ylabel("Residuals (s)")
                axs1.set_xlabel("MJD")
                fig2.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_red_real.png")


        axs[0].set_title(psrname + " residuals + ecorr")
        axs[0].legend()
        axs[0].set_ylabel("Residuals (s)")
        axs[1].set_title(psrname + " GP ECORR")
        axs[1].legend()
        axs[1].set_ylabel("Residuals (s)")

        fig1.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/ecorr_realisations/"+psrname+"_ecorr_real.png")
        


parfile = pulsar+"_tdb.par"
timfile = pulsar+".tim"

lazy_noise_reducer_ecorr_gauss(parfile, timfile)
