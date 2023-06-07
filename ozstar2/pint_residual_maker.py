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

def lazy_noise_reducer_ecorr_gauss(parfile, timfile, sw_extract = False, gw_subtract=True):
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
    
    # Creating average version of these.
    # TBD: These should be weighted averages as well. Future Matt's problem.
    #ave_noise_dict = {}
    #for col in noise_df.columns:
    #    if "noise" in col:
    #        ave_noise_dict[col] = noise_df.groupby("MJD")[col].mean()
    
    if "SW_noise" in glsfit.resids.noise_resids.keys():
        deter_sw = solar_wind(n_earth=m.SWNEARTH.quantity)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
        sw_array = mean_sw(psr).get_delay()
        sw_pert_array = noise_df["SW_noise"].values

        sw_total = sw_array + sw_pert_array
        #sw_total = sw_pert_array
        noise_df["SW_noise"] = sw_total
    
    fig1, axs= plt.subplots(3,figsize=(16,15))
    axs[0].errorbar(x=noise_df["MJD"], y=glsfit.resids.resids.value,yerr=glsfit.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals")
    axs[0].set_title(psrname + " residuals")
    axs[0].legend()
    axs[0].set_ylabel("Residuals (s)")
    
    #fig, ax = plt.subplots(figsize=(16,9))
    axs[1].errorbar(x=noise_df["MJD"], y=glsfit.resids.resids.value,yerr=glsfit.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals",alpha=1,zorder=1)
    for col in noise_df.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            #if col != "ecorr_noise":
            if not gw_subtract:
                if col != "pl_gw_noise":
                    axs[1].plot(noise_df["MJD"], noise_df[col], ".", label = col, zorder=2, alpha=0.25)
            else:
                axs[1].plot(noise_df["MJD"], noise_df[col], ".", label = col, zorder=2, alpha=0.25)

    axs[1].set_title(psrname + " residuals + noise processes")
    axs[1].legend()
    axs[1].set_ylabel("Residuals (s)")
    
    whitened_residuals = glsfit.resids.resids.value
    for col in noise_df.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            #if col != "ecorr_noise":
            if not gw_subtract:
                if col != "pl_gw_noise":
                    whitened_residuals -= noise_df[col].values
            else:
                whitened_residuals -= noise_df[col].values
            
    #fig, ax = plt.subplots(figsize=(16,9))

    #if sw_extract == True:
    #    return noise_df[["MJD", "SW_noise"]].values
    
    uncs = glsfit.toas.get_errors().to(u.s).value
    
    
    try:
        efac = m.EFAC1.value
        #efac = 0.958275479428316
        uncs *= efac
    except:
        pass
    
    try:
        equad = m.TNEQ1.value
        equad = 10**(equad)
        uncs = np.sqrt(equad**2 + uncs**2)
    except:
        pass
    

    res_df = pd.DataFrame(np.array([mjds, whitened_residuals]).T,columns=["MJD","Noise subtracted (s)"])
    res_df["Rounded MJD"] = noise_df["MJD"].round(decimals=4)
    res_df["WN Scaled Uncertainty (s)"] = uncs
    res_df["Freqs"] = glsfit.toas.table["freq"]

    try:
        ecorr = m.TECORR1.value
        ecorr = 10**(ecorr)
        res_df["WN Scaled Uncertainty (s)"] = res_df.groupby("Rounded MJD").apply(ecorr_apply, "WN Scaled Uncertainty (s)", ecorr).values
        unc_WA = np.sqrt(ecorr**2 + unc_WA**2)
    except:
        pass
    
    num_fp = len(m.free_params)
    
    chi_square = np.sum(((whitened_residuals-np.mean(whitened_residuals))**2)/(res_df["WN Scaled Uncertainty (s)"].values**2))
    num_obs = len(whitened_residuals)
    red_chisq = chi_square / (num_obs - num_fp)
    dof = num_obs - num_fp

    p_value = 1 - stats.chi2.cdf(float(chi_square), dof)
    #os.system("echo "+str(p_value)+" >> /fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/p_values.list")
    #axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=glsfit.toas.get_errors().to(u.s).value, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=res_df["WN Scaled Uncertainty (s)"].values, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].set_title(psrname + ' whitened residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(red_chisq)+'; p-value = {:.4f}'.format(p_value)+'; (1-p) = {:.4f}'.format(1-p_value))
    axs[2].legend()
    axs[2].set_xlabel("MJD")
    axs[2].set_ylabel("Residuals (s)")

    fig1.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/clock_residuals/"+pulsar+"_residual_plot.png")
    plt.close()

    return glsfit, m, res_df, noise_df
    


parfile = pulsar+"_tdb.par"
timfile = pulsar+".tim"

glsfit, model, res_df, noise_df = lazy_noise_reducer_ecorr_gauss(parfile, timfile, sw_extract = False, gw_subtract=True)

res_df.to_pickle("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/clock_residuals/"+pulsar+"_residuals.pkl")