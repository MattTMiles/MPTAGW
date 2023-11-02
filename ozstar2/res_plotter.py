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

def lazy_noise_reducer(parfile, timfile, sw_extract = False, gw_subtract=True):
    # Both the pulsar and the fitter object are obscenely heavy, so this code needs to garbage collect everything first otherwise it taps out
    maindir = "/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/tdb_partim_w_noise/"
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

        #sw_total = sw_array + sw_pert_array
        sw_total = sw_pert_array
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
    

    res_WA = res_df.groupby("Rounded MJD").apply(weighted_average, "Noise subtracted (s)", "WN Scaled Uncertainty (s)")
    unc_WA = res_df.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")

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
    os.system("echo "+str(p_value)+" >> /fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/p_values.list")
    #axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=glsfit.toas.get_errors().to(u.s).value, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=res_df["WN Scaled Uncertainty (s)"].values, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].set_title(psrname + ' whitened residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(red_chisq)+'; p-value = {:.4f}'.format(p_value)+'; (1-p) = {:.4f}'.format(1-p_value))
    axs[2].legend()
    axs[2].set_xlabel("MJD")
    axs[2].set_ylabel("Residuals (s)")

    fig1.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/"+pulsar+"_residual_plot.png")
    plt.close()
    
    ws = res_df["Noise subtracted (s)"]/res_df["WN Scaled Uncertainty (s)"].values
    ws = ws - np.mean(ws)
    ws_std_full = np.std(ws)
    res_ws = anderson(ws.astype(np.float64))

    fig, ax_4 = plt.subplots()
    ws.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std_full), label = "ADS: "+format(res_ws.statistic),legend=True, ax=ax_4)
    fig = ax_4.get_figure()
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/"+pulsar+"_anderson_check.png")


    ave_chi_square = np.sum(((res_WA-np.mean(res_WA))**2)/(unc_WA**2))
    ave_num_obs = len(res_WA)
    ave_red_chisq = ave_chi_square / (ave_num_obs - num_fp)
    ave_dof = ave_num_obs - num_fp

    ave_p_value = 1 - stats.chi2.cdf(float(ave_red_chisq), ave_dof)
    
    unique_mjd_rounded = np.array(sorted(list(set(res_df["Rounded MJD"].values))))
    
    fig, ax2 = plt.subplots(figsize=(15,5))
    ax2.errorbar(x=unique_mjd_rounded, y=res_WA-np.mean(res_WA), yerr=unc_WA.values, linestyle="", marker=".", label = "Whitened Averaged Residuals")
    ax2.set_title(pulsar+' whitened averaged residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(ave_red_chisq)+'; p-value = {:.4f}'.format(ave_p_value)+'; (1-p) = {:.4f}'.format(1-ave_p_value))
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Residuals (s)")

    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/"+pulsar+"_white_ave.png")

    white_scaled = res_WA/unc_WA.values
    white_scaled = white_scaled - np.mean(white_scaled)

    ws_std = np.std(white_scaled)

    res = anderson(white_scaled.values.astype(np.float64))
    
    fig, ax_new = plt.subplots()
    white_scaled.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std), label = "ADS: "+format(res.statistic),legend=True, ax=ax_new)
    fig = ax_new.get_figure()
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/"+pulsar+"_ave_anderson_check.png")

    return glsfit, m, res_df, noise_df, res_ws.statistic, ws_std_full
    

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
    
    
    #fig_gw, ax_gw = plt.subplots(figsize=(16,5))
    #ax_gw.plot(noise_df["MJD"], noise_df["pl_gw_noise"], ".", label = "pl_gw_noise", zorder=2, alpha=0.25)
    #ax_gw.set_title(psrname+" GW realisation")
    #ax_gw.legend()
    #ax_gw.set_ylabel("Residuals (s)")
    #fig_gw.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/"+pulsar+"_GW_plot.png")

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
    

    res_WA = res_df.groupby("Rounded MJD").apply(weighted_average, "Noise subtracted (s)", "WN Scaled Uncertainty (s)")
    unc_WA = res_df.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")

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
    os.system("echo "+str(p_value)+" >> /fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/p_values.list")
    #axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=glsfit.toas.get_errors().to(u.s).value, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].errorbar(x=noise_df["MJD"], y=whitened_residuals, yerr=res_df["WN Scaled Uncertainty (s)"].values, linestyle="", marker=".", label = "Whitened Residuals")
    axs[2].set_title(psrname + ' whitened residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(red_chisq)+'; p-value = {:.4f}'.format(p_value)+'; (1-p) = {:.4f}'.format(1-p_value))
    axs[2].legend()
    axs[2].set_xlabel("MJD")
    axs[2].set_ylabel("Residuals (s)")

    fig1.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/"+pulsar+"_residual_plot.png")
    plt.close()
    
    ws = res_df["Noise subtracted (s)"]/res_df["WN Scaled Uncertainty (s)"].values
    ws = ws - np.mean(ws)
    ws_std_full = np.std(ws)
    res_ws = anderson(ws.astype(np.float64))

    fig, ax_4 = plt.subplots()
    ws.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std_full), label = "ADS: "+format(res_ws.statistic),legend=True, ax=ax_4)
    fig = ax_4.get_figure()
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/"+pulsar+"_anderson_check.png")


    ave_chi_square = np.sum(((res_WA-np.mean(res_WA))**2)/(unc_WA**2))
    ave_num_obs = len(res_WA)
    ave_red_chisq = ave_chi_square / (ave_num_obs - num_fp)
    ave_dof = ave_num_obs - num_fp

    ave_p_value = 1 - stats.chi2.cdf(float(ave_red_chisq), ave_dof)
    
    unique_mjd_rounded = np.array(sorted(list(set(res_df["Rounded MJD"].values))))
    
    fig, ax2 = plt.subplots(figsize=(15,5))
    ax2.errorbar(x=unique_mjd_rounded, y=res_WA-np.mean(res_WA), yerr=unc_WA.values, linestyle="", marker=".", label = "Whitened Averaged Residuals")
    ax2.set_title(pulsar+' whitened averaged residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(ave_red_chisq)+'; p-value = {:.4f}'.format(ave_p_value)+'; (1-p) = {:.4f}'.format(1-ave_p_value))
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Residuals (s)")

    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/"+pulsar+"_white_ave.png")

    white_scaled = res_WA/unc_WA.values
    white_scaled = white_scaled - np.mean(white_scaled)

    ws_std = np.std(white_scaled)

    res = anderson(white_scaled.values.astype(np.float64))
    
    fig, ax_new = plt.subplots()
    white_scaled.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std), label = "ADS: "+format(res.statistic),legend=True, ax=ax_new)
    fig = ax_new.get_figure()
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/"+pulsar+"_ave_anderson_check.png")

    return glsfit, m, res_df, noise_df, res_ws.statistic, ws_std_full
    



parfile = pulsar+"_tdb.par"
timfile = pulsar+".tim"

glsfit, model, res_df, noise_df, ads, ws_std = lazy_noise_reducer_ecorr_gauss(parfile, timfile, sw_extract = False, gw_subtract=True)


if ads > 2.49: 
    print(pulsar+" does not pass gaussianity check")
    os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/")
    os.system("touch FAILED_ADS")
else:
    if 0.75 < ws_std < 0.9:
        print(pulsar+" passes gaussianity check with low standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch LOW_STD_DEV")
    elif ws_std < 0.75:
        print(pulsar+" passes gaussianity check with extremely low standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch VERY_LOW_STD_DEV")
    elif ws_std > 1.25:
        print(pulsar+" passes gaussianity check with very high standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch VERY_HIGH_STD_DEV")
    else:
        print(pulsar+" passes gaussianity check")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_gaussian_process/"+pulsar+"/")
        os.system("touch PASSED_ADS")