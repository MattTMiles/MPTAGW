import numpy as np
import sys
import importlib.util
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.table import QTable, Table, Column
import pandas as pd
import astropy.time as time

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
import enterprise.constants as econst
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
parser.add_argument("-dest", dest="dest", help="Outdir for plots", required = False)
args = parser.parse_args()
pulsar = str(args.pulsar)
dest = str(args.dest)

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

def chrom_yearly_sinusoid(toas, freqs, log10_Amp, phase, idx):
    """
    Chromatic annual sinusoid.
    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid
    :param idx: index of chromatic dependence
    :return wf: delay time-series [s]
    """

    wf = 10**log10_Amp * np.sin(2 * np.pi * econst.fyr * toas + phase)
    return wf * (1400 / freqs) ** idx

def chrom_gaussian_bump(toas, freqs, log10_Amp=-2.5, sign_param=1.0,
                    t0=53890, sigma=81, idx=2):
    """
    Chromatic time-domain Gaussian delay term in TOAs.
    Example: J1603-7202 in Lentati et al, MNRAS 458, 2016.
    """
    #t0 *= const.day
    #sigma *= const.day
    wf = 10**log10_Amp * np.exp(-(toas - t0)**2/2/sigma**2)
    return np.sign(sign_param) * wf * (1400 / freqs) ** idx



def lazy_noise_reducer_ecorr_gauss(parfile, timfile, dest, sw_extract = False, gw_subtract=True):
    # Both the pulsar and the fitter object are obscenely heavy, so this code needs to garbage collect everything first otherwise it taps out
    maindir = "/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/pp_8/tdb_partim_w_noise/"
    psr = Pulsar(maindir+parfile, maindir+timfile, ephem="DE440")
    
    m, t_all = get_model_and_toas(maindir+parfile, maindir+timfile, allow_name_mixing=True, planets=True)
    psrname = m.name.split("/")[-1].replace("_tdb_noise.par","")

    if "CHROMANNUALAMP" in m.params:
        t_all.table["mjd_float"].convert_unit_to("s")
        annual_wf = chrom_yearly_sinusoid(t_all.table["mjd_float"], t_all.table["freq"], m.CHROMANNUALAMP.value, m.CHROMANNUALPHASE.value, m.CHROMANNUALIDX.value)
        deltaT = time.TimeDelta(annual_wf)
        t_all.adjust_TOAs(-deltaT)

    if "CHROMBUMPAMP" in m.params:
        t_all.table["mjd_float"].convert_unit_to("s")
        gauss_wf = chrom_gaussian_bump(t_all.table["mjd_float"], t_all.table["freq"], log10_Amp=m.CHROMBUMPAMP.value, sign_param=m.CHROMBUMPSIGN.value,t0=m.CHROMBUMPT.value, sigma=m.CHROMBUMPSIGMA.value, idx=m.CHROMBUMPIDX.value)
        deltaT = time.TimeDelta(gauss_wf)
        t_all.adjust_TOAs(-deltaT)

    if hasattr(m, "SWNEARTH"):
        m.NE_SW.set(m.SWNEARTH.value)
    else:
        m.NE_SW.set(4)

    glsfit = pint.fitter.GLSFitter(toas=t_all, model=m)
    glsfit.fit_toas(maxiter=10)
    mjds = glsfit.toas.get_mjds()
    noise_df = pd.DataFrame.from_dict(glsfit.resids.noise_resids)
    noise_df["MJD"] = mjds.value
    
    # Creating average version of these.
    # TBD: These should be weighted averages as well. Future Matt's problem.
    #ave_noise_dict = {}
    #for col in noise_df.columns:
    #    if "noise" in col:
    #        ave_noise_dict[col] = noise_df.groupby("MJD")[col].mean()
    
    # if "SW_noise" in glsfit.resids.noise_resids.keys():
    #     deter_sw = solar_wind(n_earth=m.SWNEARTH.quantity)
    #     mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
    #     sw_array = mean_sw(psr).get_delay()
    #     sw_pert_array = noise_df["SW_noise"].values

    #     sw_total = sw_array + sw_pert_array
    #     #sw_total = sw_pert_array
    #     noise_df["SW_noise"] = sw_total
    


    res_copy = glsfit.resids.resids.value.copy()

    fig1, axs= plt.subplots(3,figsize=(16,15))
    axs[0].errorbar(x=noise_df["MJD"], y=res_copy,yerr=glsfit.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals")
    axs[0].set_title(psrname + " residuals")
    axs[0].legend()
    axs[0].set_ylabel("Residuals (s)")
    
    #fig, ax = plt.subplots(figsize=(16,9))
    axs[1].errorbar(x=noise_df["MJD"], y=res_copy,yerr=glsfit.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals",alpha=1,zorder=1)
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
    
    whitened_residuals = np.copy(glsfit.resids.resids.value)
    non_whitened_res = np.copy(glsfit.resids.resids.value)

    noise_df["Rounded MJD"] = noise_df["MJD"].round(decimals=4)
    noisedf_ave = {}

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
    
    uncs = np.copy(glsfit.toas.get_errors().to(u.s).value)
    
    
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
    

    res_df = pd.DataFrame(np.array([mjds, whitened_residuals, non_whitened_res]).T,columns=["MJD","Noise subtracted (s)", "non_whitened_res"])
    res_df["Rounded MJD"] = noise_df["MJD"].round(decimals=4)
    res_df["WN Scaled Uncertainty (s)"] = uncs
    res_df["Freqs"] = glsfit.toas.table["freq"]

    try:
        ecorr = m.TECORR1.value
        print(ecorr)
        ecorr = 10**(ecorr)
        res_df["WN Scaled Uncertainty (s)"] = res_df.groupby("Rounded MJD").apply(ecorr_apply, "WN Scaled Uncertainty (s)", ecorr).values
        unc_WA = np.sqrt(ecorr**2 + unc_WA**2)
    except:
        pass
    
    noise_df["WN Scaled Uncertainty (s)"] = uncs
    noisedf_ave["Rounded MJD"] = np.unique(noise_df["MJD"].round(decimals=4))
    mjd_round = noisedf_ave["Rounded MJD"]
    np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/pp_8/noise_reals/"+psrname+"_mjd_round.npy", mjd_round)
    noisedf_ave["unc ave"] = noise_df.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")
    ave_unc = noisedf_ave["unc ave"].values
    np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/pp_8/noise_reals/"+psrname+"_unc_ave.npy", ave_unc)
    print(noise_df.columns)
    for col in noise_df.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":

            noisedf_ave[col+" ave"] = noise_df.groupby("Rounded MJD").apply(weighted_average, col, "WN Scaled Uncertainty (s)")
            if "gw" in col:
                gw_real = noisedf_ave[col+" ave"].values

                np.save("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/pp_8/noise_reals/"+psrname+"_gw_ave_real.npy", gw_real)


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

    fig1.savefig(dest+"/"+pulsar+"_residual_plot.png")
    plt.close()

    ws = res_df["Noise subtracted (s)"]/res_df["WN Scaled Uncertainty (s)"].values
    ws = ws - np.median(ws)
    ws_std_full = np.std(ws)
    res_ws = anderson(ws.astype(np.float64))

    fig, ax_4 = plt.subplots()
    ws.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std_full), label = "ADS: "+format(res_ws.statistic),legend=True, ax=ax_4)
    fig = ax_4.get_figure()
    fig.savefig(dest+"/"+pulsar+"_anderson_check.png")

    fighist, axhist = plt.subplots()
    axhist.hist(ws,bins=100)
    fighist.savefig(dest+"/"+pulsar+"_norm_hist.png")

    res_WA = res_df.groupby("Rounded MJD").apply(weighted_average, "Noise subtracted (s)", "WN Scaled Uncertainty (s)")
    unc_WA = res_df.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")
    nonres_WA = res_df.groupby("Rounded MJD").apply(weighted_average, "non_whitened_res", "WN Scaled Uncertainty (s)")


    ave_chi_square = np.sum(((res_WA-np.mean(res_WA))**2)/(unc_WA**2))
    ave_num_obs = len(res_WA)
    ave_red_chisq = ave_chi_square / (ave_num_obs - num_fp)
    ave_dof = ave_num_obs - num_fp

    ave_p_value = 1 - stats.chi2.cdf(float(ave_red_chisq), ave_dof)
    
    unique_mjd_rounded = np.array(sorted(list(set(res_df["Rounded MJD"].values))))

    weights = 1/(unc_WA.values**2)
    xbar = np.sum(nonres_WA*weights)/np.sum(weights)
    non_rms = np.sqrt(np.sum((weights)*((nonres_WA-xbar)**2))/np.sum(weights))*10**6

    res_WA = res_WA - np.mean(res_WA)
    fig2, ax2 = plt.subplots(figsize=(16,5))
    ax2.errorbar(x=unique_mjd_rounded, y=res_WA, yerr=unc_WA, linestyle="", marker=".", label="Frequency averaged; Noise removed")
    ax2.legend()
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Residuals (s)")
    fig2.savefig(dest+"/"+pulsar+"_ave_residual_plot.png")
    plt.close()

    figres, axres = plt.subplots(figsize=(15,5))
    axres.errorbar(x=unique_mjd_rounded, y=(nonres_WA-np.mean(nonres_WA))*1e6, yerr=unc_WA.values*1e6, linestyle="", marker=".", label = r"RMS: {:.2f} $\mu$s".format(non_rms))
    #ax2.set_title(pulsar+' whitened averaged residuals; '+r'$\chi^2_{red}$'+'= {:.2f}'.format(ave_red_chisq)+'; p-value = {:.4f}'.format(ave_p_value)+'; (1-p) = {:.4f}'.format(1-ave_p_value))
    axres.legend(fontsize=15)
    axres.set_xlabel("MJD", fontsize=15)
    axres.set_ylabel(r"Residuals ($\mu$s)", fontsize=15)
    axres.tick_params(labelsize=15)
    figres.savefig(dest+"/"+pulsar+"_uncorrected.png")

    white_scaled = res_WA/unc_WA.values
    white_scaled = white_scaled - np.mean(white_scaled)

    ws_std = np.std(white_scaled)

    res = anderson(white_scaled.values.astype(np.float64))
    
    fig, ax_new = plt.subplots()
    white_scaled.plot(style=".",figsize=(15,5), title=pulsar+"; std dev: "+format(ws_std), label = "ADS: "+format(res.statistic),legend=True, ax=ax_new)
    fig = ax_new.get_figure()
    fig.savefig(dest+"/"+pulsar+"_ave_anderson_check.png")



    return glsfit, m, res_df, noise_df
    


parfile = pulsar+"_tdb_noise.par"
timfile = pulsar+".tim"

glsfit, model, res_df, noise_df = lazy_noise_reducer_ecorr_gauss(parfile, timfile, dest, sw_extract = False, gw_subtract=True)

