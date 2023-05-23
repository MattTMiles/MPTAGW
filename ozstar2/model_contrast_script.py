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
    psr1 = Pulsar(maindir+parfile, maindir+timfile, ephem="DE440")
    psr2 = Pulsar(maindir+parfile_misspec, maindir+timfile, ephem="DE440")
    m1, t_all = get_model_and_toas(maindir+parfile, maindir+timfile, allow_name_mixing=True)
    m2, t_all = get_model_and_toas(maindir+parfile_misspec, maindir+timfile, allow_name_mixing=True)
    psrname = m1.name.split("/")[-1].replace("_tdb.par","")

    glsfit1 = pint.fitter.GLSFitter(toas=t_all, model=m1)
    glsfit1.fit_toas(maxiter=3)
    glsfit2 = pint.fitter.GLSFitter(toas=t_all, model=m2)
    glsfit2.fit_toas(maxiter=3)
    
    mjds = glsfit1.toas.get_mjds()
    noise_df1 = pd.DataFrame.from_dict(glsfit1.resids.noise_resids)
    noise_df1["MJD"] = mjds.value
    noise_df1["Residuals"] = glsfit1.resids.resids.value
    noise_df2 = pd.DataFrame.from_dict(glsfit2.resids.noise_resids)
    noise_df2["MJD"] = mjds.value
    noise_df2["Residuals"] = glsfit1.resids.resids.value
    # Creating average version of these.
    # TBD: These should be weighted averages as well. Future Matt's problem.
    #ave_noise_dict = {}
    #for col in noise_df.columns:
    #    if "noise" in col:
    #        ave_noise_dict[col] = noise_df.groupby("MJD")[col].mean()
    
    if "SW_noise" in glsfit1.resids.noise_resids.keys():
        deter_sw = solar_wind(n_earth=m1.SWNEARTH.quantity)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
        sw_array = mean_sw(psr1).get_delay()
        sw_pert_array = noise_df1["SW_noise"].values

        sw_total = sw_array + sw_pert_array
        noise_df1["SW_noise"] = sw_total
    
    if "SW_noise" in glsfit2.resids.noise_resids.keys():
        deter_sw = solar_wind(n_earth=m2.SWNEARTH.quantity)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')
        sw_array = mean_sw(psr2).get_delay()
        sw_pert_array = noise_df2["SW_noise"].values

        sw_total = sw_array + sw_pert_array
        noise_df2["SW_noise"] = sw_total
    
    font = 15
    
    fig1, axs= plt.subplots(3,figsize=(16,15))
    axs[0].errorbar(x=noise_df1["MJD"], y=noise_df1["Residuals"],yerr=glsfit1.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals")
    axs[0].set_title(psrname + " residuals",fontsize=font)
    axs[0].legend()
    axs[0].set_ylabel("Residuals (s)")
    
    #fig, ax = plt.subplots(figsize=(16,9))
    noise_df1["Rounded MJD"] = noise_df1["MJD"].round(decimals=4)
    noise_df2["Rounded MJD"] = noise_df2["MJD"].round(decimals=4)
    uncs = glsfit1.toas.get_errors().to(u.s).value
    try:
        efac = m1.EFAC1.value
        #efac = 0.958275479428316
        uncs *= efac
    except:
        pass
    
    try:
        equad = m1.TNEQ1.value
        equad = 10**(equad)
        uncs = np.sqrt(equad**2 + uncs**2)
    except:
        pass
    noise_df1["Scaled_unc"] = uncs
    noise_df2["Scaled_unc"] = uncs
    noisedf_ave1 ={}
    axs[1].errorbar(x=noise_df1["MJD"], y=glsfit1.resids.resids.value,yerr=glsfit1.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals",alpha=0.2,zorder=1)
    for col in noise_df1.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        if "chrom" in col:
                            lab = "Chromatic Noise"
                            clr = "tab:orange"
                        elif "DM" in col:
                            lab = "DM Noise"
                            clr = "tab:green"
                        elif "red" in col:
                            lab = "Red Noise"
                            clr = "tab:red"
                        elif "SW" in col:
                            lab = "Solar Wind"
                            clr = "tab:brown"
                        else:
                            lab = col
                            clr = "tab:purple"
                        axs[1].plot(noise_df1["MJD"], noise_df1[col], ".", label = lab, zorder=2, alpha=0.4, color=clr)
                        noisedf_ave1[col+" ave"] = noise_df1.groupby("Rounded MJD").apply(weighted_average, col, "Scaled_unc")
                else:
                    axs[1].plot(noise_df1["MJD"], noise_df1[col], ".", label = col, zorder=2, alpha=0.4)
    
    axs[1].set_title(psrname + " residuals + correct noise", fontsize=font)
    axs[1].legend()
    axs[1].set_ylabel("Residuals (s)")

    axs[2].errorbar(x=noise_df2["MJD"], y=glsfit2.resids.resids.value,yerr=glsfit2.toas.get_errors().to(u.s).value,linestyle="",marker=".", label = "Residuals",alpha=0.2,zorder=1)
    noisedf_ave2 = {}
    for col in noise_df2.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        if "chrom" in col:
                            lab = "Chromatic Noise"
                            clr = "tab:orange"
                        elif "dm" in col:
                            lab = "DM Noise"
                            clr = "tab:green"
                        elif "red" in col:
                            lab = "Red Noise"
                            clr = "tab:red"
                        elif "SW" in col:
                            lab = "Solar Wind"
                            clr = "tab:brown"
                        else:
                            lab = col
                            clr = "tab:purple"
                        axs[2].plot(noise_df2["MJD"], noise_df2[col], ".", label = lab, zorder=2, alpha=0.4,color=clr)
                        noisedf_ave2[col+" ave"] = noise_df2.groupby("Rounded MJD").apply(weighted_average, col, "Scaled_unc")
                else:
                    axs[2].plot(noise_df2["MJD"], noise_df2[col], ".", label = col, zorder=2, alpha=0.4)
    
    axs[2].set_title(psrname + " residuals + incorrect noise", fontsize = font)
    axs[2].legend(fontsize=font)
    axs[2].set_ylabel("Residuals (s)")
    axs[0].xaxis.get_label().set_fontsize(font)
    axs[0].yaxis.get_label().set_fontsize(font)
    axs[1].xaxis.get_label().set_fontsize(font)
    axs[1].yaxis.get_label().set_fontsize(font)
    axs[2].xaxis.get_label().set_fontsize(font)
    axs[2].yaxis.get_label().set_fontsize(font)
    fig1.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/misspec_comparisons/"+pulsar+"/"+pulsar+"_misspec_plot.png")
    plt.close()
    #noisedf_ave1["Residuals ave"] = 


    whitened_residuals1 = glsfit1.resids.resids.value
    for col in noise_df1.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        whitened_residuals1 -= noise_df1[col].values
                else:
                    whitened_residuals1 -= noise_df1[col].values
    
    whitened_residuals2 = glsfit2.resids.resids.value
    for col in noise_df2.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        whitened_residuals2 -= noise_df2[col].values
                else:
                    whitened_residuals2 -= noise_df2[col].values


    res_df1 = pd.DataFrame(np.array([mjds, whitened_residuals1]).T,columns=["MJD","Noise subtracted (s)"])
    res_df1["Rounded MJD"] = noise_df1["MJD"].round(decimals=4)
    res_df1["WN Scaled Uncertainty (s)"] = uncs

    res_df2 = pd.DataFrame(np.array([mjds, whitened_residuals2]).T,columns=["MJD","Noise subtracted (s)"])
    res_df2["Rounded MJD"] = noise_df2["MJD"].round(decimals=4)
    res_df2["WN Scaled Uncertainty (s)"] = uncs


    res_WA1 = res_df1.groupby("Rounded MJD").apply(weighted_average, "Noise subtracted (s)", "WN Scaled Uncertainty (s)")
    unc_WA1 = res_df1.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")

    res_WA2 = res_df2.groupby("Rounded MJD").apply(weighted_average, "Noise subtracted (s)", "WN Scaled Uncertainty (s)")
    unc_WA2 = res_df2.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")

    #noise_df1["Residuals"] = glsfit1.resids.resids.value
    #noise_df2["Residuals"] = glsfit2.resids.resids.value

    noisedf_ave1["Residuals ave"] = noise_df1.groupby("Rounded MJD").apply(weighted_average, "Residuals", "Scaled_unc")
    noisedf_ave2["Residuals ave"] = noise_df2.groupby("Rounded MJD").apply(weighted_average, "Residuals", "Scaled_unc")
    unc_WA = res_df1.groupby("Rounded MJD").apply(uncertainty_scaled, "WN Scaled Uncertainty (s)")
    noisedf_ave1["unc ave"] = unc_WA
    noisedf_ave2["unc ave"] = unc_WA


    noisedf_ave1 = pd.DataFrame.from_dict(noisedf_ave1)
    noisedf_ave2 = pd.DataFrame.from_dict(noisedf_ave2)

    unique_mjd_rounded = np.array(sorted(list(set(noise_df1["Rounded MJD"].values))))

    fig2, axs2 = plt.subplots(3,figsize=(16,15))

    axs2[0].errorbar(x=unique_mjd_rounded, y=noisedf_ave2["Residuals ave"].values*1e6,yerr=noisedf_ave2["unc ave"].values*1e6,linestyle="",marker=".", label = "Residuals")
    axs2[0].set_title(psrname + ": frequency averaged residuals", fontsize=font)
    axs2[0].legend(fontsize=font)
    axs2[0].set_ylabel("Residuals ($\mu$s)")

    axs2[1].errorbar(x=unique_mjd_rounded, y=(noisedf_ave2["Residuals ave"].values - noisedf_ave2["pl_DM_noise ave"].values)*1e6,yerr=noisedf_ave2["unc ave"].values*1e6,linestyle="",marker=".", label = "Residuals")
    axs2[1].set_title(psrname + ": incorrect realisation (DM noise subtracted)", fontsize=font)

    for col in noisedf_ave2.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        if "chrom" in col:
                            lab = "Chromatic Noise"
                            clr = "tab:orange"
                        elif "DM" in col:
                            lab = "DM Noise"
                            clr = "tab:green"
                        elif "red" in col:
                            lab = "Red noise"
                            clr = "tab:red"
                        elif "SW" in col:
                            lab = "Solar Wind"
                            clr = "tab:brown"
                        else:
                            lab = col
                            clr = "tab:purple"
                        #axs2[1].plot(unique_mjd_rounded, noisedf_ave2[col], ".", label = lab, zorder=2, alpha=0.7,color=clr)

    axs2[1].legend(fontsize=font)
    axs2[1].set_ylabel("Residuals ($\mu$s)")

    axs2[2].errorbar(x=unique_mjd_rounded, y=(res_WA1-np.mean(res_WA1))*1e6, yerr=unc_WA1.values*1e6, linestyle="",marker=".", label = "Residuals")
    axs2[2].set_title(psrname + ": correct realisation (DM and Chromatic noise subtracted)",fontsize=font)
    axs2[0].set_ylim(-6,6)
    axs2[1].set_ylim(-6,6)
    axs2[2].set_ylim(-6,6)
    for col in noisedf_ave1.columns:
        if "noise" in col:
            #if col != "ecorr_noise" and  col != "SW_noise":
            if col != "ecorr_noise":
                if not gw_subtract:
                    if col != "pl_gw_noise":
                        if "chrom" in col:
                            lab = "Chromatic Noise"
                            clr = "tab:orange"
                        elif "DM" in col:
                            lab = "DM Noise"
                            clr = "tab:green"
                        elif "red" in col:
                            lab = "Red noise"
                            clr = "tab:red"
                        elif "SW" in col:
                            lab = "Solar Wind"
                            clr = "tab:brown"
                        else:
                            lab = col
                            clr = "tab:purple"
                        #axs2[2].plot(unique_mjd_rounded, noisedf_ave1[col], ".", label = lab, zorder=2, alpha=0.7,color=clr)

    axs2[2].legend(fontsize=font)
    axs2[2].set_ylabel("Residuals ($\mu$s)")
    axs2[2].set_xlabel("MJD")
    axs2[0].xaxis.get_label().set_fontsize(font)
    axs2[0].yaxis.get_label().set_fontsize(font)
    axs2[1].xaxis.get_label().set_fontsize(font)
    axs2[1].yaxis.get_label().set_fontsize(font)
    axs2[2].xaxis.get_label().set_fontsize(font)
    axs2[2].yaxis.get_label().set_fontsize(font)
    axs2[0].tick_params(axis="both", labelsize=font)
    axs2[1].tick_params(axis="both", labelsize=font)
    axs2[2].tick_params(axis="both", labelsize=font)



    fig2.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/misspec_comparisons/"+pulsar+"/"+pulsar+"_averaged_misspec_plot.png")
    plt.close()
    
    res_noISM = noisedf_ave2["Residuals ave"].values - noisedf_ave2["pl_DM_noise ave"].values - noisedf_ave2["SW_noise ave"].values
    weights = 1/(noisedf_ave2["unc ave"].values**2)
    xbar = np.sum(res_noISM*weights)/np.sum(weights)

    wrms = np.sqrt(np.sum((weights)*((res_noISM-xbar)**2))/np.sum(weights))*10**6
    fig3, axs3 = plt.subplots(figsize=(15,5))
    axs3.errorbar(x=unique_mjd_rounded, y=res_noISM*1e6,yerr=noisedf_ave2["unc ave"].values*1e6,linestyle="",marker=".", label = "$\sigma_\mathrm{RMS}$ = %.2f $\mu$s" %wrms)
    axs3.set_ylabel("Residuals ($\mu$s)")
    axs3.set_xlabel("MJD")
    axs3.set_title(psrname + ": corrected for ISM effects", fontsize=font)
    axs3.legend(fontsize=font)
    axs3.tick_params(axis="both", labelsize=font)
    axs3.xaxis.get_label().set_fontsize(font)
    axs3.yaxis.get_label().set_fontsize(font)
    fig3.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/misspec_comparisons/"+pulsar+"/"+pulsar+"_ISM_corrected.png")

    res_nonoise = noisedf_ave2["Residuals ave"].values - noisedf_ave2["pl_DM_noise ave"].values - noisedf_ave2["SW_noise ave"].values - noisedf_ave2["pl_red_noise ave"].values
    weights2 = 1/(noisedf_ave2["unc ave"].values**2)
    xbar2 = np.sum(res_nonoise*weights2)/np.sum(weights2)

    wrms2 = np.sqrt(np.sum((weights2)*((res_nonoise-xbar2)**2))/np.sum(weights2))*10**6
    fig4, axs4 = plt.subplots(figsize=(15,5))
    axs4.errorbar(x=unique_mjd_rounded, y=res_nonoise*1e6,yerr=noisedf_ave2["unc ave"].values*1e6,linestyle="",marker=".", label = "$\sigma_\mathrm{RMS}$ = %.3f $\mu$s" %wrms2)
    axs4.set_ylabel("Residuals ($\mu$s)")
    axs4.set_xlabel("MJD")
    axs4.set_title(psrname + ": corrected for all noise inc. SGWB", fontsize =font)
    axs4.legend(fontsize=font)
    axs4.tick_params(axis="both", labelsize=font)
    axs4.xaxis.get_label().set_fontsize(font)
    axs4.yaxis.get_label().set_fontsize(font)
    fig4.savefig("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/misspec_comparisons/"+pulsar+"/"+pulsar+"_all_corrected.png")
    


    '''

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
    '''

    '''
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
    '''
    return glsfit1, glsfit2, m1, m2, noise_df1, noise_df2, noisedf_ave1, noisedf_ave2
    

   
parfile = pulsar+"_tdb.par"
parfile_misspec = pulsar+"_tdb_misspec.par"
timfile = pulsar+".tim"

glsfit1, glsfit2, model1, model2, noise_df1, noise_df2, noisedf_ave1, noisedf_ave2 = lazy_noise_reducer(parfile, timfile, sw_extract = False, gw_subtract=False)
'''


if ads > 2.49: 
    print(pulsar+" does not pass gaussianity check")
    os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/")
    os.system("touch FAILED_ADS")
else:
    if 0.75 < ws_std < 0.9:
        print(pulsar+" passes gaussianity check with low standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch LOW_STD_DEV")
    elif ws_std < 0.75:
        print(pulsar+" passes gaussianity check with extremely low standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch VERY_LOW_STD_DEV")
    elif ws_std > 1.25:
        print(pulsar+" passes gaussianity check with very high standard dev")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/")
        os.system("touch PASSED_ADS")
        os.system("touch VERY_HIGH_STD_DEV")
    else:
        print(pulsar+" passes gaussianity check")
        os.chdir("/fred/oz002/users/mmiles/MPTA_GW/gaussianity_checks/Pulsar_checks/ECORR_after_averaging/"+pulsar+"/")
        os.system("touch PASSED_ADS")
'''