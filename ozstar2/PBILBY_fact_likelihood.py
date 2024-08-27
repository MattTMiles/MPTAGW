import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import corner
from scipy.stats import gaussian_kde
import os
import glob
from sklearn.neighbors import KernelDensity

import pandas as pd
import json

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/"
#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []


rebin_num =50

i=0
for pulsar in to_use:
    psrname = pulsar.strip("\n")
    print(psrname)
    try:
        psr_dir = gw_dir + "/" + psrname + "/"

        psr_SPGWC = glob.glob(gw_dir + "/" + psrname + "*SGWB/")[0]

        #result_SPGW = bilby.result.read_in_result(psr_SPGW+"/SPGW1000_ER_result.json")
        result_SPGWC = pd.read_json(json.load(open(glob.glob(psr_SPGWC+"/*final_res.json")[0])))

        #if not psrname+"_red_noise_log10_A" in result_SPGW.parameter_labels:
        #    posts_SPGW_A = result_SPGW.posterior["log10_A_gw"].values
        #    posts_SPGW_g = result_SPGW.posterior["gamma_gw"].values
        #else:
        #    posts_SPGW_A = result_SPGW.posterior[result_SPGW.posterior[psrname+"_red_noise_log10_A"] < -16.5]["log10_A_gw"].values
        #    posts_SPGW_g = result_SPGW.posterior[result_SPGW.posterior[psrname+"_red_noise_log10_A"] < -16.5]["gamma_gw"].values

        posts_SPGWC_A = result_SPGWC["log10_A_gw"].values

        #pdf_SPGW_A = np.histogram(posts_SPGW_A,bins=np.linspace(-18,-12,99),density=True)[0] + 1e-20
        #pdf_SPGW_g = np.histogram(posts_SPGW_g,bins=np.linspace(0,7,99),density=True)[0] + 1e-20
        
        pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,50),density=True)[0] + 1e-20
        #pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,24),density=True)[0] + 1e-20

        stacked.append(pdf_SPGWC_A)
        plt.figure(0)
        p, bins, patches = plt.hist(posts_SPGWC_A, bins=np.linspace(-18,-12,50), range=(-18, -12), density=True, alpha=0.6, histtype='step')

        bins_new = np.linspace(-18,-12,rebin_num)
        kdeAMP = KernelDensity(bandwidth="scott", kernel="gaussian")
        kdeAMP.fit(pdf_SPGWC_A[:, None])
        logprob = kdeAMP.score_samples(bins_new[:, None])
        kde_eval_AMP = np.exp(logprob)

        ind = np.argmax(p)
        centres = bins[0:-1] + np.diff(bins)
        print(centres[ind])


    except:
        continue
    



    if i == 0:
        p_total1 = (p + 1e-20)
        p_kde1 = (kde_eval_AMP + 1e-20)
    elif i==1:
        p_total2 = (p + 1e-20)
        p_kde2 = (kde_eval_AMP + 1e-20)
    elif i%2 == 0:
        p_total1 *= (p + 1e-20)
        p_kde1 *= (kde_eval_AMP + 1e-20)
    else:
        p_total2 *= (p + 1e-20)
        p_kde2 *= (kde_eval_AMP + 1e-20)
    


    if i==0:
        p_total = (p + 1e-20)
        p_kde = (kde_eval_AMP + 1e-20)
    else:
        p_total *= (p + 1e-20)
        p_kde *= (kde_eval_AMP + 1e-20)

    i=i+1


amps = np.linspace(-18,-12,49)

plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, rebin_num+1)
new_bindiff = bins_new_new[1] - bins_new_new[0]
bindiff = bins[1]-bins[0]

kde_pdf = p_kde/(np.sum(p_kde)*new_bindiff)

curn_peak = bins_new_new[kde_pdf.argmax()]



plt.stairs(p_total/(np.sum(p_total)*bindiff), bins, color='k', zorder=0, linewidth=2)
plt.ylim(1e-6)
plt.stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
plt.stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="Kernel Density Estimate", fill=True, alpha=0.25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.xlim(-18,-12)
plt.axvline(curn_peak, label="CURN = {}".format(curn_peak), linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/SMBHB_WN_Amp_FL.png")
plt.clf()

plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, rebin_num+1)
new_bindiff = bins_new_new[1] - bins_new_new[0]
bindiff = bins[1]-bins[0]

plt.stairs(p_total1/(np.sum(p_total1)*bindiff), bins, color='k', zorder=0, linewidth=2)
plt.ylim(1e-6)
plt.stairs(p_kde1/(np.sum(p_kde1)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
plt.stairs(p_kde1/(np.sum(p_kde1)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="Kernel Density Estimate", fill=True, alpha=0.25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.xlim(-18,-12)
plt.axvline(curn_peak, label="CURN = {}".format(curn_peak), linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/SMBHB_WN_Amp_FL_first.png")
plt.clf()


plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, rebin_num+1)
new_bindiff = bins_new_new[1] - bins_new_new[0]
bindiff = bins[1]-bins[0]

plt.stairs(p_total2/(np.sum(p_total2)*bindiff), bins, color='k', zorder=0, linewidth=2)
plt.ylim(1e-6)
plt.stairs(p_kde2/(np.sum(p_kde2)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
plt.stairs(p_kde2/(np.sum(p_kde2)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="Kernel Density Estimate", fill=True, alpha=0.25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.xlim(-18,-12)
plt.axvline(curn_peak, label="CURN = {}".format(curn_peak), linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/SMBHB_WN_Amp_FL_second.png")
plt.clf()


fig, ax = plt.subplots(2, figsize=(12,20))

ax[0].stairs(p_total/(np.sum(p_total)*bindiff), bins, color='k', zorder=0, linewidth=2)
ax[0].set_ylim(1e-4)
ax[0].stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
ax[0].stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="MPTA: Full Array", fill=True, alpha=0.25)
ax[0].tick_params(axis="both", which="both", labelsize=20)
ax[0].set_yscale("log")
ax[0].set_xlim(-18,-12)
ax[0].axvline(curn_peak, label="CURN = {}".format(curn_peak), linestyle="--", color="grey", lw=2)
ax[0].set_xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
ax[0].set_ylabel("Probability Density", fontsize=20)
ax[0].legend(fontsize=15)

ax[1].stairs(p_total1/(np.sum(p_total1)*bindiff), bins, color='k', zorder=0, linewidth=2, alpha=0.5)
ax[1].stairs(p_kde1/(np.sum(p_kde1)*new_bindiff),bins_new_new, color='tab:blue', zorder=0, linewidth=2, fill=False)
ax[1].stairs(p_kde1/(np.sum(p_kde1)*new_bindiff),bins_new_new, color='tab:blue', zorder=0, linewidth=2, label="MPTA: First half", fill=True, alpha=0.25)

ax[1].stairs(p_total2/(np.sum(p_total2)*bindiff), bins, color='k', zorder=0, linewidth=2, alpha=0.5)
ax[1].stairs(p_kde2/(np.sum(p_kde2)*new_bindiff),bins_new_new, color='tab:orange', zorder=0, linewidth=2, fill=False)
ax[1].stairs(p_kde2/(np.sum(p_kde2)*new_bindiff),bins_new_new, color='tab:orange', zorder=0, linewidth=2, label="MPTA: Second half", fill=True, alpha=0.25)

ax[1].set_ylim(1e-4)
ax[1].tick_params(axis="both", which="both", labelsize=20)
ax[1].set_yscale("log")
ax[1].set_xlim(-18,-12)
ax[1].axvline(curn_peak, label="CURN = {}".format(curn_peak), linestyle="--", color="grey", lw=2)
ax[1].set_xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
ax[1].set_ylabel("Probability Density", fontsize=20)
ax[1].legend(fontsize=15)

fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/SMBHB_WN_Amp_FL_full_and_split.png")
fig.clf()
plt.clf()
