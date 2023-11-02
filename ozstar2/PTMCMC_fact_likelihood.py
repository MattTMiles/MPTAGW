import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import corner
from scipy.stats import gaussian_kde
import os
import glob
from sklearn.neighbors import KernelDensity

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/"
#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []


i=0
for pulsar in to_use:
    psrname = pulsar.strip("\n")
    print(psrname)
    try:
        psr_dir = gw_dir + "/" + psrname + "/"

        psr_gw_chains = []

        for spgwc_dir in glob.glob(psr_dir+"/SPGWC*"):
            if "WIDE" not in spgwc_dir:
                if "ER" in spgwc_dir:
                    likelihood = os.popen("cat "+spgwc_dir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
                    if not likelihood == "-inf":
                        chain_i = np.loadtxt(spgwc_dir+"/chain_1.txt").squeeze()
                        lenchain = len(chain_i)
                        b_chain_i = chain_i[int(0.5*lenchain):,:]
                        b_chain_gw = b_chain_i[:,-6]
                        psr_gw_chains.append(b_chain_gw)

        psr_gw_array = np.hstack(psr_gw_chains)


        p, bins, patches = plt.hist(psr_gw_array, bins=50, range=(-18, -12), density=True, alpha=0.6, histtype='step')

        bins_new = np.linspace(-18,-12,1000)
        kdeAMP = KernelDensity(bandwidth="scott", kernel="gaussian")
        kdeAMP.fit(psr_gw_array[:, None])
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


amps = np.linspace(-18,-12,99)

plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, 1001)
new_bindiff = bins_new_new[1] - bins_new_new[0]
bindiff = bins[1]-bins[0]


plt.stairs(p_total/(np.sum(p_total)*bindiff), bins, color='k', zorder=0, linewidth=2)
plt.ylim(1e-6)
plt.stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
plt.stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="Kernel Density Estimate", fill=True, alpha=0.25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.xlim(-18,-12)
plt.axvline(-14.29, label="CURN = -14.29", linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Amp_FL_ER.png")
plt.clf()

plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, 1001)
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
plt.axvline(-14.29, label="CURN = -14.29", linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Amp_FL_ER_first.png")
plt.clf()


plt.figure(figsize=(12,10))

bins_new_new = np.linspace(-18,-12, 1001)
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
plt.axvline(-14.29, label="CURN = -14.29", linestyle="--", color="grey", lw=2)
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.legend(fontsize=15)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Amp_FL_ER_second.png")
plt.clf()


fig, ax = plt.subplots(2, figsize=(12,20))

ax[0].stairs(p_total/(np.sum(p_total)*bindiff), bins, color='k', zorder=0, linewidth=2)
ax[0].set_ylim(1e-4)
ax[0].stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, fill=False)
ax[0].stairs(p_kde/(np.sum(p_kde)*new_bindiff),bins_new_new, color='tab:green', zorder=0, linewidth=2, label="MPTA: Full Array", fill=True, alpha=0.25)
ax[0].tick_params(axis="both", which="both", labelsize=20)
ax[0].set_yscale("log")
ax[0].set_xlim(-18,-12)
ax[0].axvline(-14.29, label="CURN = -14.29", linestyle="--", color="grey", lw=2)
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
ax[1].axvline(-14.29, label="CURN = -14.29", linestyle="--", color="grey", lw=2)
ax[1].set_xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CURN, FL}$", fontsize=20)
ax[1].set_ylabel("Probability Density", fontsize=20)
ax[1].legend(fontsize=15)

fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Amp_FL_ER_full_and_split.png")
