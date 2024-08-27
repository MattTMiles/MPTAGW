import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import corner
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.neighbors import KernelDensity
import os
import seaborn as sns

mpta_dir = "//fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/"

psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"

partim = "/fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/"

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []

nbins = 20

i20=0
j20=0

i40=0
j40=0

i60=0
j60=0

i80=0
j80=0

i100=0
j100=0

iother=0
jother=0

for pulsar in to_use:
    psrname = pulsar.strip("\n")
    if psrname != "J00000":
        print(psrname)
        try:
            psr_dir = mpta_dir + "/" + psrname

            dm_value = float(os.popen("grep 'DM ' " + partim + "/" +psrname + ".par | awk '{print $(2)}'").read().strip("\n"))

            #result_psr = bilby.result.read_in_result(psr_dir+"/MPTA_PM_result.json")

            #result_psr = np.load(psr_dir+"/"+psrname+"_burnt_chain.npy")
            result_psr = np.load(psr_dir+"/thinned_chain_combined.npy")
            newxbins2d = np.linspace(-18,-11,nbins*100)
            newybins2d = np.linspace(0,7,nbins*100)

            pars = list(open(psr_dir+"/pars.txt").readlines())
            pars = [ p.strip("\n") for p in pars ]

            if psrname+"_dm_gp_log10_A" in pars:
                print(psrname+" has DM noise")
                
                amp_index = pars.index(psrname+"_dm_gp_log10_A")
                gamma_index = pars.index(psrname+"_dm_gp_gamma")

                posts_DM_amp = result_psr[:, amp_index]
                posts_DM_gamma = result_psr[:, gamma_index]

                pDMA, binsDMA, patchesDMA = plt.hist(posts_DM_amp, bins=nbins, range=(-18, -11), density=True, alpha=0.6, histtype='step')
                pDMG, binsDMG, patchesDMG = plt.hist(posts_DM_gamma, bins=nbins, range=(0, 7), density=True, alpha=0.6, histtype='step')
            

                kdeDMAMP = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeDMAMP.fit(posts_DM_amp[:, None])
                logprob = kdeDMAMP.score_samples(newxbins2d[:, None])
                kde_eval_DMAMP = np.exp(logprob)

                #kde_eval_DMAMP = pDMA

                kdeDMGAM = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeDMGAM.fit(posts_DM_gamma[:, None])
                logprob = kdeDMGAM.score_samples(newybins2d[:, None])
                kde_eval_DMGAM = np.exp(logprob)

                #kde_eval_DMGAM = pDMG

                if dm_value < 20:
                    if i20==0:

                        p_totalDMamp20 = (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam20 = (kde_eval_DMGAM + 1e-20)
                    else:

                        p_totalDMamp20 *= (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam20 *= (kde_eval_DMGAM + 1e-20)

                    i20=i20+1

                if 20 < dm_value < 40:
                    if i40==0:
                        # p_totalDMamp = (pDMA + 1e-20)
                        # p_totalDMgam = (pDMG + 1e-20)
                        # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp40 = (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam40 = (kde_eval_DMGAM + 1e-20)
                    else:
                        # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp40 *= (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam40 *= (kde_eval_DMGAM + 1e-20)
                        # p_totalDMamp *= (pDMA + 1e-20)
                        # p_totalDMgam *= (pDMG + 1e-20)
                    i40=i40+1

                if 40 < dm_value < 60:
                    if i60==0:
                        # p_totalDMamp = (pDMA + 1e-20)
                        # p_totalDMgam = (pDMG + 1e-20)
                        # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp60 = (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam60 = (kde_eval_DMGAM + 1e-20)
                    else:
                        # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp60 *= (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam60 *= (kde_eval_DMGAM + 1e-20)
                        # p_totalDMamp *= (pDMA + 1e-20)
                        # p_totalDMgam *= (pDMG + 1e-20)
                    i60=i60+1

                if 60 < dm_value < 80:
                    if i80==0:
                        # p_totalDMamp = (pDMA + 1e-20)
                        # p_totalDMgam = (pDMG + 1e-20)
                        # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp80 = (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam80 = (kde_eval_DMGAM + 1e-20)
                    else:
                        # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                        p_totalDMamp80 *= (kde_eval_DMAMP + 1e-20)
                        p_totalDMgam80 *= (kde_eval_DMGAM + 1e-20)
                        # p_totalDMamp *= (pDMA + 1e-20)
                        # p_totalDMgam *= (pDMG + 1e-20)
                    i80=i80+1

                # if 80 < dm_value < 100:
                #     if i100==0:
                #         # p_totalDMamp = (pDMA + 1e-20)
                #         # p_totalDMgam = (pDMG + 1e-20)
                #         # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                #         # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                #         p_totalDMamp100 = (kde_eval_DMAMP + 1e-20)
                #         p_totalDMgam100 = (kde_eval_DMGAM + 1e-20)
                #     else:
                #         # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                #         # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                #         p_totalDMamp100 *= (kde_eval_DMAMP + 1e-20)
                #         p_totalDMgam100 *= (kde_eval_DMGAM + 1e-20)
                #         # p_totalDMamp *= (pDMA + 1e-20)
                #         # p_totalDMgam *= (pDMG + 1e-20)
                #     i100=i100+1

                if dm_value >= 80:
                    if iother==0:
                        # p_totalDMamp = (pDMA + 1e-20)
                        # p_totalDMgam = (pDMG + 1e-20)
                        # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                        p_totalDMampother = (kde_eval_DMAMP + 1e-20)
                        p_totalDMgamother = (kde_eval_DMGAM + 1e-20)
                    else:
                        # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                        # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                        p_totalDMampother *= (kde_eval_DMAMP + 1e-20)
                        p_totalDMgamother *= (kde_eval_DMGAM + 1e-20)
                        # p_totalDMamp *= (pDMA + 1e-20)
                        # p_totalDMgam *= (pDMG + 1e-20)
                    iother=iother+1
            
            if psrname+"_chrom_gp_log10_A" in pars:
                print(psrname+" has Chromatic noise")
                
                amp_index = pars.index(psrname+"_chrom_gp_log10_A")
                gamma_index = pars.index(psrname+"_chrom_gp_gamma")

                posts_CHROM_amp = result_psr[:, amp_index]
                posts_CHROM_gamma = result_psr[:, gamma_index]

                pCHROMA, binsCHROMA, patchesCHROMA = plt.hist(posts_CHROM_amp, bins=nbins, range=(-18, -11), density=True, alpha=0.6, histtype='step')
                pCHROMG, binsCHROMG, patchesCHROMG = plt.hist(posts_CHROM_gamma, bins=nbins, range=(0, 7), density=True, alpha=0.6, histtype='step')

                kdeCHROMAMP = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeCHROMAMP.fit(posts_CHROM_amp[:, None])
                logprob = kdeCHROMAMP.score_samples(newxbins2d[:, None])
                kde_eval_CHROMAMP = np.exp(logprob)

                #kde_eval_CHROMAMP = pCHROMA

                kdeCHROMGAMMA = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeCHROMGAMMA.fit(posts_CHROM_gamma[:, None])
                logprob = kdeCHROMGAMMA.score_samples(newybins2d[:, None])
                kde_eval_CHROMGAMMA = np.exp(logprob)

                #kde_eval_CHROMGAMMA = pCHROMG

                if dm_value < 20:
                    if j20==0:

                        p_totalCHROMamp20 = (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam20 = (kde_eval_CHROMGAMMA + 1e-20)

                    else:

                        p_totalCHROMamp20 *= (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam20 *= (kde_eval_CHROMGAMMA + 1e-20)

                    j20=j20+1

                if 20 < dm_value < 40:
                    if j40==0:

                        p_totalCHROMamp40 = (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam40 = (kde_eval_CHROMGAMMA + 1e-20)

                    else:

                        p_totalCHROMamp40 *= (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam40 *= (kde_eval_CHROMGAMMA + 1e-20)

                    j40=j40+1

                if 40 < dm_value < 60:
                    if j60==0:

                        p_totalCHROMamp60 = (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam60 = (kde_eval_CHROMGAMMA + 1e-20)

                    else:

                        p_totalCHROMamp60 *= (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam60 *= (kde_eval_CHROMGAMMA + 1e-20)

                    j60=j60+1

                if 60 < dm_value < 80:
                    if j80==0:

                        p_totalCHROMamp80 = (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam80 = (kde_eval_CHROMGAMMA + 1e-20)

                    else:

                        p_totalCHROMamp80 *= (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgam80 *= (kde_eval_CHROMGAMMA + 1e-20)

                    j80=j80+1

                # if 80 < dm_value < 100:
                #     if j100==0:

                #         p_totalCHROMamp100 = (kde_eval_CHROMAMP + 1e-20)
                #         p_totalCHROMgam100 = (kde_eval_CHROMGAMMA + 1e-20)

                #     else:

                #         p_totalCHROMamp100 *= (kde_eval_CHROMAMP + 1e-20)
                #         p_totalCHROMgam100 *= (kde_eval_CHROMGAMMA + 1e-20)

                #     j100=j100+1

                if dm_value >= 80:
                    if jother==0:

                        p_totalCHROMampother = (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgamother = (kde_eval_CHROMGAMMA + 1e-20)

                    else:

                        p_totalCHROMampother *= (kde_eval_CHROMAMP + 1e-20)
                        p_totalCHROMgamother *= (kde_eval_CHROMGAMMA + 1e-20)

                    jother=jother+1
                    
        except:
            continue

prod = np.prod(stacked,axis=0)


X, Y = np.meshgrid(newxbins2d, newybins2d)

DMbindiffamp = binsDMA[1]-binsDMA[0]
DMbindiffgam = binsDMG[1]-binsDMG[0]
DMpgam_adjust20 = p_totalDMgam20/(np.sum(p_totalDMgam20)*DMbindiffgam)
DMpamp_adjust20 = p_totalDMamp20/(np.sum(p_totalDMamp20)*DMbindiffamp)
DMpgam_adjust40 = p_totalDMgam40/(np.sum(p_totalDMgam40)*DMbindiffgam)
DMpamp_adjust40 = p_totalDMamp40/(np.sum(p_totalDMamp40)*DMbindiffamp)
DMpgam_adjust60 = p_totalDMgam60/(np.sum(p_totalDMgam60)*DMbindiffgam)
DMpamp_adjust60 = p_totalDMamp60/(np.sum(p_totalDMamp60)*DMbindiffamp)
DMpgam_adjust80 = p_totalDMgam80/(np.sum(p_totalDMgam80)*DMbindiffgam)
DMpamp_adjust80 = p_totalDMamp80/(np.sum(p_totalDMamp80)*DMbindiffamp)
#DMpgam_adjust100 = p_totalDMgam100/(np.sum(p_totalDMgam100)*DMbindiffgam)
#DMpamp_adjust100 = p_totalDMamp100/(np.sum(p_totalDMamp100)*DMbindiffamp)
DMpgam_adjustother = p_totalDMgamother/(np.sum(p_totalDMgamother)*DMbindiffgam)
DMpamp_adjustother = p_totalDMampother/(np.sum(p_totalDMampother)*DMbindiffamp)

CHROMbindiffamp = binsCHROMA[1]-binsCHROMA[0]
CHROMbindiffgam = binsCHROMG[1]-binsCHROMG[0]
try:
    CHROMpgam_adjust20 = p_totalCHROMgam20/(np.sum(p_totalCHROMgam20)*CHROMbindiffgam)  
    CHROMpamp_adjust20 = p_totalCHROMamp20/(np.sum(p_totalCHROMamp20)*CHROMbindiffamp)
except:
    pass
try:
    CHROMpgam_adjust40 = p_totalCHROMgam40/(np.sum(p_totalCHROMgam40)*CHROMbindiffgam)
    CHROMpamp_adjust40 = p_totalCHROMamp40/(np.sum(p_totalCHROMamp40)*CHROMbindiffamp)
except:
    pass
try:
    CHROMpgam_adjust60 = p_totalCHROMgam60/(np.sum(p_totalCHROMgam60)*CHROMbindiffgam)
    CHROMpamp_adjust60 = p_totalCHROMamp60/(np.sum(p_totalCHROMamp60)*CHROMbindiffamp)
except:
    pass
try:    
    CHROMpgam_adjust80 = p_totalCHROMgam80/(np.sum(p_totalCHROMgam80)*CHROMbindiffgam)
    CHROMpamp_adjust80 = p_totalCHROMamp80/(np.sum(p_totalCHROMamp80)*CHROMbindiffamp)
except:
    pass
try:
    CHROMpgam_adjust100 = p_totalCHROMgam100/(np.sum(p_totalCHROMgam100)*CHROMbindiffgam)
    CHROMpamp_adjust100 = p_totalCHROMamp100/(np.sum(p_totalCHROMamp100)*CHROMbindiffamp)
except:
    pass
try:
    CHROMpgam_adjustother = p_totalCHROMgamother/(np.sum(p_totalCHROMgamother)*CHROMbindiffgam)
    CHROMpamp_adjustother = p_totalCHROMampother/(np.sum(p_totalCHROMampother)*CHROMbindiffamp)
except:
    pass

DMampsample20 = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust20/np.sum(DMpamp_adjust20))
DMgammasample20 = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust20/np.sum(DMpgam_adjust20))
CHROMampsample20 = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust20/np.sum(CHROMpamp_adjust20))
CHROMgammasample20 = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust20/np.sum(CHROMpgam_adjust20))
DMampsample40 = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust40/np.sum(DMpamp_adjust40))
DMgammasample40 = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust40/np.sum(DMpgam_adjust40))
try:
    CHROMampsample40 = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust40/np.sum(CHROMpamp_adjust40))
    CHROMgammasample40 = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust40/np.sum(CHROMpgam_adjust40))
except:
    pass
DMampsample60 = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust60/np.sum(DMpamp_adjust60))
DMgammasample60 = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust60/np.sum(DMpgam_adjust60))
try:
    CHROMampsample60 = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust60/np.sum(CHROMpamp_adjust60))
    CHROMgammasample60 = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust60/np.sum(CHROMpgam_adjust60))
except:
    pass
DMampsample80 = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust80/np.sum(DMpamp_adjust80))
DMgammasample80 = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust80/np.sum(DMpgam_adjust80))
try:
    CHROMampsample80 = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust80/np.sum(CHROMpamp_adjust80))
    CHROMgammasample80 = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust80/np.sum(CHROMpgam_adjust80))
except:
    pass
#DMampsample100 = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust100/np.sum(DMpamp_adjust100))
#DMgammasample100 = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust100/np.sum(DMpgam_adjust100))
try:
    CHROMampsample100 = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust100/np.sum(CHROMpamp_adjust100))
    CHROMgammasample100 = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust100/np.sum(CHROMpgam_adjust100))
except:
    pass
DMampsampleother = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjustother/np.sum(DMpamp_adjustother))
DMgammasampleother = np.random.choice(newybins2d,size=10000, p=DMpgam_adjustother/np.sum(DMpgam_adjustother))
try:
    CHROMampsampleother = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjustother/np.sum(CHROMpamp_adjustother))
    CHROMgammasampleother = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjustother/np.sum(CHROMpgam_adjustother))
except:
    pass

# fig = corner.corner(np.array([DMampsample,DMgammasample]).T, bins=nbins, smooth=True, smooth1d=True, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="blue", plot_density=True, density=True)
# corner.corner(np.array([CHROMampsample,CHROMgammasample]).T, bins=nbins, smooth=True, smooth1d=True, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="orange", plot_density=True,density=True, fig=fig)

tag = "viridis"

cmap = sns.color_palette(tag,n_colors=6)

fig = corner.corner(np.array([DMgammasample20,DMampsample20]).T, bins=20, labels=[r"$\gamma_\mathrm{DM}$", r"$\mathrm{log_{10}A_{DM}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[0], plot_density=True, density=True, hist_kwargs={"density":True})
corner.corner(np.array([DMgammasample40,DMampsample40]).T, bins=20, labels=[r"$\gamma_\mathrm{DM}$", r"$\mathrm{log_{10}A_{DM}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[1], plot_density=True,density=True, fig=fig, range=([0,7],[-18, -11]), hist_kwargs={"density":True})
corner.corner(np.array([DMgammasample60,DMampsample60]).T, bins=20, labels=[r"$\gamma_\mathrm{DM}$", r"$\mathrm{log_{10}A_{DM}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[2], plot_density=True,density=True, fig=fig, range=([0,7],[-18, -11]), hist_kwargs={"density":True})
corner.corner(np.array([DMgammasample80,DMampsample80]).T, bins=20, labels=[r"$\gamma_\mathrm{DM}$", r"$\mathrm{log_{10}A_{DM}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[3], plot_density=True,density=True, fig=fig, range=([0,7],[-18, -11]), hist_kwargs={"density":True})
#corner.corner(np.array([DMampsample100,DMgammasample100]).T, bins=20, labels=[r"$\mathrm{log_{10}A_{DM}}$", r"$\gamma_\mathrm{DM}$"], color = cmap[4], plot_density=True,density=True, fig=fig, range=([-18, -11],[0,7]), hist_kwargs={"density":True})
corner.corner(np.array([DMgammasampleother,DMampsampleother]).T, bins=20, labels=[r"$\gamma_\mathrm{DM}$", r"$\mathrm{log_{10}A_{DM}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[4], plot_density=True,density=True, fig=fig, range=([0,7],[-18, -11]), hist_kwargs={"density":True})

# corner.corner(np.array([CHROMampsample,CHROMgammasample]).T, bins=20, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="orange", plot_density=True,density=True, fig=fig)
corner.overplot_lines(fig, xs=(8/3, None), color="grey", linestyle="--")

clr = cmap
clr = [cmap[0],cmap[1],cmap[2],cmap[3],cmap[4]]
#labels_legend = ["DM < 20", "20 < DM < 40", "40 < DM < 60", "60 < DM < 80", "80 < DM < 100", "DM > 100"]
labels_legend = ["DM < 20", "20 < DM < 40", "40 < DM < 60", "60 < DM < 80", "DM > 80"]
patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
legend_elements = patches
#sgwb_line = plt.plot(0,1, linestyle = "--", color="grey") 
legend_elements.append(Line2D([0,1],[0,1], color="grey", linestyle="--",label=r"$\gamma_\mathrm{Kolmogorov}=8/3$"))

fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/DM_factorised_likelihood_through_DM_HYPER.png")
fig.clf()

chrom_labels = []

fig = corner.corner(np.array([CHROMgammasample20,CHROMampsample20]).T, bins=20, labels=[r"$\gamma_\mathrm{Chrom}$", r"$\mathrm{log_{10}A_{Chrom}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[0], plot_density=True, density=True, hist_kwargs={"density":True})
chrom_labels.append("DM < 20")
try:
    corner.corner(np.array([CHROMgammasample40,CHROMampsample40]).T, bins=20, labels=[r"$\gamma_\mathrm{Chrom}$", r"$\mathrm{log_{10}A_{Chrom}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[1], plot_density=True,density=True, fig=fig, hist_kwargs={"density":True})
    chrom_labels.append("20 < DM < 40")
except:
    pass
try:
    corner.corner(np.array([CHROMgammasample60,CHROMampsample60]).T, bins=20, labels=[r"$\gamma_\mathrm{Chrom}$", r"$\mathrm{log_{10}A_{Chrom}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[2], plot_density=True,density=True, fig=fig, hist_kwargs={"density":True})
    chrom_labels.append("40 < DM < 60")
except:
    pass
try:
    corner.corner(np.array([CHROMgammasample80,CHROMampsample80]).T, bins=20, labels=[r"$\gamma_\mathrm{Chrom}$", r"$\mathrm{log_{10}A_{Chrom}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[3], plot_density=True,density=True, fig=fig, hist_kwargs={"density":True})
    chrom_labels.append("60 < DM < 80")
except:
    pass
#try:
    #corner.corner(np.array([CHROMampsample100,CHROMgammasample100]).T, bins=20, labels=[r"$\mathrm{log_{10}A_{CHROM}}$", r"$\gamma_\mathrm{CHROM}$"], color = cmap[4], plot_density=True,density=True, fig=fig, hist_kwargs={"density":True})
    #chrom_labels.append("80 < DM < 100")
#except:
#    pass
try:
    corner.corner(np.array([CHROMgammasampleother,CHROMampsampleother]).T, bins=20, labels=[r"$\gamma_\mathrm{Chrom}$", r"$\mathrm{log_{10}A_{Chrom}}$"], label_kwargs={"fontsize":14}, figsize=(19.20,10.80), color = cmap[4], plot_density=True,density=True, fig=fig, hist_kwargs={"density":True})
    chrom_labels.append("DM > 80")
except:
    pass
# corner.corner(np.array([CHROMampsample,CHROMgammasample]).T, bins=20, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="orange", plot_density=True,density=True, fig=fig)

cmap2 = sns.color_palette(tag,n_colors=len(chrom_labels))
clr = cmap
#labels_legend = ["DM < 20", "20 < DM < 40", "40 < DM < 60", "60 < DM < 80", "80 < DM < 100", "DM > 100"]
labels_legend = chrom_labels
patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
legend_elements = patches
fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.tight_layout()
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Chrom_factorised_likelihood_through_DM_HYPER.png")
fig.clf()
