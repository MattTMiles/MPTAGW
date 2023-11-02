import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import corner
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from sklearn.neighbors import KernelDensity

mpta_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM_nlive600/"

psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []

nbins = 20

i=0
j=0
for pulsar in to_use:
    psrname = pulsar.strip("\n")
    if psrname != "J00000":
        print(psrname)
        try:
            psr_dir = mpta_dir + "/" + psrname + "_MPTA_PM"

            result_psr = bilby.result.read_in_result(psr_dir+"/MPTA_PM_result.json")

            newxbins2d = np.linspace(-18,-11,10000)
            newybins2d = np.linspace(0,7,10000)

            try:
                posts_DM_amp = result_psr.posterior[psrname+"_dm_gp_log10_A"].values
                posts_DM_gamma = result_psr.posterior[psrname+"_dm_gp_gamma"].values

                pDMA, binsDMA, patchesDMA = plt.hist(posts_DM_amp, bins=nbins, range=(-18, -11), density=True, alpha=0.6, histtype='step')
                pDMG, binsDMG, patchesDMG = plt.hist(posts_DM_gamma, bins=nbins, range=(0, 7), density=True, alpha=0.6, histtype='step')
            
                # FWHM=50
                # FWHMalt = 2
                #FWHM2 = 
                # sigma = fwhm2sigma(FWHM)
                # sigmaalt = fwhm2sigma(FWHMalt)

                # smoothed_vals_dmamp = np.zeros(pDMA.shape)
                # x_vals = np.linspace(0,nbins-1,nbins)
                # for x_position in x_vals:
                #     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
                #     kernel = kernel / sum(kernel)
                #     smoothed_vals_dmamp[int(x_position)] = sum(pDMA * kernel)

                #kdeDMAMP = gaussian_kde(posts_DM_amp)
                kdeDMAMP = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeDMAMP.fit(posts_DM_amp[:, None])
                logprob = kdeDMAMP.score_samples(newxbins2d[:, None])
                kde_eval_DMAMP = np.exp(logprob)

                # smoothed_vals_dmgam = np.zeros(pDMG.shape)
                # x_vals = np.linspace(0,nbins-1,nbins)
                # for x_position in x_vals:
                #     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
                #     kernel = kernel / sum(kernel)
                #     smoothed_vals_dmgam[int(x_position)] = sum(pDMG * kernel)

                #kdeDMGAM = gaussian_kde(posts_DM_gamma)
                kdeDMGAM = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeDMGAM.fit(posts_DM_gamma[:, None])
                logprob = kdeDMGAM.score_samples(newybins2d[:, None])
                kde_eval_DMGAM = np.exp(logprob)

                if i==0:
                    # p_totalDMamp = (pDMA + 1e-20)
                    # p_totalDMgam = (pDMG + 1e-20)
                    # p_totalDMamp = (smoothed_vals_dmamp + 1e-20)
                    # p_totalDMgam = (smoothed_vals_dmgam + 1e-20)
                    p_totalDMamp = (kde_eval_DMAMP + 1e-20)
                    p_totalDMgam = (kde_eval_DMGAM + 1e-20)
                else:
                    # p_totalDMamp *= (smoothed_vals_dmamp + 1e-20)
                    # p_totalDMgam *= (smoothed_vals_dmgam + 1e-20)
                    p_totalDMamp *= (kde_eval_DMAMP + 1e-20)
                    p_totalDMgam *= (kde_eval_DMGAM + 1e-20)
                    # p_totalDMamp *= (pDMA + 1e-20)
                    # p_totalDMgam *= (pDMG + 1e-20)
                i=i+1

            except:
                pass
            
            try:
                posts_CHROM_amp = result_psr.posterior[psrname+"_chrom_gp_log10_A"].values
                posts_CHROM_gamma = result_psr.posterior[psrname+"_chrom_gp_gamma"].values
                
                pCHROMA, binsCHROMA, patchesCHROMA = plt.hist(posts_CHROM_amp, bins=nbins, range=(-18, -11), density=True, alpha=0.6, histtype='step')
                pCHROMG, binsCHROMG, patchesCHROMG = plt.hist(posts_CHROM_gamma, bins=nbins, range=(0, 7), density=True, alpha=0.6, histtype='step')
            
                # smoothed_vals_chromamp = np.zeros(pCHROMA.shape)
                # x_vals = np.linspace(0,nbins-1,nbins)
                # for x_position in x_vals:
                #     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
                #     kernel = kernel / sum(kernel)
                #     smoothed_vals_chromamp[int(x_position)] = sum(pCHROMA * kernel)

                # kdeCHROMAMP = gaussian_kde(posts_CHROM_amp)
                # kde_eval_CHROMAMP = kdeCHROMAMP(newxbins2d)

                kdeCHROMAMP = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeCHROMAMP.fit(posts_CHROM_amp[:, None])
                logprob = kdeCHROMAMP.score_samples(newxbins2d[:, None])
                kde_eval_CHROMAMP = np.exp(logprob)
                # smoothed_vals_chromgam = np.zeros(pCHROMG.shape)
                # x_vals = np.linspace(0,nbins-1,nbins)
                # for x_position in x_vals:
                #     kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
                #     kernel = kernel / sum(kernel)
                #     smoothed_vals_chromgam[int(x_position)] = sum(pCHROMG * kernel)

                # kdeCHROMGAMMA = gaussian_kde(posts_CHROM_gamma)
                # kde_eval_CHROMGAMMA = kdeCHROMGAMMA(newybins2d)

                kdeCHROMGAMMA = KernelDensity(bandwidth=1.0, kernel="gaussian")
                kdeCHROMGAMMA.fit(posts_CHROM_gamma[:, None])
                logprob = kdeCHROMGAMMA.score_samples(newybins2d[:, None])
                kde_eval_CHROMGAMMA = np.exp(logprob)

                if j==0:
                    # p_totalCHROMamp = (smoothed_vals_chromamp + 1e-20)
                    # p_totalCHROMgam = (smoothed_vals_chromgam + 1e-20)
                    p_totalCHROMamp = (kde_eval_CHROMAMP + 1e-20)
                    p_totalCHROMgam = (kde_eval_CHROMGAMMA + 1e-20)
                    # p_totalCHROMamp = (pCHROMA + 1e-20)
                    # p_totalCHROMgam = (pCHROMG + 1e-20)
                else:
                    # p_totalCHROMamp *= (smoothed_vals_chromamp + 1e-20)
                    # p_totalCHROMgam *= (smoothed_vals_chromgam + 1e-20)
                    p_totalCHROMamp *= (kde_eval_CHROMAMP + 1e-20)
                    p_totalCHROMgam *= (kde_eval_CHROMGAMMA + 1e-20)
                    # p_totalCHROMamp *= (pCHROMA + 1e-20)
                    # p_totalCHROMgam *= (pCHROMG + 1e-20)
                j=j+1

            except:
                pass

        except:
            continue

prod = np.prod(stacked,axis=0)


X, Y = np.meshgrid(newxbins2d, newybins2d)

DMbindiffamp = binsDMA[1]-binsDMA[0]
DMbindiffgam = binsDMG[1]-binsDMG[0]
DMpgam_adjust = p_totalDMgam/(np.sum(p_totalDMgam)*DMbindiffgam)
DMpamp_adjust = p_totalDMamp/(np.sum(p_totalDMamp)*DMbindiffamp)

CHROMbindiffamp = binsCHROMA[1]-binsCHROMA[0]
CHROMbindiffgam = binsCHROMG[1]-binsCHROMG[0]
CHROMpgam_adjust = p_totalCHROMgam/(np.sum(p_totalCHROMgam)*CHROMbindiffgam)
CHROMpamp_adjust = p_totalCHROMamp/(np.sum(p_totalCHROMamp)*CHROMbindiffamp)


DMampsample = np.random.choice(newxbins2d,size=10000, p=DMpamp_adjust/np.sum(DMpamp_adjust))
DMgammasample = np.random.choice(newybins2d,size=10000, p=DMpgam_adjust/np.sum(DMpgam_adjust))

CHROMampsample = np.random.choice(newxbins2d,size=10000, p=CHROMpamp_adjust/np.sum(CHROMpamp_adjust))
CHROMgammasample = np.random.choice(newybins2d,size=10000, p=CHROMpgam_adjust/np.sum(CHROMpgam_adjust))


# fig = corner.corner(np.array([DMampsample,DMgammasample]).T, bins=nbins, smooth=True, smooth1d=True, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="blue", plot_density=True, density=True)
# corner.corner(np.array([CHROMampsample,CHROMgammasample]).T, bins=nbins, smooth=True, smooth1d=True, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="orange", plot_density=True,density=True, fig=fig)

fig = corner.corner(np.array([DMampsample,DMgammasample]).T, bins=20, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="blue", plot_density=True, density=True)
corner.corner(np.array([CHROMampsample,CHROMgammasample]).T, bins=20, labels=[r"$\mathrm{log_{10}A}$", r"$\gamma$"], color="orange", plot_density=True,density=True, fig=fig)



clr = ["blue", "orange"]
labels_legend = ["DM Noise", "Chromatic Noise"]
patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
legend_elements = patches
fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/CHROM_DM_factorised_likelihood.png")
fig.clf()

