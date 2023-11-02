import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.optimize import curve_fit

res_file = "/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/1017_resid.dat"

res_file = np.loadtxt(res_file)

res = res_file[:,1]*1e6
unc = res_file[:,2]
freq = res_file[:,3]

chrom_real = np.load("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_chrom_noise_real.npy")
chrom_real = chrom_real*1e6

dm_real = np.load("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_DM_noise_real.npy")
dm_real = dm_real*1e6

red_real = np.load("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_red_noise_real.npy")
red_real = red_real*1e6

ecorr_real = np.load("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_ecorr_noise_real.npy")
ecorr_real = ecorr_real*1e6

sw_real = np.load("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_sw_noise_real.npy")
sw_real = sw_real*1e6

def chrom_map(freqs, res, unc):

    w_ave = np.average(res, weights = unc)
    ref_freq = 1400

    return ((freqs/ref_freq)**-4)

def dm_map(freqs, res, unc):

    w_ave = np.average(res, weights = unc)
    ref_freq = 1400

    return 


chrom_fit = chrom_map(freq, res, unc)
dm_fit = dm_map(freq, res, unc)

def dm_solve(freqs, dm_delta, offset):

    ref_freq = 1400

    return dm_delta*(((freqs/ref_freq)**-2)) + offset

p_optdm, p_covdm = curve_fit(dm_solve, freq, res, sigma=unc)

def chrom_solve(freqs, chrom_delta, offset):

    ref_freq = 1400

    return chrom_delta*(((freqs/ref_freq)**-4)) + offset

p_optchrom, p_covchrom = curve_fit(chrom_solve, freq, res, sigma=unc)

def chrom_dm(freqs, dm_delta, chrom_delta, offset):

    ref_freq = 1400

    chrom = chrom_delta*(((freqs/ref_freq)**-4))
    dm = dm_delta*(((freqs/ref_freq)**-2))

    return chrom+ dm + offset

total_real = chrom_real+dm_real+red_real+ecorr_real+sw_real

p_optchromdm, p_covchromdm = curve_fit(chrom_dm, freq, total_real)



freq2 = np.linspace(800, 5000, 10000)

freq3 = np.linspace(1647.899, 5000, 10000)


fig, ax = plt.subplots(figsize=(10,10))

ax.plot(freq, chrom_real+dm_real+red_real+ecorr_real+sw_real, label="Epoch noise realisation", color="black", lw=3)
ax.plot(freq3, chrom_dm(freq3, p_optchromdm[0], p_optchromdm[1], p_optchromdm[2]), label = r"Extrapolation", color="black", linestyle="--", lw=3)
ax.plot(freq2, chrom_solve(freq2, p_optchrom[0], p_optchrom[1]), label = r"$\beta=4$", color="tab:green")
ax.plot(freq2, dm_solve(freq2, p_optdm[0], p_optdm[1]), label = r"$\beta=2$", color="tab:orange")

ax.set_xscale("log")
ax.errorbar(freq, res, yerr=unc, label = "Timing Residuals", linestyle="", marker=".", ms=10, color="tab:blue")

ax.set_ylabel("Timing Residuals ($\mu$s)", fontsize=20)
ax.set_xlabel("Frequency (MHz)", fontsize=20)
ax.tick_params(axis="both", labelsize=20, which="both")
ax.legend(fontsize=20)

fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_epoch_CHROM_DM")
plt.clf()

