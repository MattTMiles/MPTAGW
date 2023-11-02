# Qucik script to plot the growth of the background through time

import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.timeseries import LombScargle
import sys
import astropy.units as u
import pandas as pd
from scipy.optimize import curve_fit
import glob

red_dir = "/fred/oz002/users/mmiles/theses_plots/red_reals/"

#pulsar = sys.argv[1]

mjd_globs = sorted(glob.glob(red_dir+"/*MJD*npy"))
red_globs = sorted(glob.glob(red_dir+"/*RED*npy"))

# mjds = np.load(red_dir+"/"+pulsar+"_MJD.npy")
# red = np.load(red_dir+"/"+pulsar+"_RED.npy")
# mjds_days = mjds*(u.d)
# mjds_seconds = mjds_days.to(u.s)
# mjd_vals = mjds_seconds.value

# mjd_vals = []
# red_vals = []

fig, ax = plt.subplots(figsize=(12,8))

#ax.set_title(pulsar+" linear scale, Nobs/Tspan")
ax.set_xscale("log")


for i, mjd in enumerate(mjd_globs):
    psrname=mjd.split("/")[-1]
    psrname = psrname.split("_")[0].strip("\n")
    mjds = np.load(mjd)
    mjds_days = mjds*(u.d)
    mjds_seconds = mjds_days.to(u.s)
    mjd_val = mjds_seconds.value

    reds = np.load(red_globs[i])
    reds = reds*(u.s)

    data = [mjd_val, mjds_days, reds]
    data = np.array(data)
    df = pd.DataFrame(data.T,columns=["MJD","MJD_DAYS", "RED"])
    df["roundMJD"] = df["MJD_DAYS"].round()
    red_grouped = df.groupby("roundMJD").mean()

    roundMJD = red_grouped["MJD"].values
    roundRED = red_grouped["RED"].values
    roundRED = roundRED*(u.s)
    roundMJD = roundMJD*(u.s)
    ls = LombScargle(roundMJD, roundRED, normalization="psd")
    frequency, power = ls.autopower(minimum_frequency=1/(roundMJD.max()-roundMJD.min()))
    norm = len(roundRED)/(roundMJD.max()-roundMJD.min())

    ax.plot(frequency, (power/norm)/((power/norm).max()), label = psrname)


#ax.legend(fontsize=20)
ax.tick_params(axis="both", which="both",labelsize=15)
ax.set_ylabel("Normalised Power Spectral Density", fontsize=15)
ax.set_xlabel("Frequency (Hz)", fontsize=15)
ax.legend(fontsize=15)
ax.yaxis.get_offset_text().set_fontsize(15)
fig.tight_layout()
fig.savefig(red_dir+"/total_RED_PSD_linear_nobsTspan.png")
fig.clf()


# red = red*(u.s)
# data = [mjd_vals, mjds_days, red]
# data = np.array(data)
# df = pd.DataFrame(data.T,columns=["MJD","MJD_DAYS", "RED"])
# df["roundMJD"] = df["MJD_DAYS"].round()
# red_grouped = df.groupby("roundMJD").mean()
# roundMJD = red_grouped["MJD"].values
# roundRED = red_grouped["RED"].values
# roundRED = roundRED*(u.s)
# roundMJD = roundMJD*(u.s)
# ls = LombScargle(roundMJD, roundRED, normalization="psd")
# frequency, power = ls.autopower(minimum_frequency=1/(roundMJD.max()-roundMJD.min()))
# norm = len(roundRED)/(roundMJD.max()-roundMJD.min())

# fig, ax = plt.subplots(figsize=(10,10))

# ax.set_title(pulsar+" log scale")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.plot(frequency, power*(roundMJD.max()-roundMJD.min()), label = "Norm: 1/Tspan")
# ax.plot(frequency, power/norm, label = "Norm: Nobs/Tspan")
# ax.plot(frequency, power/frequency, label = "Norm: Frequency")
# ax.legend()

# fig.savefig(red_dir+"/"+pulsar+"_RED_PSD.png")
# fig.clf()
# plt.clf()

# fig, ax = plt.subplots(figsize=(10,10))

# ax.set_title(pulsar+" linear scale")
# ax.set_xscale("log")
# ax.plot(frequency, power*(roundMJD.max()-roundMJD.min()), label = "Norm: 1/Tspan")
# ax.plot(frequency, power/norm, label = "Norm: Nobs/Tspan")
# ax.plot(frequency, power/frequency, label = "Norm: Frequency")
# ax.legend()

# fig.savefig(red_dir+"/"+pulsar+"_RED_PSD_linear.png")
# fig.clf()
# plt.clf()


# fig, ax = plt.subplots(figsize=(12,8))

# #ax.set_title(pulsar+" linear scale, Nobs/Tspan")
# ax.set_xscale("log")
# ax.plot(frequency, power/norm, label = "Power Spectral Density")

# #ax.legend()
# pwl = np.sqrt((10**-14.21)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(4.3333-3) * frequency**(-4.3333) * frequency[0])

# def powerlaw(frequency, amp, gamma):
    
#     return np.sqrt((10**amp)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(gamma-3) * frequency**(gamma) * frequency[0])


# #popt, pcov = curve_fit(broken_powerlaw, frequency.value, power.value/norm.value)

# #ax.plot(frequency, pwl, label=r"$\log_{10}\mathrm{A_{CURN}} = -14.21; \gamma_{\mathrm{CURN}} = 13/3$", lw=2, color="black")
# #ax.plot(frequency.value, broken_powerlaw(frequency.value, popt[0], popt[1], popt[2], popt[3]), color="red")

# ax.set_xlabel("Frequency (Hz)", fontsize=20)
# ax.set_ylabel(r"P ($\mathrm{s}^{3}}$)", fontsize=20)
# #ax.set_yscale("log")
# ax.legend(fontsize=20)
# ax.tick_params(axis="both", which="both",labelsize=15)
# ax.yaxis.get_offset_text().set_fontsize(15)
# fig.tight_layout()
# fig.savefig(red_dir+"/"+pulsar+"_RED_PSD_linear_nobsTspan.png")
# fig.clf()
# plt.clf()
