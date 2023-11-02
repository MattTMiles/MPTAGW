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
from scipy.interpolate import interp1d
import astropy.units as u
import glob
from scipy import signal

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = True)
parser.add_argument("-directory", dest="directory", help="Directory where par and tim are found", required = True)
args = parser.parse_args()
pulsar = str(args.pulsar)
maindir = str(args.directory)

real_dir = "/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_realiser/"+pulsar+"/reals/"

dm_files = glob.glob(real_dir+"real_*/dm_gpt.npy")
red_files = glob.glob(real_dir+"real_*/red_gpt.npy")
# ecorr_files = glob.glob(real_dir+"real_*/ecorr_gpt.npy")
sw_files = glob.glob(real_dir+"real_*/sw_gpt.npy")

dm_reals = [ np.load(dm_real) for dm_real in dm_files ]
dm_combine = np.vstack(dm_reals)

red_reals = [ np.load(red_real) for red_real in red_files ]
red_combine = np.vstack(red_reals)

# ecorr_reals = [ np.load(ecorr_real) for ecorr_real in ecorr_files ]
# for i, ecorr in enumerate(ecorr_reals):
#     if len(ecorr) != 99999:
#         ecorr_reals.pop(i)
# ecorr_combine = np.vstack(ecorr_reals)


sw_reals = [ np.load(sw_real) for sw_real in sw_files ]

for i, sw in enumerate(sw_reals):

    if len(sw) != 99999:
        sw_reals.pop(i)
try:
    sw_combine = np.vstack(sw_reals)
except:
    print("SW needs more trim")
    for i, sw in enumerate(sw_reals):
        if len(sw) != 99999:
            sw_reals.pop(i)
    sw_combine = np.vstack(sw_reals)

med_dm = np.array([ np.nanmedian(dm_c) for dm_c in dm_combine.T ])
med_red = np.array([ np.nanmedian(red_c) for red_c in red_combine.T ])
# med_ecorr = np.array([ np.nanmedian(ecorr_c) for ecorr_c in ecorr_combine.T ])
med_sw = np.array([ np.nanmedian(sw_c) for sw_c in sw_combine.T ])

dm_time = np.load(real_dir+"real_1/dm_interp_region.npy")*(u.s)
red_time = np.load(real_dir+"real_1/red_interp_region.npy")*(u.s)
# ecorr_time = np.load(real_dir+"real_1/ecorr_interp_region.npy")*(u.d)
sw_time = np.load(real_dir+"real_1/sw_interp_region.npy")*(u.d)

dm_days = dm_time.to(u.d)
red_days = red_time.to(u.d)
# ecorr_days = ecorr_time
sw_days = sw_time

maindir = maindir+"/"
psr = Pulsar(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", ephem="DE440")

m, t_all = get_model_and_toas(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", allow_name_mixing=True)

psrname = psr.name

f = pint.fitter.DownhillGLSFitter(toas=t_all, model=m)
f.fit_toas(maxiter=3, debug=True)

mjds = f.toas.get_mjds()
noise_df = pd.DataFrame.from_dict(f.resids.noise_resids)
noise_df["MJD"] = mjds.value


font=15

fig, ax = plt.subplots(3, figsize = (15, 15))
ax[0].errorbar(x=noise_df["MJD"], y=f.resids.resids.value*1e6,yerr=f.toas.get_errors().to(u.us).value,linestyle="",marker=".", label = "Residuals")
ax[1].errorbar(x=noise_df["MJD"], y=f.resids.resids.value*1e6,yerr=f.toas.get_errors().to(u.us).value,linestyle="",marker=".", label = "Residuals", alpha=0.15, zorder=2)

whitened_residuals = np.copy(f.resids.resids.value)
for col in noise_df.columns:
    if "noise" in col:

        whitened_residuals -= noise_df[col].values

ax[2].errorbar(x=noise_df["MJD"], y=whitened_residuals*1e6,yerr=f.toas.get_errors().to(u.us).value,linestyle="",marker=".", label = "Whitened Residuals")


#ax[0].set_title(psrname + " residuals")
ax[0].legend(fontsize=font)
ax[0].set_ylabel(r"Residuals ($\mu$s)", fontsize=font)

ax[1].plot(dm_days, dm_combine.T*1e6, color="tab:purple", alpha=0.1, lw=0.1)
ax[1].plot(dm_days, med_dm.T*1e6, color="tab:purple", alpha=1, lw=3, label="Dispersion Measure")

ax[1].plot(red_days, red_combine.T*1e6, color="tab:red", alpha=0.1, lw=0.1)
ax[1].plot(red_days, med_red.T*1e6, color="tab:red", alpha=1, lw=3, label="Achromatic Red")

# if len(ecorr_days) != 99999:
#     ecorr_days = ecorr_days[:-1]

# ax[1].plot(ecorr_days, ecorr_combine.T*1e6, color="tab:orange", alpha=0.1, lw=0.1)
# ax[1].plot(ecorr_days, med_ecorr.T*1e6, color="tab:orange", alpha=1, lw=3, label="GP ECORR Realisation")

if len(sw_days) != 99999:
    sw_days = sw_days[:-1]

sg_swcombine = signal.savgol_filter(sw_combine, 4501, 4)
sg_med_sw = signal.savgol_filter(med_sw, 4501, 4)

ax[1].plot(sw_days, sg_swcombine.T*1e6, color="tab:green", alpha=0.1, lw=0.1)
ax[1].plot(sw_days, sg_med_sw.T*1e6, color="tab:green", alpha=1, lw=3, label="Solar Wind")

ax[1].set_xlabel("MJD", fontsize=font)
ax[1].set_ylim(-1, 1)
ax[1].set_ylabel(r"Residuals ($\mu$s)", fontsize=font)

leg = ax[1].legend(loc='upper right', fontsize=font)

for lh in leg.legend_handles: 
    lh.set_alpha(1)

ax[0].tick_params(axis='both', which='both', labelsize=15)
ax[1].tick_params(axis='both', which='both', labelsize=15)

ax[2].set_xlabel("MJD", fontsize=font)
ax[2].set_ylabel(r"Residuals ($\mu$s)", fontsize=font)

ax[2].legend(loc='upper right', fontsize=font)

ax[2].tick_params(axis='both', which='both', labelsize=15)
fig.tight_layout()
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/noise_realiser/J1909-3744/resids_noise_reals_SW_SG.png")
fig.clf()
plt.clf()
