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
import glob
import time
import random

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
#parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = True)
parser.add_argument("-directory", dest="directory", help="Directory where par and tim are found", required = True)
args = parser.parse_args()
#pulsar = str(args.pulsar)
maindir = str(args.directory)


def dp_orf(pos1, pos2):
    """Hellings & Downs spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1
    else:
        return np.dot(pos1, pos2)
       

def hd_orf(pos1, pos2):
    """Hellings & Downs spatial correlation function."""
    if np.all(pos1 == pos2):
        return 0.5
    else:
        omc2 = (1 - np.dot(pos1, pos2)) / 2
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5


par1909 = maindir+"/J1909-3744_tdb.par"
tim1909 = maindir+"/J1909-3744.tim"

psr1909 = Pulsar(par1909, tim1909, ephem="DE440")

psrs = sorted(glob.glob(maindir + '/J*gpt*'))
#parfiles = sorted(glob.glob(maindir + '/J*par'))
parfiles = [ p.replace("_gpt.npy","_tdb.par") for p in psrs ]
#timfiles = sorted(glob.glob(maindir + '/J*tim'))
timfiles = [ p.replace("_gpt.npy",".tim") for p in psrs ]
ephemeris = "DE440"

psrs = []
for p, t in zip(parfiles, timfiles):
    #if "J1903" not in p and "J1455" not in p:
    if "J1903" not in p and "J1455" not in p and "J1643" not in p and "J1804-2717" not in p and "J1933-6211" not in p and "J1909" not in p:
        
        psr = Pulsar(p, t, ephem=ephemeris)
        print(psr.name)
        psrs.append(psr)

gw_t_all = []
i=0
while i < 100:

    ct_ps = []

    weights = []
    
    npsr = 0

    while npsr < 74:

        psr = random.choice(psrs)

        gpt = np.load(psr.name+"_gpt.npy")

        corr = hd_orf(psr1909.pos,psr.pos)

        weights.append(corr)
        ct_p = corr*gpt
        

        ct_ps.append(ct_p)
        npsr = npsr+1


    #time /= (86400.)
    ct_ps_sum = np.sum(ct_ps,axis=0)

    total_weights = np.sum(weights)

    gw_t_real = ct_ps_sum/total_weights

    gw_t_all.append(gw_t_real)

    i = i+1

ct_ps = []

weights = []

for gptfile in glob.glob(maindir+"/*gpt.npy"):
    if "J1909" not in gptfile:
        parfile = gptfile.replace("_gpt.npy", "_tdb.par")
        timfile = gptfile.replace("_gpt.npy", ".tim")

        gpt = np.load(gptfile)

        psr = Pulsar(parfile, timfile, ephem="DE440")

        corr = hd_orf(psr1909.pos,psr.pos)
        #corr = dp_orf(psr1909.pos,psr.pos)

        weights.append(corr)
        ct_p = corr*gpt
        

        ct_ps.append(ct_p)

ct_ps_sum = np.sum(ct_ps,axis=0)

total_weights = np.sum(weights)

gw_t_total = ct_ps_sum/total_weights

time = np.load("J1909-3744_time_series.npy")
time = time/86400

plt.figure(figsize=(15,5))

for gwr in gw_t_all:
    plt.plot(time, gwr,linestyle="",marker=".")


plt.savefig("bootstrap_100real_1909nored_HD05_noJ1909.png")
plt.clf()

data = np.vstack(gw_t_all)

stdevs = []
for i in range(len(data.T)):
    stdev = np.std(data[:,i])
    stdevs.append(stdev)

stdevs= np.array(stdevs)

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(time, gw_t_total, color="tab:green")
ax.fill_between(time, gw_t_total - stdevs, gw_t_total + stdevs,color="tab:green",alpha=0.25)
#ax.errorbar(time, gw_t_total, yerr=stdevs)
ax.set_title("Coherent Sum towards J1909-3744")
ax.set_ylabel("GW Amplitude (s)")
ax.set_xlabel("MJD")

fig.savefig("CoherentSum_dirJ1909_bootstrap_err_1909nored_HD05_noJ1909.png")
plt.clf()


ct_ps_sum1 = np.sum(ct_ps[::2],axis=0)
ct_ps_sum2 = np.sum(ct_ps[1::2],axis=0)

weights1 = np.sum(weights[::2])
weights2 = np.sum(weights[1::2])

gw_firsthalf = ct_ps_sum1/weights1
gw_secondhalf = ct_ps_sum2/weights2




fig, ax = plt.subplots(figsize=(15,5))
ax.plot(time, gw_firsthalf,label = "first",color="tab:orange")
ax.plot(time, gw_secondhalf,label = "second",color="tab:blue")
ax.fill_between(time, gw_firsthalf - stdevs, gw_firsthalf + stdevs, color="tab:orange", alpha=0.25)
ax.fill_between(time, gw_secondhalf - stdevs, gw_secondhalf + stdevs, color="tab:blue", alpha=0.25)
ax.legend()
fig.savefig("CoherentSum_dirJ1909_bootstrap_err_split_array_1909nored_HD05_noJ1909.png")
plt.clf()
