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
from decimal import *

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
parser.add_argument("-dir", dest="dir", help="Directory to scoop files from", required = False)
args = parser.parse_args()
pulsar = str(args.pulsar)
datadir = str(args.dir)

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

def lazy_noise_reducer(parfile, timfile, datadir):
    # Both the pulsar and the fitter object are obscenely heavy, so this code needs to garbage collect everything first otherwise it taps out
    maindir = datadir
    psr1 = Pulsar(maindir+parfile, maindir+timfile, ephem="DE440")
    m1, t_all = get_model_and_toas(maindir+parfile, maindir+timfile, allow_name_mixing=True)
    psrname = m1.name.split("/")[-1].replace("_tdb.par","")

    glsfit1 = pint.fitter.GLSFitter(toas=t_all, model=m1)
    glsfit1.fit_toas(maxiter=3)

    og_tim = np.loadtxt(maindir+timfile,skiprows = 2, usecols=2, dtype=np.float128)
    og_tim = u.Quantity(og_tim, dtype=Decimal)*u.d
    
    mjds = glsfit1.toas.get_mjds()
    noise_df1 = pd.DataFrame.from_dict(glsfit1.resids.noise_resids)
    noise_df1["MJD"] = mjds.value
    noise_df1["Residuals"] = glsfit1.resids.resids.value

    alt_tim = og_tim
    for col in noise_df1.columns:
        if "noise" in col:
            if "pl_gw" not in col:
                if "pl_red" not in col:
                    noiseoff = u.Quantity(noise_df1[col].values, dtype=Decimal)*u.s
                    alt_tim = alt_tim - noiseoff

    #Replace the tim file
    with open(maindir+timfile,"r") as original, open(maindir+"/"+psrname+"_off_exceptRedGW.tim","w") as newfile:
        for i, line in enumerate(original):
            newline = line.split()
            if i>1:
                #newline[0] = " "+line[0]
                newline[2] = str(alt_tim[i-2].value)
            nl = " ".join(newline) + " \n"
            
            newfile.write(nl)

    return glsfit1, m1, noise_df1, alt_tim
    

   
parfile = pulsar+"_tdb.par"
#parfile_misspec = pulsar+"_tdb_misspec.par"
timfile = pulsar+".tim"

glsfit1, model1, noise_df1, alt_tim = lazy_noise_reducer(parfile, timfile, datadir)
