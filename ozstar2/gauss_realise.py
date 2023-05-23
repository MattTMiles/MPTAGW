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
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = True)
parser.add_argument("-directory", dest="directory", help="Directory where par and tim are found", required = True)
parser.add_argument("-times", dest="times", help="Path of file with times in it in seconds", required = True)
args = parser.parse_args()
pulsar = str(args.pulsar)
maindir = str(args.directory)
times = str(args.times)

def gauss_realise(directory, pulsar, times):

    ab_coeff_file = directory+"/"+pulsar+"_ab_coeffs.npy"
    ab_coeffs = np.load(ab_coeff_file,allow_pickle=True)

    freq_coeff_file = directory+"/"+pulsar+"_freq_coeffs.npy"
    freq_coeffs = np.load(freq_coeff_file,allow_pickle=True)

    times = np.load(times)

    gpt = 0
    i=0
    for f in freq_coeffs:
        
        gpt += ab_coeffs[i]*np.sin(2*np.pi*f*times) + ab_coeffs[i+1]*np.cos(2*np.pi*f*times)
        i +=2

    return gpt




gpt = gauss_realise(maindir, pulsar, times)


np.save(maindir+"/"+pulsar+"_gpt.npy", gpt)

#plt.plot()
