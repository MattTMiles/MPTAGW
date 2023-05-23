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

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
#parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = True)
parser.add_argument("-directory", dest="directory", help="Directory where par and tim are found", required = True)
args = parser.parse_args()
#pulsar = str(args.pulsar)
maindir = str(args.directory)



for gpt in glob.glob(maindir+"/*")


def co_coll(directory, pulsar):

    maindir = directory+"/"
    psr = Pulsar(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", ephem="DE440")

    m, t_all = get_model_and_toas(maindir+pulsar+"_tdb.par", maindir+pulsar+".tim", allow_name_mixing=True)
    psrname = psr.name
    f = pint.fitter.DownhillGLSFitter(toas=t_all, model=m)
    f.fit_toas(maxiter=3, debug=True)

    noise_dims = f.current_state.model.noise_model_dimensions(f.toas)
    ntmpar = len(f.model.free_params)

    p0 = noise_dims["pl_gw_noise"][0] + ntmpar +1
    p1 = noise_dims["pl_gw_noise"][0] + ntmpar +1 + noise_dims["pl_gw_noise"][1]

    xhat = f.current_state.xhat
    ab_coeffs = xhat[p0:p1]/f.current_state.norm[p0:p1]

    Tspan = np.max(psr.toas) - np.min(psr.toas)

    f_coeffs = np.linspace(1/Tspan, ((p1-p0)/2)/Tspan, int((p1-p0)/2))


    return ab_coeffs, f_coeffs



ab, f = co_coll(maindir, pulsar)


np.save(maindir+"/"+pulsar+"_ab_coeffs.npy", ab)
np.save(maindir+"/"+pulsar+"_freq_coeffs.npy", f)