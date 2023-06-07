# This script runs enterprise for individual pulsars, or for an entire gravitational wave source

from __future__ import division

import os, glob, json, pickle, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

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
from enterprise_extensions.blocks import common_red_noise_block
import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
#sys.path.insert(0, '/fred/oz002/rshannon/enterprise_warp/')
from enterprise_warp import bilby_warp

import bilby
import argparse
import time
import faulthandler
import dill


faulthandler.enable()
## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")
#parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
parser.add_argument("-pta", dest="pta", help="Enterprise PTA object to load in for the PTMCMC run",required = True)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-nlive", dest="nlive", type=int, help="Number of nlive points to use (default=1000)")

args = parser.parse_args()

ptafile = str(args.pta)
custom_results = str(args.alt_dir)
results_dir = str(args.results)
nlive=args.nlive

#while True:
#    try:
pta = dill.load(open(ptafile,"rb"))
#    except SystemError:
#        pass
#    break


if custom_results is not None and custom_results != "":
    header_dir = custom_results
else:
    header_dir = "out_ppc"


priors = bilby_warp.get_bilby_prior_dict(pta)

parameters = dict.fromkeys(priors.keys())
likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

outDir='/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/'+header_dir+'/{0}'.format(results_dir)
print(outDir)
try:
    os.mkdir(outDir)
except:
    pass
try:
    with open(outDir+"/run_summary.txt","w") as f:
        print(pta.summary(), file=f)
except:
    pass
try:
    print(pta.params)
    filename = outDir + "/pars.txt"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "a") as f:
    #f = open(filename, "a")
        for par in pta.param_names:
            f.write(par + '\n')
except:
    pass


if nlive is not None:
    results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=results_dir, sampler='PyPolyChord', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
else:
    results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=results_dir, sampler='PyPolyChord', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

results.plot_corner()

