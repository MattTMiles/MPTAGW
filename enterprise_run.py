# This script runs enterprise for individual pulsars, or for an entire gravitational wave source depending on commands

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

import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
from enterprise_warp import bilby_warp

import bilby
import argparse
import time

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
parser.add_argument("-partim", dest="partim", help="Par and tim files for the pulsars to be used",required = False)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-noisefile", type = str, dest="noisefile", help="The noisefile used for the noise analysis.", required = False)
parser.add_argument("-noise_search", type = str.lower, nargs="+",dest="noise_search", help="The noise parameters to search over. Timing model is default. Include as '-noise_search noise1 noise2 noise3' etc. The _c variations of the noise redirects the noise to the constant noisefile values", \
    choices={"efac", "equad", "ecorr", "red", "efac_c", "equad_c", "ecorr_c", "red_c", "dm", "chrom", "chrom_c", "dm_c", "gw", "gw_const_gamma", "gw_c"})
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc"}, required=True)
parser.add_argument("-pool",dest="pool", type=int, help="Number of cores to request")
args = parser.parse_args()

pulsar = str(args.pulsar)
results_dir = str(args.results)
noisefile = args.noisefile
noise = args.noise_search
sampler = args.sampler
pool = args.pool
partim = args.partim

psrlist=None
if pulsar != "None":
    psrlist = [pulsar]

## Static data directory at the moment 
datadir = partim
if not os.path.isdir(datadir):
    datadir = '../partim'
print(datadir)

parfiles = sorted(glob.glob(datadir + '/*par'))
timfiles = sorted(glob.glob(datadir + '/*tim'))

# filter
if psrlist is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]

## Read into enterprise objects
psrs = []
ephemeris = 'DE438' # Static as not using bayesephem
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)
    time.sleep(2)

## Get parameter noise dictionary
params = {}
if noisefile is not None:
    with open(noisefile, 'r') as fp:
        params.update(json.load(fp))

## Find the time span to set the GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

## Defining selection by observing backend
selection = selections.Selection(selections.by_backend)

## Define a dm noise to use
def dm_noise(log10_A,gamma,Tspan,components=30,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = 30
    if option=="powerlaw":
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, components=components)
    #elif option=="turnover":
    #  fc = parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1])
    #  pl = powerlaw_bpl(log10_A=log10_A, gamma=gamma, fc=fc,
    #                    components=components)
    dm_basis = utils.createfourierdesignmatrix_dm(nmodes = components,
                                                  Tspan=Tspan)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

    return dmn


## Defining the noise parameters

if "efac" in noise:
    efac = parameter.Uniform(0,10)
if "efac_c" in noise:
    efac = parameter.Constant()
if "equad" in noise:
    equad = parameter.Uniform(-10,0) 
if "equad_c" in noise:
    equad = parameter.Constant()
if "ecorr" in noise:
    ecorr = parameter.Uniform(-10,0) 
if "ecorr_c" in noise:
    ecorr = parameter.Constant()

if "red" in noise:
    log10_A_red = parameter.Uniform(-20, -11)
    gamma_red = parameter.Uniform(0, 7)
if "red_c" in noise:
    log10_A_red = parameter.Constant()
    gamma_red = parameter.Constant()

if "dm" in noise:
    log10_A_dm = parameter.Uniform(-20, -11)
    gamma_dm = parameter.Uniform(0, 7)
if "dm_c" in noise:
    log10_A_dm = parameter.Constant()
    gamma_dm = parameter.Constant()

if "chrom" in noise:
    log10_A_chrom_prior = parameter.Uniform(-20, -11)
    gamma_chrom_prior = parameter.Uniform(0, 7)
    chrom_gp_idx = parameter.Uniform(0,7)
if "chrom_c" in noise:
    log10_A_chrom_prior = parameter.Constant()
    gamma_chrom_prior = parameter.Constant()
    chrom_gp_idx = parameter.Constant()

if "gw" in noise:
    log10_A_gw = parameter.Uniform(-18,-14)('log10_A_gw')
    gamma_gw = parameter.Uniform(0,7)('gamma_gw')
if "gw_const_gamma" in noise:
    log10_A_gw = parameter.Uniform(-18,-14)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')
if "gw_c" in noise:
    log10_A_gw = parameter.Constant(-14)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')


## Put together the signal model

tm = gp_signals.TimingModel(use_svd=True)
s = tm

if "efac" in noise or "efac_c" in noise:
    ef = white_signals.MeasurementNoise(efac=efac,log10_t2equad=equad, selection=selection)
    s += ef
if "ecorr" in noise or "ecorr_c" in noise:
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    s += ec

if "red" in noise or "red_c" in noise:
    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
    s += rn

if "dm" in noise or "dm_c" in noise:
    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
    s += dm


if "chrom" in noise or "chrom_c" in noise:
    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                gamma=gamma_chrom_prior)
    idx = chrom_gp_idx
    components = 30
    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                            idx=idx)
    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
    s += chrom

if "gw" in noise or "gw_c" in noise or "gw_const_gamma" in noise:
    gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
    s += gw

models = []
        
for p in psrs:    
    models.append(s(p))
    
pta = signal_base.PTA(models)
pta.set_default_params(params)

if sampler == "bilby":

    priors = bilby_warp.get_bilby_prior_dict(pta)

    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

    label = results_dir
    #nlive of 400 for now. Can be played with.
    if pulsar != "None":
        if pool is not None:

            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir='out/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=pool, verbose=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir='out/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=2, verbose=True)
    else:
        if pool is not None:

            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir='out/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=pool, verbose=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir='out/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=2, verbose=True)

    results.plot_corner()

elif sampler =="ptmcmc":
    
    pta.set_default_params(params)
    # set initial parameters drawn from prior
    x0 = np.hstack([p.sample() for p in pta.params])
    ndim = len(x0)

    cov = np.diag(np.ones(ndim) * 0.01**2)
    outDir = 'out/{0}_{1}/'.format(pulsar,results_dir)

    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, 
                    outDir=outDir, resume=True)

    N = int(1e5)  # This will have to be played around with a bit
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )


