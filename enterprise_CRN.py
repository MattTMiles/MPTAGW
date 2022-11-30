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
from enterprise_extensions.blocks import common_red_noise_block
from enterprise_extensions import hypermodel
from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as OS

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
parser.add_argument("-partim", dest="partim", help="Par and tim files for the pulsars to be used",required = False)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-noisefile", type = str, dest="noisefile", help="The noisefile used for the noise analysis.", required = False)
parser.add_argument("-noise_search", type = str.lower, nargs="+",dest="noise_search",help="Which GW search to do. If multiple are chosen it will combine them i.e. HD + monopole", \
    choices={"pl_nocorr_freegam","pl_nocorr_fixgam","bpl_nocorr_freegam","freespec_nocorr","pl_orf_bins","pl_orf_spline","pl_hd_fixgam","pl_hdnoauto_fixgam",\
        "freespec_hd","pl_dp","freespec_dp","pl_mono","freespec_monopole"})
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc"}, required=True)
parser.add_argument("-pool",dest="pool", type=int, help="Number of cores to request (default=1)")
parser.add_argument("-nlive", dest="nlive", type=int, help="Number of nlive points to use (default=1000)")
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-psrlist", dest="psrlist", nargs="+", help=r"List of pulsars to use", required = False)

args = parser.parse_args()

results_dir = str(args.results)
noisefile = args.noisefile
noise = args.noise_search
sampler = args.sampler
pool = args.pool
nlive=args.nlive
partim = args.partim
custom_results = str(args.alt_dir)
psr_list = args.psrlist

psrlist=None
if psr_list is not None and psr_list != "":
    if type(psr_list) != list:
        psrlist=[ x.strip("\n") for x in open(str(psr_list[0])).readlines() ]
    else:
        psrlist = list(psr_list)

## Static data directory at the moment 
datadir = partim
if not os.path.isdir(datadir):
    datadir = '../partim'
print(datadir)

parfiles = sorted(glob.glob(datadir + '/*par'))
timfiles = sorted(glob.glob(datadir + '/*clean_sn.tim'))

# filter
if psrlist is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('_')[0] in psrlist]

## Read into enterprise objects
psrs = []
ephemeris = 'DE438' # Static as not using bayesephem
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)
    time.sleep(3)

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

#def powerlaw_no_components(f, log10_A=-16, gamma=5):
#    df = np.diff(np.concatenate((np.array([0]), f)))
#    return (
#        (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * df
#    )
## Define a dm noise to use
def dm_noise(log10_A,gamma,Tspan,components=30,option="powerlaw"):
    """
    A term to account for stochastic variations in DM. It is based on spin
    noise model, with Fourier amplitudes depending on radio frequency nu
    as ~ 1/nu^2.
    """
    nfreqs = 30
    if option=="powerlaw":
      #pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, components=components)
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
      #pl = enterprise.signals.gp_priors.powerlaw_no_components(log10_A=log10_A, gamma=gamma)

    #elif option=="turnover":
    #  fc = parameter.Uniform(self.params.sn_fc[0],self.params.sn_fc[1])
    #  pl = powerlaw_bpl(log10_A=log10_A, gamma=gamma, fc=fc,
    #                    components=components)
    dm_basis = utils.createfourierdesignmatrix_dm(nmodes = components,
                                                  Tspan=Tspan)
    dmn = gp_signals.BasisGP(pl, dm_basis, name='dm_gp')

    return dmn


def low_frequencies(freqs):
    """Selection for obs frequencies <=960MHz"""
    return dict(zip(['low'], [freqs <= 1284]))

def high_frequencies(freqs):
    """Selection for obs frequencies >=2048MHz"""
    return dict(zip(['high'], [freqs > 1284]))

low_freq = selections.Selection(low_frequencies)
high_freq = selections.Selection(high_frequencies)

## Defining the noise parameters


efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant() 

components = 30

for i,n in enumerate(noise):
    #power law, free spectral index, no correlations
    if "pl_nocorr_freegam" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=None,
                             components=components, orf=None, name='gw')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                             components=components, orf=None, name='gw')

    #power law, fixed 13/3 spectral index, no correlations
    if "pl_nocorr_fixgam" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.3333,
                             components=components, orf=None, name='gw')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.3333,
                             components=components, orf=None, name='gw')

    #broken power law, free spectral index, no correlations
    if "bpl_nocorr_freegam" == n:
        if i==0:
            crn = common_red_noise_block(psd='broken_powerlaw', prior='log-uniform',
                                components=30, orf=None, name='gw')
        else:
            crn += common_red_noise_block(psd='broken_powerlaw', prior='log-uniform',
                                components=30, orf=None, name='gw')

    #free-spectrum power law, no correlations
    if "freespec_nocorr" == n:
        if i==0:
            crn = common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 60,
                             orf = None, name = 'gw')
        else:
            crn += common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 60,
                             orf = None, name = 'gw')

    #Correlated CRN models - free ORF
    if "pl_orf_bins" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                                components=components, orf='bin_orf', name='gw_bins')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                                components=components, orf='bin_orf', name='gw_bins')
    
    if "pl_orf_spline" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                                components=components, orf='spline_orf', name='gw_bins')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                             components=components, orf='spline_orf', name='gw_bins')

    # Powerlaw Hellings-Downs
    if "pl_hd_fixgam" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='hd', name='gwb', gamma_val = 4.333)
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='hd', name='gwb', gamma_val = 4.333)
    
    # Powerlaw fixed-gamma Hellings-Downs cross-correlations only
    if "pl_hdnoauto_fixgam" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                              components=components, orf='zero_diag_hd', name='gwb', gamma_val = 4.333)
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                              components=components, orf='zero_diag_hd', name='gwb', gamma_val = 4.333)

    # Free-spectrum Hellings-Downs
    if "freespec_hd" == n:
        if i==0:
            crn = common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 60,
                             orf = 'hd', name = 'gwb')
        else:
            crn += common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 60,
                             orf = 'hd', name = 'gwb')

    # Correlated CRN models - Dipole
    if "pl_dp" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                              components=components, orf='dipole', name='gw_dipole')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                              components=components, orf='dipole', name='gw_dipole')
    
    if "freespec_dp" == n:
        if i==0:
            crn = common_red_noise_block(psd='spectrum', prior='log-uniform',
                              components=60, orf='dipole', name='gw_dipole')
        else:
            crn += common_red_noise_block(psd='spectrum', prior='log-uniform',
                              components=60, orf='dipole', name='gw_dipole')

    # Correlated CRN models - Monopole
    # Powerlaw Monopole
    if "pl_mono" == n:
        if i==0:
            crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='monopole', name='gw_monopole')
        else:
            crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='monopole', name='gw_monopole')

    if "freespec_monopole" == n:
        if i==0:
            crn = common_red_noise_block(psd='spectrum', prior='log-uniform',
                              components=20, orf='monopole', name='gw_monopole')
        else:
            crn += common_red_noise_block(psd='spectrum', prior='log-uniform',
                              components=20, orf='monopole', name='gw_monopole')


## Put together the signal model
models = []
        
for pulsar in psrs:

    tm = gp_signals.TimingModel(use_svd=True)
    s = tm

    ef = white_signals.MeasurementNoise(efac=efac,log10_t2equad=equad, selection=selection)
    s += ef

    if pulsar.name+"_KAT_MKBF_log10_ecorr" in params.keys():
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
        s += ec

    ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_active_noise.json"))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar.name in psr_model ][0].split("_")[1:]

    # Check through the possibilities and add them as appropriate
    if "RN" in psrmodels or "RED" in psrmodels:
        log10_A_red = parameter.Uniform(-20, -11)
        gamma_red = parameter.Uniform(0, 7)
        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
        s += rn

    if "DM" in psrmodels:
        log10_A_dm = parameter.Uniform(-20, -11)
        gamma_dm = parameter.Uniform(0, 7)
        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
        s += dm

    if "CHROM" in psrmodels:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Uniform(0,7)
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
        s += chrom

    if "CHROMCIDX" in psrmodels:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Constant(4)
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
        s += chrom

    if "BL" in psrmodels:
        log10_A_bn = parameter.Uniform(-20, -11)
        gamma_bn = parameter.Uniform(0, 7)
        band_components = 30
        bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                    selection=low_freq, name='low_band_noise')
        s += bnl

    if "BH" in psrmodels:
        log10_A_bn = parameter.Uniform(-20, -11)
        gamma_bn = parameter.Uniform(0, 7)
        band_components = 30
        bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                    selection=high_freq, name='high_band_noise')
        s += bnh

    s += crn

    models.append(s(pulsar))
    
pta = signal_base.PTA(models)
pta.set_default_params(params)


if sampler == "bilby":

    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out"

    outDir=header_dir+'/{0}'.format(results_dir)
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
    priors = bilby_warp.get_bilby_prior_dict(pta)

    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

    label = results_dir


    if pool is not None:
        if nlive is not None:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
    else:
        if nlive is not None:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    results.plot_corner()

elif sampler =="ptmcmc":
    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out_ptmcmc"
    
    pta.set_default_params(params)
    # set initial parameters drawn from prior
    x0 = np.hstack([p.sample() for p in pta.params])
    ndim = len(x0)

    cov = np.diag(np.ones(ndim) * 0.01**2)
    outDir = header_dir+'/{0}/'.format(results_dir)
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
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, 
                    outDir=outDir, resume=True)

    N = int(1e6)  # This will have to be played around with a bit
    x0 = np.hstack([p.sample() for p in pta.params])
    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )

elif sampler =="ppc":
    priors = bilby_warp.get_bilby_prior_dict(pta)

    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

    label = results_dir
    if custom_results is not None and custom_results != "" and custom_results !="None":
        header_dir = custom_results
    else:
        header_dir = "out_ppc"
    
    outDir='/fred/oz002/users/mmiles/MPTA_GW/enterprise/'+header_dir+'/{0}'.format(results_dir)
    
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

    if pool is not None:
        if nlive is not None:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
    else:
        if nlive is not None:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
        else:
            results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    results.plot_corner()


