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

import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

from collections import OrderedDict

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
#sys.path.insert(0, '/fred/oz002/rshannon/enterprise_warp/')
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
    choices={"efac", "equad", "ecorr", "red", "efac_c", "equad_c", "ecorr_c", "ecorr_check", "red_c", "dm", "chrom", "chrom_c","chrom_cidx", "dm_c", "gw", "gw_const_gamma", "gw_c", "dm_wide", "dm_wider", "red_wide", "chrom_wide", "chrom_cidx_wide", "efac_wide",\
        "band_low","band_low_c","band_high","band_high_c", "band_high_wide", "spgw", "spgwc", "spgwc_18", "pm_wn", "pm_wn_no_equad", "pm_wn_sw"})
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc"}, required=True)
parser.add_argument("-pool",dest="pool", type=int, help="Number of cores to request (default=1)")
parser.add_argument("-nlive", dest="nlive", type=int, help="Number of nlive points to use (default=1000)")
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-sse", dest="sse", type = str.upper, help=r"Choose an alternative solar system ephemeris to use (SSE). Default is DE438.", required = False)

args = parser.parse_args()

pulsar = str(args.pulsar)
results_dir = str(args.results)
noisefile = args.noisefile
noise = args.noise_search
sampler = args.sampler
pool = args.pool
nlive=args.nlive
partim = args.partim
custom_results = str(args.alt_dir)
sse = str(args.sse)

psrlist=None
if pulsar != "None":
    psrlist = [pulsar]

## Static data directory at the moment 
datadir = partim
if not os.path.isdir(datadir):
    datadir = '../partim'
print(datadir)

parfiles = sorted(glob.glob(datadir + '/*par'))
#timfiles = sorted(glob.glob(datadir + '/*clean_sn.tim'))
timfiles = sorted(glob.glob(datadir + '/*tim'))

# filter
if psrlist is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]

## Read into enterprise objects
psrs = []
if sse is not None and sse != "" and sse != "None":
    ephemeris = sse
else:
    ephemeris = 'DE440' # Static as not using bayesephem

for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris, drop_t2pulsar=False)
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

def pta_split(freqs, backend_flags):

    arr = np.array(['UWL' in val for val in backend_flags]).squeeze()
    d =  dict(zip(['40CM', '20CM', '10CM', '40CM_uwl', '20CM_uwl', '10CM_uwl'],
            [(702 < freqs) * (960 > freqs) * ~arr, (960 < freqs) * (freqs < 2048) * ~arr, (2048 < freqs) * (freqs < 4032) * ~arr,
            (702 < freqs) * (960 > freqs) *  arr, (960 < freqs) * (freqs < 2048) *  arr, (2048 < freqs) * (freqs < 4032) *  arr]))
    delkeys = []
    for key in d.keys():
        if np.sum(d[key]) == 0:  # all False
            delkeys.append(key)
    for key in delkeys:
        del d[key]

    flagvals = np.unique(backend_flags)
    noUWL = [ "UWL" not in x for x in flagvals ]
    flagvals_noUWL = flagvals[noUWL]
    temp =  {val: backend_flags == val for val in flagvals_noUWL}
    d.update(temp)
    return d


#ecorr_selection = selections.Selection(pta_split)
ecorr_selection = selections.Selection(selections.by_backend)
#ecor_selection = pta_split
selection = selections.Selection(selections.by_backend)

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

if "efac" in noise:
    efac = parameter.Uniform(0.1,5)
if "efac_wide" in noise:
    efac = parameter.Uniform(0.1,10)
if "efac_c" in noise:
    efac = parameter.Constant()
if "equad" in noise:
    equad = parameter.Uniform(-10,-1) 
if "equad_c" in noise:
    equad = parameter.Constant()
if "ecorr" in noise:
    ecorr = parameter.Uniform(-10,-1) 
if "ecorr_c" in noise:
    ecorr = parameter.Constant()
if "ecorr_check" in noise and "ecorr_c" not in noise:
    ecorr = parameter.Uniform(-10,-1)

if "red" in noise:
    log10_A_red = parameter.Uniform(-20, -11)
    gamma_red = parameter.Uniform(0, 7)
if "red_wide" in noise:
    log10_A_red = parameter.Uniform(-20, -11)
    gamma_red = parameter.Uniform(0, 12)
if "red_c" in noise:
    log10_A_red = parameter.Constant()
    gamma_red = parameter.Constant()

if "dm" in noise:
    log10_A_dm = parameter.Uniform(-20, -11)
    gamma_dm = parameter.Uniform(0, 7)
if "dm_wide" in noise:
    log10_A_dm = parameter.Uniform(-20, -11)
    gamma_dm = parameter.Uniform(0, 12)
if "dm_wider" in noise:
    log10_A_dm = parameter.Uniform(-20, -11)
    gamma_dm = parameter.Uniform(0, 16)
if "dm_c" in noise:
    log10_A_dm = parameter.Constant()
    gamma_dm = parameter.Constant()

if "chrom" in noise:
    log10_A_chrom_prior = parameter.Uniform(-20, -11)
    gamma_chrom_prior = parameter.Uniform(0, 7)
    chrom_gp_idx = parameter.Uniform(0,7)
if "chrom_wide" in noise:
    log10_A_chrom_prior = parameter.Uniform(-20, -11)
    gamma_chrom_prior = parameter.Uniform(0, 14)
    chrom_gp_idx = parameter.Uniform(0,14)
if "chrom_c" in noise:
    log10_A_chrom_prior = parameter.Constant()
    gamma_chrom_prior = parameter.Constant()
    chrom_gp_idx = parameter.Constant()
if "chrom_cidx" in noise:
    log10_A_chrom_prior = parameter.Uniform(-20, -11)
    gamma_chrom_prior = parameter.Uniform(0, 7)
    chrom_gp_idx = parameter.Constant(4)
if "chrom_cidx_wide" in noise:
    log10_A_chrom_prior = parameter.Uniform(-20, -11)
    gamma_chrom_prior = parameter.Uniform(0, 14)
    chrom_gp_idx = parameter.Constant(4)

if "addchrom" in noise:
    log10_A_add_chrom_prior = parameter.Uniform(-20, -11)
    gamma_add_chrom_prior = parameter.Uniform(0, 14)
    add_chrom_gp_idx = parameter.Uniform(0,14)

if "gw" in noise:
    log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
    gamma_gw = parameter.Uniform(0,7)('gamma_gw')
if "gw_c" in noise:
    log10_A_gw = parameter.Constant(-14)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')
if "gw_const_gamma" in noise:
    log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')

if "band_low"in noise:
    log10_A_bn = parameter.Uniform(-20, -11)
    gamma_bn = parameter.Uniform(0, 7)
if "band_low_c" in noise:
    log10_A_bn = parameter.Constant()
    gamma_bn = parameter.Constant()

if "band_high"in noise:
    log10_A_bn = parameter.Uniform(-20, -11)
    gamma_bn = parameter.Uniform(0, 7)

if "band_high_wide"in noise:
    log10_A_bn = parameter.Uniform(-20, -11)
    gamma_bn = parameter.Uniform(0, 14)

if "band_high_c" in noise:
    log10_A_bn = parameter.Constant()
    gamma_bn = parameter.Constant()



## Put together the signal model

#tm = gp_signals.TimingModel(use_svd=True)

psr.tmparams_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
for key in psr.tmparams_orig:

    val = psr.t2pulsar[key].val
    err = psr.t2pulsar[key].err
    psr.tmparams_orig[key] = (val, err)
    if err == 0:
        raise("No error found!")
print(psr.tmparams_orig)
print("Varying:")
print(psr.tmparams_orig.keys())
tm = timing.timing_block(tmparam_list=psr.tmparams_orig.keys(), linear=True)


s = tm

if "efac" in noise or "efac_c" in noise or "efac_wide" in noise:
    if "equad" in noise or "equad_c" in noise:
        ef = white_signals.MeasurementNoise(efac=efac,log10_t2equad=equad, selection=selection)
        s += ef
    else:
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        s += ef
if "ecorr" in noise and "ecorr_c" not in noise:
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
    s += ec

#### IPTA ECORR MODEL NEEDED FOR PPTA UWL DATA. HAS ISSUES THAT IT GETS INCLUDED IN THE BASIS MODEL AND CAN BREAK
#    ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=ecorr_selection)
#    s += ec
####

if "ecorr_c" in noise and "ecorr_check" not in noise:
    if pulsar+"_KAT_MKBF_log10_ecorr" in params.keys():
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
        s += ec

if "ecorr_check" in noise:
    if pulsar+"_KAT_MKBF_log10_ecorr" in params.keys():
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
        s += ec

if "red" in noise or "red_c" in noise or "red_wide" in noise:
    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
    s += rn

if "dm" in noise or "dm_c" in noise or "dm_wide" in noise or "dm_wider" in noise:
    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
    s += dm

if "chrom" in noise or "chrom_wide" in noise or "chrom_c" in noise or "chrom_cidx" in noise or "chrom_cidx_wide" in noise:
    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                gamma=gamma_chrom_prior)
    idx = chrom_gp_idx
    components = 30
    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                            idx=idx)
    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
    s += chrom

if "addchrom" in noise:
    chrom_model = utils.powerlaw(log10_A=log10_A_add_chrom_prior,
                                gamma=gamma_add_chrom_prior)
    idx = add_chrom_gp_idx
    components = 30
    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                            idx=idx)
    add_chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='add_chrom_gp')
    s += add_chrom

if "gw" in noise or "gw_c" in noise or "gw_const_gamma" in noise:
    gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
    s += gw

if "band_low"in noise or "band_low_c" in noise:
    #max_cadence = 60  # days
    #band_components = int(Tspan / (max_cadence*86400))
    band_components = 30
    bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
    bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                selection=low_freq, name='low_band_noise')
    s += bnl

if "band_high"in noise or "band_high_wide" in noise or "band_high_c" in noise:
    #max_cadence = 60  # days
    #band_components = int(Tspan / (max_cadence*86400))
    band_components = 30
    bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
    bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                selection=high_freq, name='high_band_noise')
    s += bnh

if "spgw" in noise or "spgwc" in noise or "spgwcm" in noise or "spgwc_18" in noise:
    ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_active_noise.json"))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
    
    for pm in psrmodels:

        if pm == "RN" or pm == "RED":
            log10_A_red = parameter.Uniform(-20, -11)
            gamma_red = parameter.Uniform(0, 7)
            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
            s += rn

        if pm == "RNWIDE":
            log10_A_red = parameter.Uniform(-20, -11)
            gamma_red = parameter.Uniform(0, 14)
            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
            s += rn

        if pm =="DM":
            log10_A_dm = parameter.Uniform(-20, -11)
            gamma_dm = parameter.Uniform(0, 7)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
            s += dm
        
        if pm == "DMWIDE":
            log10_A_dm = parameter.Uniform(-20, -11)
            gamma_dm = parameter.Uniform(0, 14)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
            s += dm

        if pm == "CHROM":
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

        if pm == "CHROMWIDE":
            log10_A_chrom_prior = parameter.Uniform(-20, -11)
            gamma_chrom_prior = parameter.Uniform(0, 14)
            chrom_gp_idx = parameter.Uniform(0,14)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 30
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
            s += chrom

        if pm == "CHROMCIDX":
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

        if pm == "CHROMCIDXWIDE":
            log10_A_chrom_prior = parameter.Uniform(-20, -11)
            gamma_chrom_prior = parameter.Uniform(0, 14)
            chrom_gp_idx = parameter.Constant(4)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 30
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
            s += chrom

        if pm == "BL":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 7)
            band_components = 30
            bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                        selection=low_freq, name='low_band_noise')
            s += bnl

        if pm == "BLWIDE":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 14)
            band_components = 30
            bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                        selection=low_freq, name='low_band_noise')
            s += bnl

        if pm == "BH":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 7)
            band_components = 30
            bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                        selection=high_freq, name='high_band_noise')
            s += bnh

        if pm == "BHWIDE":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 14)
            band_components = 30
            bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                        selection=high_freq, name='high_band_noise')
            s += bnh

    if "spgwc" in noise:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')

    #This is a place holder for a quick monopole check, this is not finished, activating it won't do anything
    elif "spgwcm" in noise:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')

    elif "spgwc_18" in noise:
        log10_A_gw = parameter.Uniform(-18,-17.8)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    else:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Uniform(0,7)('gamma_gw')

    gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
    s += gw

if "pm_wn" in noise or "pm_wn_no_equad" in noise or "pm_wn_sw" in noise:
    pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_active_noise.json"))
    wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_ALL_NOISE_EQUAD_CHECKED.json"))
    keys = list(pmev_json.keys())
    wnkeys = list(wn_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
    wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
    # Check through the possibilities and add them as appropriate

    for pm in psrmodels:

        if pm == "RN" or pm == "RED":
            log10_A_red = parameter.Uniform(-20, -11)
            gamma_red = parameter.Uniform(0, 7)
            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
            s += rn

        if pm == "RNWIDE":
            log10_A_red = parameter.Uniform(-20, -11)
            gamma_red = parameter.Uniform(0, 14)
            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
            s += rn

        if pm =="DM":
            log10_A_dm = parameter.Uniform(-20, -11)
            gamma_dm = parameter.Uniform(0, 7)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
            s += dm
        
        if pm == "DMWIDE":
            log10_A_dm = parameter.Uniform(-20, -11)
            gamma_dm = parameter.Uniform(0, 14)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
            s += dm

        if pm == "CHROM":
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

        if pm == "CHROMWIDE":
            log10_A_chrom_prior = parameter.Uniform(-20, -11)
            gamma_chrom_prior = parameter.Uniform(0, 14)
            chrom_gp_idx = parameter.Uniform(0,14)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 30
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
            s += chrom

        if pm == "CHROMCIDX":
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

        if pm == "CHROMCIDXWIDE":
            log10_A_chrom_prior = parameter.Uniform(-20, -11)
            gamma_chrom_prior = parameter.Uniform(0, 14)
            chrom_gp_idx = parameter.Constant(4)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 30
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
            s += chrom

        if pm == "BL":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 7)
            band_components = 30
            bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                        selection=low_freq, name='low_band_noise')
            s += bnl

        if pm == "BLWIDE":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 14)
            band_components = 30
            bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                        selection=low_freq, name='low_band_noise')
            s += bnl

        if pm == "BH":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 7)
            band_components = 30
            bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                        selection=high_freq, name='high_band_noise')
            s += bnh

        if pm == "BHWIDE":
            log10_A_bn = parameter.Uniform(-20, -11)
            gamma_bn = parameter.Uniform(0, 14)
            band_components = 30
            bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
            bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                        selection=high_freq, name='high_band_noise')
            s += bnh

    if not "pm_wn_no_equad" in noise:
        efac = parameter.Uniform(0.1,5)
        if "t2equad" in wnmodels:
            equad = parameter.Uniform(-10,-1)
            ef = white_signals.MeasurementNoise(efac=efac,log10_t2equad=equad, selection=selection)
            s += ef
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-1)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec
        
    elif "pm_wn_no_equad" in noise:
        efac = parameter.Uniform(0.1,5)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-1)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec

    if "pm_wn_sw" in noise:

        n_earth = parameter.Uniform(0, 20)
        deter_sw = solar_wind(n_earth=n_earth)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

        Tspan = psr.toas.max() - psr.toas.min()
        max_cadence = 60
        sw_components = int(Tspan / (max_cadence*86400))

        log10_A_sw = parameter.Uniform(-10, 1)
        gamma_sw = parameter.Uniform(-4, 4)
        sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
        sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

        sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

        s += sw


models = []
        
for p in psrs:    
    models.append(s(p))
    
pta = signal_base.PTA(models)
pta.set_default_params(params)


if sampler == "bilby":

    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out"

    outDir=header_dir+'/{0}_{1}'.format(pulsar,results_dir)
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

    
    if pulsar != "None":
        if pool is not None:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    else:
        if pool is not None:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='dynesty', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

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
    outDir = header_dir+'/{0}_{1}/'.format(pulsar,results_dir)
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
    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out_ppc"

    outDir='/fred/oz002/users/mmiles/MPTA_GW/enterprise/'+header_dir+'/{0}_{1}'.format(pulsar,results_dir)
    
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
    if pulsar != "None":
        if pool is not None:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    else:
        if pool is not None:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='PyPolyChord', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    results.plot_corner()


