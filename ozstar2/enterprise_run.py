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
    choices={"efac", "equad", "t2_equad", "ecorr", "red", "efac_c", "equad_c", "ecorr_c", "ecorr_check", "red_c", "dm", "chrom", "chrom_c","chrom_cidx","high_comp_chrom", "dm_c", "gw", "gw_const_gamma","gw_const_gamma_wide", "lin_exp_gw_const_gamma_wide", "gw_c", "dm_wide", "dm_wider", "red_wide", "chrom_wide", "chrom_cidx_wide", "efac_wide",\
        "band_low","band_low_c","band_high","band_high_c", "band_high_wide", "spgw", "spgwc", "spgwc_18", "pm_wn", "pm_wn_no_equad", "pm_wn_sw","pm_wn_altpar", "pm_wn_no_equad_altpar", "wn_sw", "wn_tester", "chrom_annual", "sw", "swdet", "free_spgw", "free_spgwc", "hfred", "pm_wn_hc", "pm_wn_sw_hc", "pm_wn_sw_hc_noeq","pm_wn_hc_noeq"})
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc"}, required=True)
parser.add_argument("-pool",dest="pool", type=int, help="Number of cores to request (default=1)")
parser.add_argument("-nlive", dest="nlive", type=int, help="Number of nlive points to use (default=1000)")
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-sse", dest="sse", type = str.upper, help=r"Choose an alternative solar system ephemeris to use (SSE). Default is DE440.", required = False)

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

@signal_base.function
def chrom_yearly_sinusoid(toas, freqs, log10_Amp, phase, idx):
    """
    Chromatic annual sinusoid.
    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid
    :param idx: index of chromatic dependence
    :return wf: delay time-series [s]
    """

    wf = 10**log10_Amp * np.sin(2 * np.pi * const.fyr * toas + phase)
    return wf * (1400 / freqs) ** idx

def low_frequencies(freqs):
    """Selection for obs frequencies <=960MHz"""
    return dict(zip(['low'], [freqs <= 1284]))

def high_frequencies(freqs):
    """Selection for obs frequencies >=2048MHz"""
    return dict(zip(['high'], [freqs > 1284]))


def chrom_splitter(freqs):
    """Selection for obs frequencies in 4 subbands"""

    d =  dict(zip(['LOW', 'MID_1', 'MID_2', 'HIGH'],
            [(freqs < 1070), (1070 <= freqs) * (freqs < 1284), (1284 <= freqs) * (freqs < 1498), (freqs >=1712)]))
    
    return d


chrom_split = selections.Selection(chrom_splitter)

low_freq = selections.Selection(low_frequencies)
high_freq = selections.Selection(high_frequencies)

## Defining the noise parameters

for n in noise:

    if "efac" == n:
        efac = parameter.Uniform(0.1,5)
    if "efac_wide" == n:
        efac = parameter.Uniform(0.1,10)
    if "efac_c" == n:
        efac = parameter.Constant()
    if "equad" == n:
        equad = parameter.Uniform(-10,-1) 
    if "t2_equad" == n:
        t2_equad = parameter.Uniform(-10,-1) 
    if "equad_c" == n:
        equad = parameter.Constant()
    if "ecorr" == n:
        ecorr = parameter.Uniform(-10,-1) 
    if "ecorr_c" == n:
        ecorr = parameter.Constant()
    if "ecorr_check" == n:
        ecorr = parameter.Uniform(-10,-1)

    if "red" == n:
        log10_A_red = parameter.Uniform(-20, -11)
        gamma_red = parameter.Uniform(0, 7)
    if "red_wide" == n:
        log10_A_red = parameter.Uniform(-20, -11)
        gamma_red = parameter.Uniform(0, 12)
    if "red_c" == n:
        log10_A_red = parameter.Constant()
        gamma_red = parameter.Constant()

    if "hfred" == n:
        log10_A_hfred = parameter.Uniform(-20, -11)
        gamma_hfred = parameter.Uniform(0, 7)


    if "dm" == n:
        log10_A_dm = parameter.Uniform(-20, -11)
        gamma_dm = parameter.Uniform(0, 7)
    if "dm_wide" == n:
        log10_A_dm = parameter.Uniform(-20, -11)
        gamma_dm = parameter.Uniform(0, 12)
    if "dm_wider" == n:
        log10_A_dm = parameter.Uniform(-20, -11)
        gamma_dm = parameter.Uniform(0, 16)
    if "dm_c" == n:
        log10_A_dm = parameter.Constant()
        gamma_dm = parameter.Constant()

    if "chrom" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Uniform(0,7)
    if "high_comp_chrom" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Uniform(0,7)
    if "chrom_wide" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 14)
        chrom_gp_idx = parameter.Uniform(0,14)
    if "chrom_c" == n:
        log10_A_chrom_prior = parameter.Constant()
        gamma_chrom_prior = parameter.Constant()
        chrom_gp_idx = parameter.Constant()
    if "chrom_cidx" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Constant(4)
    if "chrom_cidx_wide" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 14)
        chrom_gp_idx = parameter.Constant(4)
    if "chromsplit" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Uniform(0,7)

    if "addchrom" == n:
        log10_A_add_chrom_prior = parameter.Uniform(-20, -11)
        gamma_add_chrom_prior = parameter.Uniform(0, 14)
        add_chrom_gp_idx = parameter.Uniform(0,14)
    if "chrom_annual" == n:
        log10_Amp_chrom1yr = parameter.Uniform(-20, -11)
        phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
        idx_chrom1yr = parameter.Uniform(0, 14)

    if "gw" == n:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Uniform(0,7)('gamma_gw')
    if "gw_c" == n:
        log10_A_gw = parameter.Constant(-14)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_const_gamma" == n:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_const_gamma_wide" == n:
        log10_A_gw = parameter.Uniform(-18,-9)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "lin_exp_gw_const_gamma_wide" == n:
        #log10_A_gw = bilby_warp.linearExponential(-18,-9,'log10_A_gw')
        log10_A_gw = parameter.LinearExp(-18,-9)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')

    if "band_low"== n:
        log10_A_bn = parameter.Uniform(-20, -11)
        gamma_bn = parameter.Uniform(0, 7)
    if "band_low_c" == n:
        log10_A_bn = parameter.Constant()
        gamma_bn = parameter.Constant()

    if "band_high"== n:
        log10_A_bn = parameter.Uniform(-20, -11)
        gamma_bn = parameter.Uniform(0, 7)

    if "band_high_wide"== n:
        log10_A_bn = parameter.Uniform(-20, -11)
        gamma_bn = parameter.Uniform(0, 14)

    if "band_high_c" == n:
        log10_A_bn = parameter.Constant()
        gamma_bn = parameter.Constant()



## Put together the signal model

tm = gp_signals.MarginalizingTimingModel(use_svd=True)

#tm = timing.timing_block()

s = tm

for n in noise:

    if "efac" == n or "efac_c" == n or "efac_wide" == n:
        if "t2_equad" in noise and "equad" not in noise:
            ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2_equad, selection=selection)
            s += ef
        elif "equad" in noise or "equad_c" in noise:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if "equad_c" in noise:
                if pulsar+"_KAT_MKBF_log10_t2equad" in params.keys() or pulsar+"_KAT_MKBF_log10_tnequad" in params.keys():
                    eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                    s += eq

            else:
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
    if "ecorr" == n:
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
        s += ec

    #### IPTA ECORR MODEL NEEDED FOR PPTA UWL DATA. HAS ISSUES THAT IT GETS INCLUDED IN THE BASIS MODEL AND CAN BREAK
    #    ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=ecorr_selection)
    #    s += ec
    ####

    if "ecorr_c" == n:
        if pulsar+"_KAT_MKBF_log10_ecorr" in params.keys():
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
            s += ec

    if "ecorr_check" == n:
        if pulsar+"_KAT_MKBF_log10_ecorr" in params.keys():
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
            s += ec

    if "red" == n or "red_c" == n or "red_wide" == n:
        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
        s += rn

    if "hfred" == n:
        hfpl = utils.powerlaw(log10_A=log10_A_hfred, gamma=gamma_hfred)
        hfrn = gp_signals.FourierBasisGP(spectrum=hfpl, components=120, Tspan=Tspan, name="hfred")
        s += hfrn

    if "dm" == n or "dm_c" == n or "dm_wide" == n or "dm_wider" == n:
        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
        s += dm

    if "chrom" == n or "chrom_wide" == n or "chrom_c" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
        s += chrom

    if "chrom_cidx" == n or "chrom_cidx_wide" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chromcidx_gp')
        s += chrom

    if "high_comp_chrom" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 60
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='highcomp_chrom_gp')
        s += chrom

    if "chromsplit" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, selection = chrom_split, name='chrom_gp_split')
        s += chrom

    if "addchrom" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_add_chrom_prior,
                                    gamma=gamma_add_chrom_prior)
        idx = add_chrom_gp_idx
        components = 30
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        add_chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='add_chrom_gp')
        s += add_chrom

    if "chrom_annual" == n:
        wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr,
                            phase=phase_chrom1yr, idx=idx_chrom1yr)
        chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
        s += chrom1yr

    if "gw" == n or "gw_c" == n or "gw_const_gamma" == n or "gw_const_gamma_wide" == n or "lin_exp_gw_const_gamma_wide" == n:
        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw

    if "band_low"== n or "band_low_c" == n:
        #max_cadence = 60  # days
        #band_components = int(Tspan / (max_cadence*86400))
        band_components = 30
        bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                    selection=low_freq, name='low_band_noise')
        s += bnl

    if "band_high"== n or "band_high_wide" == n or "band_high_c" == n:
        #max_cadence = 60  # days
        #band_components = int(Tspan / (max_cadence*86400))
        band_components = 30
        bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                    selection=high_freq, name='high_band_noise')
        s += bnh

    if "sw" == n:
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

    if "swdet" == n:
        n_earth = parameter.Uniform(0, 20)
        deter_sw = solar_wind(n_earth=n_earth)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

        sw = mean_sw

        s += sw

    if "spgw" == n or "spgwc" == n or "spgwcm" == n or "spgwc_18" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED":
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 7)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm =="DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm

            if pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh
            
            if pm == "SW":
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
            
            if pm == "SWDET":
                n_earth = parameter.Uniform(0, 20)
                deter_sw = solar_wind(n_earth=n_earth)
                mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

                sw = mean_sw

                s += sw
            
        if "SW" not in psrmodels and "SWDET" not in psrmodels:
            n_earth = parameter.Constant(4)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            sw = mean_sw

            s += sw

        if "spgwc" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')
        
        elif "spgwc_wide" == n:
            log10_A_gw = parameter.Uniform(-18,-9)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')

        #This is a place holder for a quick monopole check, this is not finished, activating it won't do anything
        elif "spgwcm" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')

        elif "spgwc_18" == n:
            log10_A_gw = parameter.Uniform(-18,-17.8)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')
        else:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Uniform(0,7)('gamma_gw')

        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw

    if "free_spgw" == n or "free_spgwc" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 7)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm =="DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm

            if pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Uniform(0,14)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 30
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 30
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh
            
            if pm == "SW":
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
            
            if pm == "SWDET":
                n_earth = parameter.Uniform(0, 20)
                deter_sw = solar_wind(n_earth=n_earth)
                mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

                sw = mean_sw

                s += sw
            
        if "SW" not in psrmodels and "SWDET" not in psrmodels:
            n_earth = parameter.Constant(4)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            sw = mean_sw

            s += sw


        if not "free_spgwc" == n:
            crn = blocks.common_red_noise_block(psd='spectrum', prior='log-uniform',Tspan = 122448047.42001152, components=10, orf=None, name='gw', delta_val=0)
            s += crn
        elif "free_spgwc" == n:
            #J1909 timespan
            Tspan = 122448047.42001152
            
            crn = blocks.common_red_noise_block(psd='spectrum', prior='log-uniform', Tspan = 122448047.42001152, gamma_val=4.33, components=10, orf=None, name='gw', delta_val=0)
            s += crn


    if "wn_sw" == n:
        wn_json = json.load(open(noisefile))
        wnkeys = list(wn_json.keys())
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

        efac = parameter.Uniform(0.1,5)
        if "t2equad" in wnmodels or "tnequad" in wnmodels:
            equad = parameter.Uniform(-10,-1)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            s += eq
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-1)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec

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



    if "wn_tester" == n:
        wn_json = json.load(open(noisefile))
        wnkeys = list(wn_json.keys())
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

        efac = parameter.Uniform(0.1,5)
        if "t2equad" in wnmodels or "tnequad" in wnmodels:
            equad = parameter.Uniform(-10,-1)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            s += eq
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-1)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec

    if "pm_wn" == n or "pm_wn_no_equad" == n or "pm_wn_sw" == n or "pm_wn_altpar" == n or "pm_wn_no_equad_altpar" == n:

        pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        wn_json = json.load(open(noisefile))
        keys = list(pmev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
        # Check through the possibilities and add them as appropriate

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 7)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm =="DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=30,option="powerlaw")
                s += dm

            if pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Uniform(0,14)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 30
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh
            

        if  "pm_wn_no_equad" != n:
            efac = parameter.Uniform(0.1,5)
            if "t2equad" in wnmodels or "tnequad" in wnmodels:
                equad = parameter.Uniform(-10,-1)
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            else:
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec
            
        elif "pm_wn_no_equad" == n:
            efac = parameter.Uniform(0.1,5)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec

        if "pm_wn_sw" == n:

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
        

    if "PM_WN_SW_extra_fbins" == n:
        pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        wn_json = json.load(open(noisefile))
        keys = list(pmev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
        # Check through the possibilities and add them as appropriate

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED":
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 7)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
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

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 30
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

        efac = parameter.Uniform(0.1,5)
        if "t2equad" in wnmodels or "tnequad" in wnmodels:
            equad = parameter.Uniform(-10,-1)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            s += eq
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-1)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec
            
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

    if "pm_wn_hc" == n or "pm_wn_sw_hc" == n:

        pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        wn_json = json.load(open(noisefile))
        keys = list(pmev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
        # Check through the possibilities and add them as appropriate
        high_comps = 120

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn
                elif i+1 == len(psrmodels):
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                s += rn

            if pm =="DM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_dm = parameter.Uniform(-20, -11)
                    gamma_dm = parameter.Uniform(0, 7)
                    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                    s += dm
                elif i+1 == len(psrmodels):
                    log10_A_dm = parameter.Uniform(-20, -11)
                    gamma_dm = parameter.Uniform(0, 7)
                    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                    s += dm 
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Uniform(0,7)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom
                elif i+1 == len(psrmodels):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Uniform(0,7)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Uniform(0,14)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "CHROMCIDX":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Constant(4)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom
                elif i+1 == len(psrmodels):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Constant(4)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom

            if pm == "BL":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                                selection=low_freq, name='low_band_noise')
                    s += bnl
                elif i+1 == len(psrmodels):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                                selection=low_freq, name='low_band_noise')
                    s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = high_comps
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                                selection=high_freq, name='high_band_noise')
                    s += bnh
                elif i+1 == len(psrmodels):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                                selection=high_freq, name='high_band_noise')
                    s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = high_comps
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh
            

        if  "pm_wn_no_equad" != n:
            efac = parameter.Uniform(0.1,5)
            if "t2equad" in wnmodels or "tnequad" in wnmodels:
                equad = parameter.Uniform(-10,-1)
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            else:
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec
            
        elif "pm_wn_no_equad" == n:
            efac = parameter.Uniform(0.1,5)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec

        if "pm_wn_sw_hc" == n:

            n_earth = parameter.Uniform(0, 20)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            Tspan = psr.toas.max() - psr.toas.min()
            max_cadence = 60
            sw_components = int(Tspan / (max_cadence*86400))

            log10_A_sw = parameter.Uniform(-10, 1)
            gamma_sw = parameter.Uniform(-4, 4)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
            sw_basis = createfourierdesignmatrix_solar_dm(nmodes=high_comps, Tspan=Tspan)

            sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

            s += sw

    if "pm_wn_hc_noeq" == n or "pm_wn_sw_hc_noeq" == n:

        pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_WN_models_NOEQUAD.json"))
        keys = list(pmev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
        # Check through the possibilities and add them as appropriate
        high_comps = 120

        for i, pm in enumerate(psrmodels):

            if pm == "RN" or pm == "RED":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn
                elif i+1 == len(psrmodels):
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                s += rn

            if pm =="DM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_dm = parameter.Uniform(-20, -11)
                    gamma_dm = parameter.Uniform(0, 7)
                    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                    s += dm
                elif i+1 == len(psrmodels):
                    log10_A_dm = parameter.Uniform(-20, -11)
                    gamma_dm = parameter.Uniform(0, 7)
                    dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                    s += dm 
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Uniform(0,7)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom
                elif i+1 == len(psrmodels):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Uniform(0,7)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

            if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Uniform(0,14)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "CHROMCIDX":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Constant(4)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom
                elif i+1 == len(psrmodels):
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 7)
                    chrom_gp_idx = parameter.Constant(4)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = high_comps
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

            if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 14)
                chrom_gp_idx = parameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom

            if pm == "BL":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                                selection=low_freq, name='low_band_noise')
                    s += bnl
                elif i+1 == len(psrmodels):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                                selection=low_freq, name='low_band_noise')
                    s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = high_comps
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                                selection=high_freq, name='high_band_noise')
                    s += bnh
                elif i+1 == len(psrmodels):
                    log10_A_bn = parameter.Uniform(-20, -11)
                    gamma_bn = parameter.Uniform(0, 7)
                    band_components = high_comps
                    bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                    bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                                selection=high_freq, name='high_band_noise')
                    s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = high_comps
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh
            

        if  "pm_wn_no_equad" != n:
            efac = parameter.Uniform(0.1,5)
            if "t2equad" in wnmodels or "tnequad" in wnmodels:
                equad = parameter.Uniform(-10,-1)
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            else:
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec
            
        elif "pm_wn_no_equad" == n:
            efac = parameter.Uniform(0.1,5)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-1)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec

        if "pm_wn_sw_hc_noeq" == n:

            n_earth = parameter.Uniform(0, 20)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            Tspan = psr.toas.max() - psr.toas.min()
            max_cadence = 60
            sw_components = int(Tspan / (max_cadence*86400))

            log10_A_sw = parameter.Uniform(-10, 1)
            gamma_sw = parameter.Uniform(-4, 4)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
            sw_basis = createfourierdesignmatrix_solar_dm(nmodes=high_comps, Tspan=Tspan)

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

    outDir='/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/'+header_dir+'/{0}_{1}'.format(pulsar,results_dir)
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


