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
from enterprise_extensions import models, model_utils, hypermodel, blocks, model_orfs
from enterprise_extensions import timing
from enterprise_extensions import deterministic

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
parser.add_argument("-partim", dest="partim", help="Par and tim files for the pulsars to be used",required = False)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-noisefile", type = str, dest="noisefile", help="The white noise values used for the noise analysis.", required = False)
parser.add_argument("-noise_search", type = str.lower, nargs="+",dest="noise_search",help="Which GW search to do. If multiple are chosen it will combine them i.e. HD + monopole", \
    choices={"pl_nocorr_freegam","pl_nocorr_fixgam","bpl_nocorr_freegam","freespec_nocorr","pl_orf_bins","pl_orf_spline","pl_hd_fixgam", "pl_hd_freegam","pl_hdnoauto_fixgam",\
        "freespec_hd","pl_dp","freespec_dp","pl_mono","freespec_monopole", "extra_red", "extra_dm", "pl_hdnoauto_freegam", "dm_red_misspec", "pl_hd_freegam_linexpsquare", "pl_hd_freegam_linexp",\
            "pl_nocorr_freegam_linexpsquare", "pl_hd_freegam_linexpsquare", "extra_red_fixed"})
parser.add_argument("-smbhb", type = str.lower, nargs="+", dest="cw_search", help="Option for a CW search. There are options to choose from.", choices={"cw_circ","cw_ecc"})
parser.add_argument("-p_irn", type = str.lower, nargs="+",dest="p_irn",help="How to search the pulsar intrinsic noise. Only choose one.", \
    choices={"native", "dm_red_misspec", "native_fixed", "nothing"})
parser.add_argument("-rn_vals", dest="rn_vals", type = float, nargs="+",help="Custom RN values, first is amp, second is gamma.")
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc"}, required=True)
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-psrlist", dest="psrlist", nargs="+", help=r"List of pulsars to use", required = False)
parser.add_argument("-ptaname", dest="ptaname", help=r"Name of pta object", required = True)
args = parser.parse_args()

results_dir = str(args.results)
noisefile = args.noisefile
noise = args.noise_search
cw_search = args.cw_search
p_irn = args.p_irn
sampler = args.sampler
partim = args.partim
custom_results = str(args.alt_dir)
psr_list = args.psrlist
ptaname= str(args.ptaname)
custom_rn = args.rn_vals

if psr_list is not None and psr_list != "":
    if type(psr_list[0]) != list:
        psrlist=[ x.strip("\n") for x in open(str(psr_list[0])).readlines() ]
    else:
        psrlist = list(psr_list)

## Static data directory at the moment 
datadir = partim
if not os.path.isdir(datadir):
    datadir = '../partim'
print(datadir)

parfiles = sorted(glob.glob(datadir + '/*par'))
timfiles = sorted(glob.glob(datadir + '/*tim'))

# filter
if psr_list is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]


ephemeris = 'DE440' # Static as not using bayesephem


psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)



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


def mk_band_split(freqs, backend_flags):
    """ Selection for splitting the band in 3 """
    d =  dict(zip(['low', 'high', 'all'],
                  [(freqs <= 1100), (freqs > 1100), (freqs >= 0)]))
    delkeys = []
    for key in d.keys():
        if np.sum(d[key]) == 0:  # all False
            delkeys.append(key)
    for key in delkeys:
        del d[key]
    return d

def mk_band_split_2(freqs, backend_flags):
    """ Selection for splitting the band in 3 """
    d =  dict(zip(['low', 'high'],
                  [(freqs <= 1100), (freqs > 1100)]))
    delkeys = []
    for key in d.keys():
        if np.sum(d[key]) == 0:  # all False
            delkeys.append(key)
    for key in delkeys:
        del d[key]
    return d

mk_ecorr_selection = selections.Selection(mk_band_split)


#ecorr_selection = selections.Selection(pta_split)
ecorr_selection = selections.Selection(selections.by_backend)

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
def chrom_gaussian_bump(toas, freqs, log10_Amp=-2.5, sign_param=1.0,
                    t0=53890, sigma=81, idx=2):
    """
    Chromatic time-domain Gaussian delay term in TOAs.
    Example: J1603-7202 in Lentati et al, MNRAS 458, 2016.
    """
    #t0 *= const.day
    #sigma *= const.day
    wf = 10**log10_Amp * np.exp(-(toas - t0)**2/2/sigma**2)
    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

def dm_gaussian_bump(tmin, tmax, idx=2, sigma_min=600000, sigma_max=1000,
    log10_A_low=-10, log10_A_high=-1, name='dm_bump'):
    """
    Returns chromatic Gaussian bump (i.e. TOA advance):
    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sigma_min, sigma_max:
        standard deviation of a Gaussian in MJD
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal
    :return dm_bump:
        chromatic Gaussian bump waveform.
    """
    sign_param = parameter.Uniform(-1,1)
    t0_dm_bump = parameter.Uniform(tmin,tmax)
    sigma_dm_bump = parameter.Uniform(sigma_min,tmax-tmin)
    log10_Amp_dm_bump = parameter.Uniform(log10_A_low, log10_A_high)
    if idx == 'vary':
        idx = parameter.Uniform(0, 14)
    wf = chrom_gaussian_bump(log10_Amp=log10_Amp_dm_bump,
                         t0=t0_dm_bump, sigma=sigma_dm_bump,
                         sign_param=sign_param, idx=idx)
    dm_bump = deterministic_signals.Deterministic(wf, name=name)

    return dm_bump

def dm_gaussian_bump_const(tmin, tmax, idx=2, sigma_min=600000, sigma_max=1000,
    log10_A_low=-10, log10_A_high=-1, name='dm_bump'):
    """
    Returns chromatic Gaussian bump (i.e. TOA advance):
    :param tmin, tmax:
        search window for exponential cusp time.
    :param idx:
        index of radio frequency dependence (i.e. DM is 2). If this is set
        to 'vary' then the index will vary from 1 - 6
    :param sigma_min, sigma_max:
        standard deviation of a Gaussian in MJD
    :param sign:
        [boolean] allow for positive or negative exponential features.
    :param name: Name of signal
    :return dm_bump:
        chromatic Gaussian bump waveform.
    """
    sign_param = parameter.Constant()
    t0_dm_bump = parameter.Constant()
    sigma_dm_bump = parameter.Constant()
    log10_Amp_dm_bump = parameter.Constant()
    if idx == 'vary':
        idx = parameter.Constant()
    wf = chrom_gaussian_bump(log10_Amp=log10_Amp_dm_bump,
                         t0=t0_dm_bump, sigma=sigma_dm_bump,
                         sign_param=sign_param, idx=idx)
    dm_bump = deterministic_signals.Deterministic(wf, name=name)

    return dm_bump


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

def LinearExpPrior_Squared(value, pmin, pmax):
    """Prior function for LinearExp parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    # works with vectors if pmin and pmax are either scalars,
    # or len(value) vectors
    return ((pmin <= value) & (value <= pmax)) * np.log(100) * ((10**value)**2) / ((100**pmax) - (100**pmin))

def LinearExpSampler_Squared(pmin, pmax, size=None):
    """Sampling function for LinearExp parameters."""

    if np.any(pmin >= pmax):
        raise ValueError("LinearExp Parameter requires pmin < pmax.")

    # works with vectors if pmin and pmax are either scalars
    # or vectors, in which case one must have len(pmin) = len(pmax) = size
    return np.log10(np.sqrt(np.random.uniform(100**(pmin), 100**(pmax), size)))

@parameter.function
def custom_common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, components=30, combine=True,
                           log10_A_val=None, gamma_val=None, delta_val=None,
                           logmin=None, logmax=None,
                           orf=None, orf_ifreq=0, leg_lmax=5,
                           name='gw', coefficients=False,
                           pshift=False, pseed=None):
    """
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum', 'broken_powerlaw']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar.
    :param log10_A_val:
        Value of log10_A parameter for fixed amplitude analyses.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param delta_val:
        Value of spectral index for high frequencies in broken power-law
        and turnover models. By default spectral index is varied in range [0,7].\
    :param logmin:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param logmax:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function [default=5]
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process

    """

    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(-1.0, 1.0, size=7)('gw_orf_bin')),
            'zero_diag_hd': model_orfs.zero_diag_hd(),
            'zero_diag_bin_orf': model_orfs.zero_diag_bin_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_zero_diag')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre')),
            'zero_diag_legendre_orf': model_orfs.zero_diag_legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre_zero_diag'))}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover', 'turnover_knee', 'broken_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        if logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'LinearExp_Squared':
                log10_Agw = parameter.UserParameter(prior=parameter.Function(LinearExpPrior_Squared, pmin=-18, pmax=-11), sampler=LinearExpSampler_Squared)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            else:
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'LinearExp_Squared':
                log10_Agw = parameter.UserParameter(prior=parameter.Function(LinearExpPrior_Squared, pmin=-18, pmax=-11), sampler=LinearExpSampler_Squared)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(-18, -14)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                    lfb=lfb_gw, lfk=lfk_gw,
                                    kappa=kappa_gw, delta=delta_gw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)

        # checking if priors specified, otherwise give default values
        if logmin is None:
            logmin = -9
        if logmax is None:
            logmax = -4

        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(logmin, logmax,
                                               size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients, combine=combine,
                                        components=components, Tspan=Tspan,
                                        name=name, pshift=pshift, pseed=pseed)
    elif orf in orfs.keys():
        if orf == 'crn':
            crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients, combine=combine,
                                            components=components, Tspan=Tspan,
                                            name=name, pshift=pshift, pseed=pseed)
        else:
            crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf],
                                                  components=components, combine=combine,
                                                  Tspan=Tspan,
                                                  name=name, pshift=pshift,
                                                  pseed=pseed)
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.FourierBasisCommonGP(cpl, orf,
                                              components=components, combine=combine,
                                              Tspan=Tspan,
                                              name=name, pshift=pshift,
                                              pseed=pseed)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn






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


# Choose which GWB
components=30
spline_comps = 3
if noise is not None:
    for i,n in enumerate(noise):
        #power law, free spectral index, no correlations
        if "pl_nocorr_freegam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=30, orf=None, name='gw')
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=30, orf=None, name='gw')

        #power law, fixed 13/3 spectral index, no correlations
        if "pl_nocorr_fixgam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.3333,
                                components=30, orf=None, name='gw')
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.3333,
                                components=30, orf=None, name='gw')

        if "pl_nocorr_freegam_linexpsquare" == n:
            if i==0:
                crn = custom_common_red_noise_block(psd='powerlaw', prior='LinearExp_Squared',
                                    components=30, orf=None, name='gwb')
            else:
                crn += custom_common_red_noise_block(psd='powerlaw', prior='LinearExp_Squared',
                                    components=30, orf=None, name='gwb')

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
                crn = common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 30,
                                orf = None, name = 'gw')
            else:
                crn += common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 30,
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
                                    components=spline_comps, orf='spline_orf', name='gw_bins')
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=4.33,
                                components=spline_comps, orf='spline_orf', name='gw_bins')

        # Powerlaw Hellings-Downs
        if "pl_hd_fixgam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                    components=30, orf='hd', name='gwb', gamma_val = 4.333)
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                    components=30, orf='hd', name='gwb', gamma_val = 4.333)

        # Powerlaw Hellings-Downs free spectral index
        if "pl_hd_freegam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                    components=30, orf='hd', name='gwb')
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                    components=30, orf='hd', name='gwb')
                
        if "pl_hd_freegam_linexpsquare" == n:
            if i==0:
                crn = custom_common_red_noise_block(psd='powerlaw', prior='LinearExp_Squared',
                                    components=30, orf='hd', name='gwb')
            else:
                crn += custom_common_red_noise_block(psd='powerlaw', prior='LinearExp_Squared',
                                    components=30, orf='hd', name='gwb')
                
        # if "pl_hd_freegam_linexp" == n:
        #     if i==0:
        #         crn = common_red_noise_block(psd='powerlaw', prior='uniform',
        #                             components=30, orf='hd', name='gwb')
        #     else:
        #         crn += common_red_noise_block(psd='powerlaw', prior='uniform',
        #                             components=30, orf='hd', name='gwb')

        if "pl_hd_freegam_linexp" == n:
            #log10_A_gw = bilby_warp.linearExponential(-20,-11, 'log10_A_gw')
            log10_A_gw = parameter.LinearExp(-20,-11)('log10_A_gw')
            gamma_gw = parameter.Uniform(0, 7)('gamma_gw')

            gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
            gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
            
            if i ==0:
                crn = gw
            else:
                crn += gw
        # Powerlaw fixed-gamma Hellings-Downs cross-correlations only
        if "pl_hdnoauto_fixgam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='zero_diag_hd', name='gwb_noauto', gamma_val = 4.333)
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='zero_diag_hd', name='gwb_noauto', gamma_val = 4.333)
                
        if "pl_hdnoauto_freegam" == n:
            if i==0:
                crn = common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='zero_diag_hd', name='gwb_noauto')
            else:
                crn += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                components=components, orf='zero_diag_hd', name='gwb_noauto')

        # Free-spectrum Hellings-Downs
        if "freespec_hd" == n:
            if i==0:
                crn = common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 30,
                                orf = 'hd', name = 'gwb')
            else:
                crn += common_red_noise_block(psd = 'spectrum', prior = "log-uniform", components = 30,
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
                                components=20, orf='dipole', name='gw_dipole')
            else:
                crn += common_red_noise_block(psd='spectrum', prior='log-uniform',
                                components=20, orf='dipole', name='gw_dipole')

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

if cw_search is not None:
    for cw_s in cw_search:
        if cw_s == "cw_circ":

            cw = deterministic.cw_block_circ(amp_prior="log-uniform", dist_prior=None, skyloc=None, log10_fgw=None, psrTerm=False, name="cw")

        elif cw_s == "cw_ecc":

            cw = deterministic.cw_block_ecc(amp_prior="log-uniform", skyloc=None, log10_F=None, ecc=None, psrTerm=False, name="cw")



models = []
ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json"))
keys = list(ev_json.keys())
wn_json = json.load(open(noisefile))
wnkeys = list(wn_json.keys())

#if not "dm_red_misspec" in noise:
for irn_select in p_irn:
    if irn_select == "native":
        for p in psrs:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            s=tm

            wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

            # Get list of models
            psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
            high_comps = 120

            tmin = p.toas.min()
            tmax = p.toas.max()
            Tspan=(tmax-tmin)
            
            efac = parameter.Constant()
            ecorr = parameter.Constant()
            equad = parameter.Constant()

            if p.name+"_KAT_MKBF_log10_tnequad" in wnkeys:
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if p.name+"_KAT_MKBF_log10_ecorr" in wnkeys:
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,selection=selection)
                s += ec

            # if "t2equad" in wnmodels or "tnequad" in wnmodels:
            #     #equad = parameter.Uniform(-10,-1)
            #     ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            #     s += ef
            #     eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            #     s += eq
            # else:
            #     ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            #     s += ef

            # if "ecorr_all" in wnmodels:
            #     #s += gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)
            #     ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=mk_ecorr_selection)
            #     s += ec

            red_comps = 30
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

                if pm == "CHROM":
                    log10_A_chrom_prior = parameter.Uniform(-20, -11)
                    gamma_chrom_prior = parameter.Uniform(0, 14)
                    chrom_gp_idx = parameter.Uniform(0,14)
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
                    components = 120
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
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
                    components = 120
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

                if pm == "CHROMANNUAL":
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -5)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr

                if pm == "CHROMBUMP":
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax-tmin, log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump

                if pm == "SW":
                    n_earth = parameter.Uniform(0, 20)
                    deter_sw = solar_wind(n_earth=n_earth)
                    mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

                    Tspan = psr.toas.max() - psr.toas.min()
                    max_cadence = 60
                    sw_components = 120

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
            
            if noise is not None:
                for i, n in enumerate(noise):
                    if "extra_red" == n:
                        if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                            log10_A_red = parameter.Uniform(-20, -12)
                            gamma_red = parameter.Uniform(0, 7)
                            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                            rn = gp_signals.FourierBasisGP(spectrum=pl, components=red_comps, Tspan=Tspan)
                            s += rn
            
            if noise is not None:
                for i, n in enumerate(noise):
                    if "extra_red_fixed" == n:
                        if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                            log10_A_red = parameter.Constant(-18)
                            gamma_red = parameter.Constant(4.333)
                            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                            rn = gp_signals.FourierBasisGP(spectrum=pl, components=red_comps, Tspan=Tspan)
                            s += rn

            if noise is not None:
                for i, n in enumerate(noise):
                    if "extra_dm" == n:
                        if len([ pn for pn in psrmodels if "DM" in pn ]) == 0:
                            log10_A_dm = parameter.Uniform(-20, -11)
                            gamma_dm = parameter.Uniform(0, 7)
                            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                            s += dm

            if custom_rn is not None:
                if len([pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Constant(custom_rn[0])
                    gamma_red = parameter.Constant(custom_rn[1])
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=red_comps, Tspan=Tspan)
                    s += rn

                        

            if noise is not None:
                s +=crn

            if cw_search is not None:
                s+=cw
                
            models.append(s(p))

    elif irn_select == "native_fixed":
        for p in psrs:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            s=tm

            wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

            # Get list of models
            psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
            high_comps = 120

            tmin = p.toas.min()
            tmax = p.toas.max()
            Tspan=(tmax-tmin)
            
            efac = parameter.Constant()
            ecorr = parameter.Constant()
            equad = parameter.Constant()

            if p.name+"_KAT_MKBF_log10_tnequad" in wnkeys:
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if p.name+"_KAT_MKBF_log10_ecorr" in wnkeys:
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,selection=selection)
                s += ec


            red_comps = 30
            for i, pm in enumerate(psrmodels):
                
                if pm == "RN" or pm == "RED":
                    if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                        log10_A_red = parameter.Constant()
                        gamma_red = parameter.Constant()
                        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                        rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                        s += rn
                    elif i+1 == len(psrmodels):
                        log10_A_red = parameter.Constant()
                        gamma_red = parameter.Constant()
                        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                        rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                        s += rn

                if pm =="DM":
                    if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                        log10_A_dm = parameter.Constant()
                        gamma_dm = parameter.Constant()
                        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                        s += dm
                    elif i+1 == len(psrmodels):
                        log10_A_dm = parameter.Constant()
                        gamma_dm = parameter.Constant()
                        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                        s += dm 

                if pm == "CHROM":
                    if p.name in ["J0613-0200", "J1125-6014", "J1643-1224", "J1708-3506"]:
                        log10_A_chrom_prior = parameter.Constant()
                        gamma_chrom_prior = parameter.Constant()
                        chrom_gp_idx = parameter.Constant()
                        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                        idx = chrom_gp_idx
                        components = high_comps
                        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                                idx=idx)
                        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                        s += chrom
                    else:
                        if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                            log10_A_chrom_prior = parameter.Constant()
                            gamma_chrom_prior = parameter.Constant()
                            chrom_gp_idx = parameter.Constant()
                            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                            idx = chrom_gp_idx
                            components = high_comps
                            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                                    idx=idx)
                            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                            s += chrom
                        elif i+1 == len(psrmodels):
                            log10_A_chrom_prior = parameter.Constant()
                            gamma_chrom_prior = parameter.Constant()
                            chrom_gp_idx = parameter.Constant()
                            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                            idx = chrom_gp_idx
                            components = high_comps
                            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                                    idx=idx)
                            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                            s += chrom

                if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                    log10_A_chrom_prior = parameter.Constant()
                    gamma_chrom_prior = parameter.Constant()
                    chrom_gp_idx = parameter.Constant()
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = 120
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

                if pm == "CHROMCIDX":
                    if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                        log10_A_chrom_prior = parameter.Constant()
                        gamma_chrom_prior = parameter.Constant()
                        chrom_gp_idx = parameter.Constant(4)
                        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                        idx = chrom_gp_idx
                        components = high_comps
                        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                                idx=idx)
                        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                        s += chrom
                    elif i+1 == len(psrmodels):
                        log10_A_chrom_prior = parameter.Constant()
                        gamma_chrom_prior = parameter.Constant()
                        chrom_gp_idx = parameter.Constant(4)
                        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                        idx = chrom_gp_idx
                        components = high_comps
                        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                                idx=idx)
                        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                        s += chrom

                if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                    log10_A_chrom_prior = parameter.Constant()
                    gamma_chrom_prior = parameter.Constant()
                    chrom_gp_idx = parameter.Constant(4)
                    chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                    idx = chrom_gp_idx
                    components = 120
                    chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                            idx=idx)
                    chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                    s += chrom

                if pm == "CHROMANNUAL":
                    log10_Amp_chrom1yr = parameter.Constant()
                    phase_chrom1yr = parameter.Constant()
                    idx_chrom1yr = parameter.Constant()
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr

                if pm == "CHROMBUMP":
                    chrom_gauss_bump = dm_gaussian_bump_const(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax-tmin, log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump

                if pm == "SW":
                    n_earth = parameter.Constant()
                    deter_sw = solar_wind(n_earth=n_earth)
                    mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

                    Tspan = psr.toas.max() - psr.toas.min()
                    max_cadence = 60
                    sw_components = 120

                    log10_A_sw = parameter.Constant()
                    gamma_sw = parameter.Constant()
                    sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
                    sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

                    sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

                    s += sw
                
                if pm == "SWDET":
                    n_earth = parameter.Constant()
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
            
            if noise is not None:
                for i, n in enumerate(noise):
                    if "extra_red" == n:
                        if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                            log10_A_red = parameter.Constant()
                            gamma_red = parameter.Constant()
                            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                            rn = gp_signals.FourierBasisGP(spectrum=pl, components=red_comps, Tspan=Tspan)
                            s += rn

            if noise is not None:
                for i, n in enumerate(noise):
                    if "extra_dm" == n:
                        if len([ pn for pn in psrmodels if "DM" in pn ]) == 0:
                            log10_A_dm = parameter.Constant()
                            gamma_dm = parameter.Constant()
                            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                            s += dm


                        

            if noise is not None:
                s +=crn

            if cw_search is not None:
                s+=cw
                
            models.append(s(p))

    elif irn_select == "nothing":
        for p in psrs:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            s=tm

            wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

            # Get list of models
            psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
            high_comps = 120

            tmin = p.toas.min()
            tmax = p.toas.max()
            Tspan=(tmax-tmin)
            
            efac = parameter.Constant(1)

            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if noise is not None:
                s +=crn

            if cw_search is not None:
                s+=cw

            models.append(s(p))

    elif irn_select == "dm_red_misspec":
        for p in psrs:
            tm = gp_signals.MarginalizingTimingModel(use_svd=True)
            s=tm

            wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

            # Get list of models
            psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
            high_comps = 120

            tmin = p.toas.min()
            tmax = p.toas.max()
            Tspan=(tmax-tmin)
            
            efac = parameter.Constant()
            ecorr = parameter.Constant()
            equad = parameter.Constant()

            if p.name+"_KAT_MKBF_log10_tnequad" in wnkeys:
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

            if p.name+"_KAT_MKBF_log10_ecorr" in wnkeys:
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,selection=selection)
                s += ec


            log10_A_red = parameter.Uniform(-20, -11)
            gamma_red = parameter.Uniform(0, 7)
            pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
            s += rn

            log10_A_dm = parameter.Uniform(-20, -11)
            gamma_dm = parameter.Uniform(0, 7)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
            s += dm 


            if noise is not None:
                s +=crn

            if cw_search is not None:
                s+=cw
                
            models.append(s(p))

# else:
#     for p in psrs:
#         tm = gp_signals.MarginalizingTimingModel(use_svd=True)
#         s=tm

#         wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

#         # Get list of models
#         psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
#         high_comps = 120


#         efac = parameter.Constant()
#         ecorr = parameter.Constant()
#         equad = parameter.Constant()
#         if "t2equad" in wnmodels or "tnequad" in wnmodels:
#             #equad = parameter.Uniform(-10,-1)
#             ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
#             s += ef
#             eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
#             s += eq
#         else:
#             ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
#             s += ef

#         if "ecorr" in wnmodels:
#             #ecorr = parameter.Uniform(-10,-1)
#             ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
#             s += ec
        
#         log10_A_red = parameter.Uniform(-20, -12)
#         gamma_red = parameter.Uniform(0, 7)
#         pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
#         rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
#         s += rn

#         log10_A_dm = parameter.Uniform(-20, -12)
#         gamma_dm = parameter.Uniform(0, 7)
#         dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
#         s += dm

#         s +=crn
#         models.append(s(p))


pta = signal_base.PTA(models)
pta.set_default_params(params)



if custom_results is not None and custom_results != "":
    header_dir = custom_results
else:
    header_dir = "out_"+sampler

outDir = header_dir

with open(outDir+"/{0}_run_summary.txt".format(ptaname),"w") as f:
            print(pta.summary(), file=f)

dill.dump(pta, file = open(outDir+"/"+ptaname+".pkl", "wb"))





