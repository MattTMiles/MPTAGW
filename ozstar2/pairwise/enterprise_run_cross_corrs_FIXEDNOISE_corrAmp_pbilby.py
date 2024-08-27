# This script runs enterprise for individual pulsars, or for an entire gravitational wave source

from __future__ import division

import os, glob, json, pickle, sys
import matplotlib.pyplot as plt
import numpy as np, os, dill, shutil, timeit
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

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
#sys.path.insert(0, '/fred/oz002/rshannon/enterprise_warp/')
from enterprise_warp import bilby_warp

sys.path.insert(0, '/home/mmiles/soft/parallel_nested_sampling_pta/')
import schwimmbad_fast
import utils as pb_utils
#from pb_utils import signal_wrapper
from schwimmbad_fast import MPIPoolFast as MPIPool

import bilby
import argparse
import time
import mpi4py
import dynesty
import datetime
import pandas as pd
from pandas import DataFrame
import signal

import types

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")
parser.add_argument("-pair", dest="pair", nargs="+", help="Pulsars to run the cross correlations on", required = False)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc", "pbilby", "hyper"}, required=True)

args = parser.parse_args()

pair = args.pair
sampler = args.sampler
custom_results = str(args.alt_dir)
results_dir = str(args.results)

ptafile = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_FIXED_NOISE/corrAmp/"+pair[0]+"_"+pair[1]+".pkl"

@signal_base.function
def single_bin_orf(pos1, pos2, params):
    '''
    Agnostic binned spatial correlation function. Bin edges are
    placed at edges and across angular separation space. Changing bin
    edges will require manual intervention to create new function.
    :param: params
        inter-pulsar correlation bin amplitudes.
    Author: S. R. Taylor (2020)
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        # bins in angsep space
        bins = np.array([1e-3,180.0]) * np.pi/180.0
        angsep = np.arccos(np.dot(pos1, pos2))
        idx = np.digitize(angsep, bins)
        return params


def UpdateFlags(timefile):
  fin = open(timefile, "r")
  lines = fin.readlines()
  fin.close()
  fout = open(timefile, "w")
  for line in lines:
    w = re.split('\s+', line)
    if w[0]=="FORMAT" or w[0]=="MODE":
      fout.write(line)
      #UpdateFlags(os.path.dirname(timefile)+"/"+w[1])
    elif ('-sys' in w) and not ('-pta' in w):
      fout.write(line[:-1]+' -pta EPTA\n')
    elif not ('-pta' in w):
      fout.write(line[:-1]+' -pta EPTA\n')
    else:
      fout.write(line)
  fout.close()
  return None

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


def prior_transform_function(theta):
  cube = np.zeros(len(theta))
  for i in range(len(pta.params)):
    cube[i] = ( pta.params[i].prior.func_kwargs['pmax'] - pta.params[i].prior.func_kwargs['pmin'])*theta[i] + pta.params[i].prior.func_kwargs['pmin']
  return list(cube)

def log_likelihood_function(cube):
  x0 = np.hstack(cube)
  return pta.get_lnlikelihood(x0)

def log_prior_function(x):
  return pta.get_lnprior(x)

def try_mkdir(outdir, par_nms):
  if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
        np.savetxt(outdir+'/pars.txt', par_nms, fmt='%s') ## works as effectively `outdir` is globally declared
    except:
        FileExistsError
  return None



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
            'single_bin_orf': single_bin_orf(params=parameter.Uniform(-1.0, 1.0, size=1)('gw_single_orf_bin')),
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

def dm_noise(log10_A,gamma,Tspan,components=120,option="powerlaw"):
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


# models = []

# ev_json = json.load(open(modelfile))
# keys = list(ev_json.keys())
# wn_json = json.load(open(noisefile))
# wnkeys = list(wn_json.keys())



# crn = custom_common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_val=None, components=10, orf="single_bin_orf", name='gw_bins')


# for p in psrs:
    
#     tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    
#     s=tm

#     wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if p.name in wn_model ]

#     # Get list of models
#     psrmodels = [ psr_model for psr_model in keys if p.name in psr_model ][0].split("_")[1:]
#     high_comps = 120

#     tmin = p.toas.min()
#     tmax = p.toas.max()
#     Tspan=(tmax-tmin)
    
#     efac = parameter.Constant()
#     ecorr = parameter.Constant()
#     equad = parameter.Constant()

#     if p.name+"_KAT_MKBF_log10_tnequad" in wnkeys:
#         eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
#         s += eq
    
#     ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
#     s += ef

#     if p.name+"_KAT_MKBF_log10_ecorr" in wnkeys:
#         ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,selection=selection)
#         s += ec

#     red_comps = 30
#     for i, pm in enumerate(psrmodels):
        
#         if pm == "RN" or pm == "RED":
#             if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
#                 log10_A_red = parameter.Uniform(-20, -11)
#                 gamma_red = parameter.Uniform(0, 7)
#                 pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
#                 rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
#                 s += rn
#             elif i+1 == len(psrmodels):
#                 log10_A_red = parameter.Uniform(-20, -11)
#                 gamma_red = parameter.Uniform(0, 7)
#                 pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
#                 rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
#                 s += rn

#         if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
#             log10_A_red = parameter.Uniform(-20, -11)
#             gamma_red = parameter.Uniform(0, 14)
#             pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
#             rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
#             s += rn

#         if pm =="DM":
#             if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
#                 log10_A_dm = parameter.Uniform(-20, -11)
#                 gamma_dm = parameter.Uniform(0, 7)
#                 dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
#                 s += dm
#             elif i+1 == len(psrmodels):
#                 log10_A_dm = parameter.Uniform(-20, -11)
#                 gamma_dm = parameter.Uniform(0, 7)
#                 dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
#                 s += dm 
        
#         if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
#             log10_A_dm = parameter.Uniform(-20, -11)
#             gamma_dm = parameter.Uniform(0, 14)
#             dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
#             s += dm

#         if pm == "CHROM":
#             if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
#                 log10_A_chrom_prior = parameter.Uniform(-20, -11)
#                 gamma_chrom_prior = parameter.Uniform(0, 14)
#                 chrom_gp_idx = parameter.Uniform(0,14)
#                 chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#                 idx = chrom_gp_idx
#                 components = high_comps
#                 chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                         idx=idx)
#                 chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#                 s += chrom
#             elif i+1 == len(psrmodels):
#                 log10_A_chrom_prior = parameter.Uniform(-20, -11)
#                 gamma_chrom_prior = parameter.Uniform(0, 14)
#                 chrom_gp_idx = parameter.Uniform(0,14)
#                 chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#                 idx = chrom_gp_idx
#                 components = high_comps
#                 chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                         idx=idx)
#                 chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#                 s += chrom

#         if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
#             log10_A_chrom_prior = parameter.Uniform(-20, -11)
#             gamma_chrom_prior = parameter.Uniform(0, 14)
#             chrom_gp_idx = parameter.Uniform(0,14)
#             chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#             idx = chrom_gp_idx
#             components = 120
#             chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                     idx=idx)
#             chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#             s += chrom

#         if pm == "CHROMCIDX":
#             if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
#                 log10_A_chrom_prior = parameter.Uniform(-20, -11)
#                 gamma_chrom_prior = parameter.Uniform(0, 7)
#                 chrom_gp_idx = parameter.Constant(4)
#                 chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#                 idx = chrom_gp_idx
#                 components = high_comps
#                 chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                         idx=idx)
#                 chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#                 s += chrom
#             elif i+1 == len(psrmodels):
#                 log10_A_chrom_prior = parameter.Uniform(-20, -11)
#                 gamma_chrom_prior = parameter.Uniform(0, 7)
#                 chrom_gp_idx = parameter.Constant(4)
#                 chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#                 idx = chrom_gp_idx
#                 components = high_comps
#                 chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                         idx=idx)
#                 chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#                 s += chrom

#         if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
#             log10_A_chrom_prior = parameter.Uniform(-20, -11)
#             gamma_chrom_prior = parameter.Uniform(0, 14)
#             chrom_gp_idx = parameter.Constant(4)
#             chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
#             idx = chrom_gp_idx
#             components = 120
#             chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
#                                                                     idx=idx)
#             chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
#             s += chrom

#         if pm == "CHROMANNUAL":
#             log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
#             phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
#             idx_chrom1yr = parameter.Uniform(0, 14)
#             wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
#             chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
#             s += chrom1yr

#         if pm == "CHROMBUMP":
#             chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax-tmin, log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
#             s += chrom_gauss_bump

#         if pm == "SW":
#             n_earth = parameter.Uniform(0, 20)
#             deter_sw = solar_wind(n_earth=n_earth)
#             mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

#             Tspan = psr.toas.max() - psr.toas.min()
#             max_cadence = 60
#             sw_components = 120

#             log10_A_sw = parameter.Uniform(-10, 1)
#             gamma_sw = parameter.Uniform(-4, 4)
#             sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
#             sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

#             sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

#             s += sw
        
#         if pm == "SWDET":
#             n_earth = parameter.Uniform(0, 20)
#             deter_sw = solar_wind(n_earth=n_earth)
#             mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

#             sw = mean_sw

#             s += sw
        
#     if "SW" not in psrmodels and "SWDET" not in psrmodels:
#         n_earth = parameter.Constant(4)
#         deter_sw = solar_wind(n_earth=n_earth)
#         mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

#         sw = mean_sw

#         s += sw

#     for n in noise:
#         if "single_bin_cross_corr" == n:
            
#             s += crn

#         elif "single_bin_cross_corr_er" == n:
            
#             if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
#                 log10_A_red = parameter.Uniform(-20, -12)
#                 gamma_red = parameter.Uniform(0, 7)
#                 pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
#                 rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
#                 s += rn

#             s += crn

        
#     models.append(s(p)) 

    
# pta = signal_base.PTA(models)
# pta.set_default_params(params)

pta = dill.load(open(ptafile,"rb"))

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
    
    #pta.set_default_params(params)
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

    try:
        chainfile = outDir+"/chain_1.txt"
        lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
        if int(lenchain) > 50000:
            os.system("touch "+outDir+"/finished")
    except:
        print("Couldn't check chain length")

elif sampler =="hyper":
    try:

        if custom_results is not None and custom_results != "":
            header_dir = custom_results
        else:
            header_dir = "out_ptmcmc"
        pta_dict = dict.fromkeys(np.arange(1))
        pta_dict[0] = pta
        hyper_model = hypermodel.HyperModel(pta_dict)
        x0 = hyper_model.initial_sample()
        ndim = len(x0)

        params = hyper_model.param_names

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
        
        N = int(1e6)
        sampler = hyper_model.setup_sampler(outdir=outDir, resume=True)

        sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

        try:
            chainfile = outDir+"/chain_1.txt"
            lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
            if int(lenchain) > 50000:
                os.system("touch "+outDir+"/finished")
        except:
            print("Couldn't check chain length")

    except ValueError:
        os.system("rm -rf "+outDir)
        
        if custom_results is not None and custom_results != "":
            header_dir = custom_results
        else:
            header_dir = "out_ptmcmc"
        pta_dict = dict.fromkeys(np.arange(1))
        pta_dict[0] = pta
        hyper_model = hypermodel.HyperModel(pta_dict)
        x0 = hyper_model.initial_sample()
        ndim = len(x0)

        params = hyper_model.param_names

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
        
        N = int(1e6)
        sampler = hyper_model.setup_sampler(outdir=outDir, resume=True)

        sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

        try:
            chainfile = outDir+"/chain_1.txt"
            lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
            if int(lenchain) > 50000:
                os.system("touch "+outDir+"/finished")
        except:
            print("Couldn't check chain length")


elif sampler == "ultra":
    priors = bilby_warp.get_bilby_prior_dict(pta)

    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta,parameters)

    label = results_dir
    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out_ultra"

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
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='Ultranest', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='Ultranest', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                # This has nlive none in it because the code that calls this has nlive in it - this will make the reactive nested sampler work
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='Ultranest', resume=True, nlive=None, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}_{1}'.format(pulsar,results_dir), label=label, sampler='Ultranest', resume=True, nlive=None, npool=1, verbose=True, plot=True)

    else:
        if pool is not None:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='Ultranest', resume=True, nlive=nlive, npool=pool, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='Ultranest', resume=True, nlive=1000, npool=pool, verbose=True, plot=True)
        else:
            if nlive is not None:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='Ultranest', resume=True, nlive=nlive, npool=1, verbose=True, plot=True)
            else:
                results = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=header_dir+'/{0}/'.format(results_dir), label=label, sampler='Ultranest', resume=True, nlive=1000, npool=1, verbose=True, plot=True)

    results.plot_corner()


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

elif sampler == "pbilby":
    
    # class really_over_sampled:
    #     def __init__(
    #             self
    #     ):
    #         #self.pool = None
        
    #     #@property
    #     #def signal_catcher(self):

    def safe_file_dump(data, filename, module):
        """Safely dump data to a .pickle file

        Parameters
        ----------
        data:
            data to dump
        filename: str
            The file to dump to
        module: pickle, dill
            The python module to use
        """

        temp_filename = filename + ".temp"
        with open(temp_filename, "wb") as file:
            module.dump(data, file)
        os.rename(temp_filename, filename)

    def write_current_state(sampler, resume_file, sampling_time, rotate=False):
        """Writes a checkpoint file
        Parameters
        ----------
        sampler: dynesty.NestedSampler
            The sampler object itself
        resume_file: str
            The name of the resume/checkpoint file to use
        sampling_time: float
            The total sampling time in seconds
        rotate: bool
            If resume_file already exists, first make a backup file (ending in '.bk').
        """
        print("")
        print("Start checkpoint writing")
        if rotate and os.path.isfile(resume_file):
            resume_file_bk = resume_file + ".bk"
            print("Backing up existing checkpoint file to {}".format(resume_file_bk))
            shutil.copyfile(resume_file, resume_file_bk)
        sampler.kwargs["sampling_time"] = sampling_time
        if dill.pickles(sampler):
            safe_file_dump(sampler, resume_file, dill)
            print("Written checkpoint file {}".format(resume_file))
        else:
            print("Cannot write pickle resume file!")

    def write_current_state_on_kill(signum=None, frame=None):
        """
        Make sure that if a pool of jobs is running only the parent tries to
        checkpoint and exit. Only the parent has a 'pool' attribute.

        For samplers that must hard exit (typically due to non-Python process)
        use :code:`os._exit` that cannot be excepted. Other samplers exiting
        can be caught as a :code:`SystemExit`.
        """
        #if self.pool.rank == 0:
        print("Killed, writing and exiting.")
        #if pool.is_master():
        #rank = mpipool.comm.Get_rank()
        #if mpipool.is_master():
        write_current_state(sampler, resume_file, sampling_time, rotate_checkpoints)
        #pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
        pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
        pb_utils.plot_current_state(sampler, outdir, label, sampling_keys)
        os._exit(130)


    label = results_dir
    if custom_results is not None and custom_results != "":
        header_dir = custom_results
    else:
        header_dir = "out_pbilby"

    outDir=header_dir+'/{0}'.format(results_dir)
    print(outDir)
    
    mpi4py.rc.threads=False
    mpi4py.rc.recv_mprobe=False

    start_time = time.perf_counter()
    path0 = os.getcwd()

    par_nms = pta.param_names
    Npar = len(par_nms)
    ndim = Npar
    outdir = outDir

    ## you might want to change any of the following depending what you are running
    #nlive=nlive
    nlive=400
    tol=0.1
    dynesty_sample='rwalk'
    dynesty_bound='multi'
    walks=100
    maxmcmc=5000
    nact=10
    facc=0.5
    min_eff=10.
    vol_dec=0.5
    vol_check=8.
    enlarge=1.5
    is_nestcheck=False
    n_check_point=5
    do_not_save_bounds_in_resume=False
    check_point_deltaT=600
    n_effective=np.inf
    max_its=1e10
    max_run_time=1.0e10
    rotate_checkpoints=False
    rotate_checkpoints = rotate_checkpoints
    rstate = np.random

    fast_mpi=False
    mpi_timing=False
    mpi_timing_interval=0
    nestcheck_flag=False

    try_mkdir(outdir, par_nms)

    try:
        with open(outDir+"/run_summary.txt","w") as f:
            print(pta.summary(), file=f)
    except:
        pass

    # getting the sampling keys
    sampling_keys = pta.param_names

    t0 = datetime.datetime.now()
    sampling_time=0

    # signal.signal(signal.SIGTERM, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))
    # signal.signal(signal.SIGINT, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))
    # signal.signal(signal.SIGALRM, pb_utils.write_current_state_on_kill(pool, pbsampler, resume_file, sampling_time, rotate_checkpoints))

    with MPIPool(parallel_comms=fast_mpi,
                time_mpi=mpi_timing,
                timing_interval=mpi_timing_interval,) as mpipool:
        #self.pool = pool
        
        if mpipool.is_master():

            signal.signal(signal.SIGTERM, write_current_state_on_kill)
            signal.signal(signal.SIGINT, write_current_state_on_kill)
            signal.signal(signal.SIGALRM, write_current_state_on_kill)

            POOL_SIZE = mpipool.size
            np.random.seed(1234)
            filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
            resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)
            resume_file = resume_file
            samples_file = "{}/{}_samples.dat".format(outdir, label)
            nestcheck_flag = is_nestcheck
            init_sampler_kwargs = dict(
                nlive=nlive,
                sample=dynesty_sample,
                bound=dynesty_bound,
                walks=walks,
                #rstate=rstate,
                #maxmcmc=maxmcmc,
                #nact=nact,
                facc=facc,
                first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
                #vol_dec=vol_dec,
                #vol_check=vol_check,
                enlarge=enlarge,
                #save_bounds=False,
            )


            pbsampler, sampling_time = pb_utils.read_saved_state(resume_file)
            #sampler = False
            if pbsampler is False:
                live_points = pb_utils.get_initial_points_from_prior(
                    ndim,
                    nlive,
                    prior_transform_function,
                    log_prior_function,
                    log_likelihood_function,
                    mpipool,
                )

                pbsampler = dynesty.NestedSampler(
                    log_likelihood_function,
                    prior_transform_function,
                    ndim,
                    pool=mpipool,
                    queue_size=POOL_SIZE,
                    #print_func=dynesty.utils.print_fn_fallback,
                    live_points=live_points,
                    #live_points=None,
                    use_pool=dict(
                        update_bound=True,
                        propose_point=True,
                        prior_transform=True,
                        loglikelihood=True,
                    ),
                    **init_sampler_kwargs,
                )
            else:
                pbsampler.pool = mpipool
                pbsampler.M = mpipool.map
                pbsampler.queue_size = POOL_SIZE
                #pbsampler.rstate = np.random
                #pbsampler.nlive=nlive
            sampler_kwargs = dict(
                n_effective=n_effective,
                dlogz=tol,
                save_bounds=not do_not_save_bounds_in_resume,

            )
            
            if dynesty_sample == "rwalk":
                bilby.core.utils.logger.info(
                    "Using the bilby-implemented rwalk sample method with ACT estimated walks. "
                    f"An average of {2 * nact} steps will be accepted up to chain length "
                    f"{maxmcmc}."
                )
                from bilby.core.sampler.dynesty_utils import AcceptanceTrackingRWalk
                from bilby.core.sampler.dynesty import DynestySetupError
                if walks > maxmcmc:
                    raise DynestySetupError("You have maxmcmc < walks (minimum mcmc)")
                if nact < 1:
                    raise DynestySetupError("Unable to run with nact < 1")
                AcceptanceTrackingRWalk.old_act = None
                dynesty.nestedsamplers._SAMPLING["rwalk"] = AcceptanceTrackingRWalk()
            
            run_time = 0
            print(pbsampler)
            #spbsampler.kwargs["live_points"] = live_points
            pbsampler.kwargs["nlive"] = nlive
            sampler = pbsampler
            
            for it, res in enumerate(pbsampler.sample(**sampler_kwargs)):
                (
                    worst,
                    ustar,
                    vstar,
                    loglstar,
                    logvol,
                    logwt,
                    logz,
                    logzvar,
                    h,
                    nc,
                    worst_it,
                    boundidx,
                    bounditer,
                    eff,
                    delta_logz,
                    blob
                ) = res
                i = it - 1
                dynesty.utils.print_fn_fallback(
                    res, i, pbsampler.ncall, dlogz=tol
                )
                if (
                    it == 0 or it % n_check_point != 0
                ) and it != max_its:
                    continue
                iteration_time = (datetime.datetime.now() - t0).total_seconds()
                t0 = datetime.datetime.now()
                sampling_time += iteration_time
                sampling_time = sampling_time
                run_time += iteration_time
                if os.path.isfile(resume_file):
                    last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
                else:
                    last_checkpoint_s = np.inf
                if (
                    last_checkpoint_s > check_point_deltaT
                    or it == max_its
                    or run_time > max_run_time
                    or delta_logz < tol
                ):
                    pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
                    pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
                    pb_utils.plot_current_state(sampler, outdir, label, sampling_keys)
                    if it == max_its:
                        print("Max iterations %d reached; stopping sampling."%max_its)
                        #sys.exit(0)
                        break
                    if run_time > max_run_time:
                        print("Max run time %e reached; stopping sampling."%max_run_time)
                        #sys.exit(0)
                        break
                    if delta_logz < tol:
                        print("Tolerance %e reached; stopping sampling."%tol)
                        #sys.exit(0)
                        break
            # Adding the final set of live points.
            for it_final, res in enumerate(pbsampler.add_live_points()):
                pass
            # Create a final checkpoint and set of plots
            pb_utils.write_current_state(pbsampler, resume_file, sampling_time, rotate_checkpoints)
            pb_utils.write_sample_dump(pbsampler, samples_file, sampling_keys)
            pb_utils.plot_current_state(pbsampler, outdir, label, sampling_keys)

            sampling_time += (datetime.datetime.now() - t0).total_seconds()
            sampling_time = sampling_time
            out = pbsampler.results
            if nestcheck_flag is True:
                ns_run = nestcheck.data_processing.process_dynesty_run(out)
                nestcheck_path = os.path.join(outdir, "Nestcheck")
                try_mkdir(nestcheck_path)
                nestcheck_result = "{}/{}_nestcheck.pickle".format(nestcheck_path, label)
                with open(nestcheck_result, "wb") as file_nest:
                    pickle.dump(ns_run, file_nest)
            weights = np.exp(out["logwt"] - out["logz"][-1])
            nested_samples = DataFrame(out.samples, columns=sampling_keys)
            nested_samples["weights"] = weights
            nested_samples["log_likelihood"] = out.logl

            samples = dynesty.utils.resample_equal(out.samples, weights)

            result_log_likelihood_evaluations = pb_utils.reorder_loglikelihoods(unsorted_loglikelihoods=out.logl,unsorted_samples=out.samples,sorted_samples=samples,)

            log_evidence = out.logz[-1]
            log_evidence_err = out.logzerr[-1]
            final_sampling_time = sampling_time

            posterior = pd.DataFrame(samples, columns=sampling_keys)
            nsamples = len(posterior)

            print("Sampling time = {}s".format(datetime.timedelta(seconds=sampling_time)))
            print('log evidence is %f'%log_evidence)
            print('error in log evidence is %f'%log_evidence_err)
            pos = posterior.to_json(orient="columns")

            with open(outdir+"/"+label+"_final_res.json", "w") as final:
                json.dump(pos, final)
            np.savetxt(outdir+"/"+"evidence.txt", np.c_[log_evidence, log_evidence_err, float(datetime.timedelta(seconds=sampling_time).total_seconds())], header="logZ \t logZ_err \t sampling_time_s")
            

    end_time = time.perf_counter()
    time_taken = end_time  - start_time
    print("This took a total time of %f seconds."%time_taken)

# really_over_sampled = really_over_sampled()
# really_over_sampled.over_sampled()



