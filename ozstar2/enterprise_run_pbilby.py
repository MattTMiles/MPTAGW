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
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

from enterprise.signals import gp_priors

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

from astropy import coordinates as coord
from astropy import units as units
from astropy import constants as aconst
import scipy.interpolate as interpolate


## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Single pulsar enterprise noise run.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
parser.add_argument("-partim", dest="partim", help="Par and tim files for the pulsars to be used",required = False)
parser.add_argument("-results", dest="results", help=r"Name of directory created in the default out directory. Will be of the form {pulsar}_{results}", required = True)
parser.add_argument("-noisefile", type = str, dest="noisefile", help="The noisefile used for the noise analysis.", required = False)
parser.add_argument("-modelfile", type = str, dest="modelfile", help="The model used for the analysis.", required = False)
parser.add_argument("-noise_search", type = str.lower, nargs="+",dest="noise_search", help="The noise parameters to search over. Timing model is default. Include as '-noise_search noise1 noise2 noise3' etc. The _c variations of the noise redirects the noise to the constant noisefile values", \
    choices={"efac", "equad", "t2_equad", "ecorr", "red", "efac_c", "equad_c", "ecorr_c", "ecorr_split_c", "ecorr_check", "red_c", "dm", "chrom", "chrom_c","chrom_cidx","high_comp_chrom", "dm_c", "gw", "gw_const_gamma", "gw_const_gamma_low","gw_const_gamma_wide", "lin_exp_gw_const_gamma_wide", "gw_c", "gw_c_low", "dm_wide", "dm_wider", "red_wide", "chrom_wide", "chrom_cidx_wide", "efac_wide",\
        "band_low","band_low_c","band_high","band_high_c", "band_high_wide", "spgw", "spgwc", "spgwc_wide", "spgwc_18", "pm_wn", "pm_wn_no_equad", "pm_wn_sw","pm_wn_altpar", "pm_wn_no_equad_altpar", "wn_sw", "wn_tester", "chrom_annual", "chrom_annual_c", "sw", "swdet", "free_spgw", "free_spgwc", "hfred", "pm_wn_hc", "pm_wn_sw_hc", "pm_wn_sw_hc_noeq","pm_wn_hc_noeq",\
            "hc_chrom_cidx", "hc_red", "hc_dm", "hc_chrom", "mpta_pm", "mpta_pm_gauss_ecorr", "extra_red", "smbhb", "smbhb_const", "ecorr_gauss", "ecorr_gauss_c", "smbhb_wn", "smbhb_wn_all", "smbhb_const_wn", "extra_chrom_annual", "dm_gauss_bump", "chrom_gauss_bump", "extra_chrom_gauss_bump", "wn", "ecorr_split", "ecorr_split_kernel", "lin_exp_gw_squared", \
                "smbhb_frank_psradd8", "smbhb_frank_psradd10", "smbhb_frank_pp8", "smbhb_frank_pp10", "free_spec_red", "planet_wave", "sw_c", "ecorr_fast", "planet_roemer_1","planet_roemer_2","planet_roemer_3","planet_roemer_4","planet_roemer_5","planet_roemer_6","planet_roemer_7","planet_roemer_8","planet_roemer_9"})
parser.add_argument("-sampler", dest="sampler", choices={"bilby", "ptmcmc","ppc", "hyper", "ultra", "pbilby"}, required=True)
parser.add_argument("-pool",dest="pool", type=int, help="Number of cores to request (default=1)")
parser.add_argument("-nlive", dest="nlive", type=int, help="Number of nlive points to use (default=1000)")
parser.add_argument("-alt_dir", dest="alt_dir", help=r"With this the head result directory will be altered. i.e. instead of out_ppc it would be {whatever}/{is_wanted}/{pulsar}_{results}", required = False)
parser.add_argument("-sse", dest="sse", type = str.upper, help=r"Choose an alternative solar system ephemeris to use (SSE). Default is DE440.", required = False)

args = parser.parse_args()

pulsar = str(args.pulsar)
results_dir = str(args.results)
noisefile = args.noisefile
modelfile = args.modelfile
noise = args.noise_search
sampler = args.sampler
pool = args.pool
nlive=args.nlive
partim = args.partim
custom_results = str(args.alt_dir)
sse = str(args.sse)

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
    t0_dm_bump = parameter.Uniform(tmin[0],tmax[0])
    sigma_dm_bump = parameter.Uniform(sigma_min,tmax[0]-tmin[0])
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

@signal_base.function
def planet_sinusoid(toas, log10_Amp, phase, log10_orb_f):
    """
    Chromatic annual sinusoid.
    :param log10_Amp: amplitude of sinusoid
    :param phase: initial phase of sinusoid

    :return wf: delay time-series [s]
    """

    
    wf = 10**log10_Amp * np.sin(2 * np.pi * (10**log10_orb_f) * toas + phase)
    return wf

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


@signal_base.function
def fit_planet_app2(toas,psrMass,mass,period,phase,omega,ecc,ismasslog = False,isecclog=False):
    # replace the highly-degenerate time of periastron
    # with a 'phase', i.e true anomaly/2pi, between [0,1), at a specific time,
    # e.g. 55000 MJD
    if ismasslog:
        mass = 10**mass
    if isecclog:
        ecc = 10**ecc

    Omega_b = 2.0*np.pi/(units.day*period)
    e = ecc

    tref = 55000 #MJD
    tref = tref*units.day
    if phase <= 0.5:
        Eref = np.arccos((e+np.cos(2*np.pi*phase))/(1+e*np.cos(2*np.pi*phase)))
    else:
        Eref = 2*np.pi-np.arccos((e+np.cos(2*np.pi*phase))/(1+e*np.cos(2*np.pi*phase)))

    mean_anom_ref = Eref - e*np.sin(Eref)
    t0 = tref - mean_anom_ref/Omega_b #should be in sec

    #inc=inc*units.degree

    M1 = psrMass * units.M_sun
    M2 = mass * units.M_earth
    Mtot = M1+M2
    Mr = M2**3 / Mtot**2
    a1 = np.power(Mr*aconst.G/Omega_b**2,1.0/3.0).to(units.m)
    #asini = a1 * np.sin(inc)
    asini = a1
    om=coord.Angle(omega*units.deg)

    def get_roemer(t):

        def ecc_anom(E,e,M):
            return (E-e*np.sin(E))-M

        mean_anom = coord.Angle((Omega_b * (t*units.s - t0)).decompose().value*units.rad)

        mean_anom.wrap_at(2*np.pi*units.rad,inplace=True)
        mean_anom = mean_anom.rad
        
        ### read/interp E solution from appropriate ecc file:
        
        e_app = np.around(e,decimals=5)
        #print('\n????? Dir: ',os.getcwd())
        path_to_ecc = "/fred/oz002/users/mmiles/MPTA_planet_checker/model_components/"
        sampled_mean_anom,sampled_E = np.load(path_to_ecc+'E_e_5dig/E_e=%.5f.npy' %e_app)
        E_from_m = interpolate.interp1d(sampled_mean_anom,sampled_E,copy=False,kind='cubic')
        #if (mean_anom<1e-6).any():
        #    print(e,e_app)
        #    print(mean_anom[mean_anom<1e-6])
	#print('-------- MEAN ANOM: '+str(mean_anom))
        #print('-------- min Sampled: '+str(min(sampled_mean_anom)))
        E = E_from_m(mean_anom)
 
        roemer = (asini*(np.cos(E)-e)*np.sin(om) + asini*np.sin(E)*np.sqrt(1.0-e**2)*np.cos(om))/aconst.c

        return roemer

    roemer = get_roemer(toas)
    return roemer.to(units.s).value


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
    if "ecorr_fast" == n:
        ecorr = parameter.Uniform(-10,-1) 
    if "ecorr_split" == n:
        ecorr = parameter.Uniform(-10,-1) 
    if "ecorr_split_kernel" == n:
        ecorr = parameter.Uniform(-10,-1) 
    if "ecorr_gauss" == n:
        ecorr = parameter.Uniform(-10,-1)
    if "ecorr_gauss_c" == n:
        ecorr = parameter.Constant()
    if "ecorr_c" == n:
        ecorr = parameter.Constant()
    if "ecorr_split_c" == n:
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
    
    if "hc_red" == n:
        log10_A_hfred = parameter.Uniform(-20, -11)
        gamma_hfred = parameter.Uniform(0, 7)

    if "free_spec_red" == n:
        log10_rho = parameter.Uniform(-9, -4, size=5)

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
    if "hc_dm" == n:
        log10_A_dm = parameter.Uniform(-20, -11)
        gamma_dm = parameter.Uniform(0, 7)

    if "sw" == n:
        n_earth = parameter.Uniform(0, 20)
        log10_A_sw = parameter.Uniform(-10, 1)
        gamma_sw = parameter.Uniform(-4, 4)
    if "sw_c" == n:
        n_earth = parameter.Constant()
        log10_A_sw = parameter.Constant()
        gamma_sw = parameter.Constant()

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
    if "hc_chrom_cidx" == n:
        log10_A_chrom_prior = parameter.Uniform(-20, -11)
        gamma_chrom_prior = parameter.Uniform(0, 7)
        chrom_gp_idx = parameter.Constant(4)
    if "hc_chrom" == n:
        log10_A_hc_chrom_prior = parameter.Uniform(-20, -11)
        gamma_hc_chrom_prior = parameter.Uniform(0, 7)
        hc_chrom_gp_idx = parameter.Uniform(0,7)

    if "addchrom" == n:
        log10_A_add_chrom_prior = parameter.Uniform(-20, -11)
        gamma_add_chrom_prior = parameter.Uniform(0, 14)
        add_chrom_gp_idx = parameter.Uniform(0,14)
    if "chrom_annual" == n:
        log10_Amp_chrom1yr = parameter.Uniform(-20, -5)
        phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
        idx_chrom1yr = parameter.Uniform(0, 14)
    if "chrom_annual_c" == n:
        log10_Amp_chrom1yr = parameter.Constant()
        phase_chrom1yr = parameter.Constant()
        idx_chrom1yr = parameter.Constant()

    if "planet_wave" == n:
        log10_Amp_planet = parameter.Uniform(-20, -5)
        phase_planet = parameter.Uniform(0, 2*np.pi)
        log10_orb_f = parameter.Uniform(-9, -6)

    if "planet_roemer_1" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(5,10.1)

    if "planet_roemer_2" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(10.1, 21.3)

    if "planet_roemer_3" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(21.3, 42.5)
    
    if "planet_roemer_4" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(42.5, 85)

    if "planet_roemer_5" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(85, 170)

    if "planet_roemer_6" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(170, 340)

    if "planet_roemer_7" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(340, 390)

    if "planet_roemer_8" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(390, 780)

    if "planet_roemer_9" == n:
        
        psrMass = parameter.Constant(1.4)
        omega = parameter.Uniform(0, 2*np.pi)
        phase = parameter.Uniform(0, 1)
        ecc = parameter.Uniform(-9, np.log10(0.9))
        mass = parameter.Uniform(-5, -1)
        period = parameter.Uniform(780, 1560)

    if "gw" == n:
        log10_A_gw = parameter.Uniform(-20,-11)('log10_A_gw')
        gamma_gw = parameter.Uniform(0,7)('gamma_gw')
    if "gw_c" == n:
        log10_A_gw = parameter.Constant(-14.25)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_c_low" == n:
        log10_A_gw = parameter.Constant(-14.1)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_const_gamma" == n:
        log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_const_gamma_low" == n:
        log10_A_gw = parameter.Uniform(-20,-11)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "gw_const_gamma_wide" == n:
        log10_A_gw = parameter.Uniform(-20,-11)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
    if "lin_exp_gw_const_gamma_wide" == n:
        #log10_A_gw = bilby_warp.linearExponential(-18,-9,'log10_A_gw')
        log10_A_gw = parameter.LinearExp(-20,-11)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')

    # if "lin_exp_gw_squared" == n:
    #     log10_A_gw = parameter.LinearExp_Squared(-20,-11)('log10_A_gw')
    #     gamma_gw = parameter.Constant(4.33)('gamma_gw')

    if "lin_exp_gw_squared" == n:
        log10_A_gw = parameter.UserParameter(prior=parameter.Function(LinearExpPrior_Squared, pmin=-20, pmax=-11), sampler=LinearExpSampler_Squared)('log10_A_gw')
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
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection, method="sherman-morrison")
        s += ec
    if "ecorr_fast" == n:
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection, method="fast-sherman-morrison")
        s += ec
    if "ecorr_split" == n:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=mk_ecorr_selection)
        s += ec
    if "ecorr_split_kernel" == n:
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=mk_ecorr_selection, method="sherman-morrison")
        s += ec
    if "ecorr_gauss" == n:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)
        s += ec

    if "wn" == n:
        wn_json = json.load(open(noisefile))
        wnkeys = list(wn_json.keys())
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

        efac = parameter.Uniform(0.1,1.5)
        if "t2equad" in wnmodels or "tnequad" in wnmodels:
            equad = parameter.Uniform(-10,-4)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            s += eq
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef

        if "ecorr" in wnmodels:
            ecorr = parameter.Uniform(-10,-4)
            #ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)
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

    if "ecorr_gauss_c" == n:
        if pulsar+"_basis_ecorr_KAT_MKBF_log10_ecorr" in params.keys():
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)
            s += ec

    if "ecorr_split_c" == n:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=mk_ecorr_selection)
        s += ec

    if "ecorr_check" == n:
        if pulsar+"_KAT_MKBF_log10_ecorr" in params.keys():
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ecorr_selection)
            s += ec

    if "red" == n or "red_c" == n or "red_wide" == n:
        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
        s += rn

    if "free_spec_red" == n:
        #pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        spectrum = gp_priors.free_spectrum(log10_rho=log10_rho)
        rn = gp_signals.FourierBasisGP(spectrum=spectrum, components=5, Tspan=Tspan)
        s += rn

    if "hfred" == n:
        hfpl = utils.powerlaw(log10_A=log10_A_hfred, gamma=gamma_hfred)
        hfrn = gp_signals.FourierBasisGP(spectrum=hfpl, components=120, Tspan=Tspan, name="hfred")
        s += hfrn

    if "hc_red" == n:
        hfpl = utils.powerlaw(log10_A=log10_A_hfred, gamma=gamma_hfred)
        hfrn = gp_signals.FourierBasisGP(spectrum=hfpl, components=120, Tspan=Tspan, name="hfred")
        s += hfrn


    if "dm" == n or "dm_c" == n or "dm_wide" == n or "dm_wider" == n:
        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
        s += dm
    
    if "hc_dm" == n:
        dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
        s += dm

    if "chrom" == n or "chrom_wide" == n or "chrom_c" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 120
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
        s += chrom


    if "chrom_cidx" == n or "chrom_cidx_wide" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 120
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chromcidx_gp')
        s += chrom
    
    if "hc_chrom_cidx" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 120
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='hc_chromcidx_gp')
        s += chrom

    if "hc_chrom" == n:
        hc_chrom_model = utils.powerlaw(log10_A=log10_A_hc_chrom_prior,
                                    gamma=gamma_hc_chrom_prior)
        idx = hc_chrom_gp_idx
        components = 120
        hc_chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        hc_chrom = gp_signals.BasisGP(hc_chrom_model, hc_chrom_basis, name='hc_chrom_gp')
        s += hc_chrom

    if "chromsplit" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior,
                                    gamma=gamma_chrom_prior)
        idx = chrom_gp_idx
        components = 120
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, selection = chrom_split, name='chrom_gp_split')
        s += chrom

    if "addchrom" == n:
        chrom_model = utils.powerlaw(log10_A=log10_A_add_chrom_prior,
                                    gamma=gamma_add_chrom_prior)
        idx = add_chrom_gp_idx
        components = 120
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                idx=idx)
        add_chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='add_chrom_gp')
        s += add_chrom

    if "chrom_annual" == n or "chrom_annual_c" == n:
        wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr,
                            phase=phase_chrom1yr, idx=idx_chrom1yr)
        chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
        s += chrom1yr

    if "planet_wave" == n:
        wf = planet_sinusoid(log10_Amp=log10_Amp_planet, phase=phase_planet, log10_orb_f=log10_orb_f)
        planet_waveform = deterministic_signals.Deterministic(wf, name="planet_wave")
        s += planet_waveform

    if "planet_roemer" in n:

        # psrMass = parameter.Constant(1.4)
        # omega = parameter.Uniform(0, 2*np.pi)
        # phase = parameter.Uniform(0, 1)
        # ecc = parameter.Uniform(-9, np.log10(0.9))
        # mass = parameter.Uniform(-5, -1)
        # period = parameter.Uniform(5,10.1)

        planet = fit_planet_app2(psrMass=psrMass, mass=mass, period=period, phase=phase, omega=omega, ecc=ecc, ismasslog = True, isecclog=True)

        bin_number = n.split("_")[-1]
        planet_signal = deterministic_signals.Deterministic(planet, name='planet_roemer_bin_'+str(bin_number))
        s += planet_signal

    if "dm_gauss_bump" == n:
        gauss_bump = dm_gaussian_bump(tmin, tmax, idx=2, sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='dm_bump')
        s += gauss_bump

    if "chrom_gauss_bump" == n:
        gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
        s += gauss_bump

    if "gw" == n or "gw_c" == n or "gw_const_gamma" == n or "gw_const_gamma_wide" == n or "lin_exp_gw_const_gamma_wide" == n or "lin_exp_gw_squared" == n:
        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw

    if "gw_const_gamma_low" == n or "gw_c_low" ==n:
        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=5, Tspan=Tspan, name='gwb')
        s += gw

    if "band_low"== n or "band_low_c" == n:
        #max_cadence = 60  # days
        #band_components = int(Tspan / (max_cadence*86400))
        band_components = 120
        bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                    selection=low_freq, name='low_band_noise')
        s += bnl

    if "band_high"== n or "band_high_wide" == n or "band_high_c" == n:
        #max_cadence = 60  # days
        #band_components = int(Tspan / (max_cadence*86400))
        band_components = 120
        bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
        bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                    selection=high_freq, name='high_band_noise')
        s += bnh

    if "sw" == n or "sw_c" == n:
        #n_earth = parameter.Uniform(0, 20)
        deter_sw = solar_wind(n_earth=n_earth)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

        Tspan = psr.toas.max() - psr.toas.min()
        max_cadence = 60
        #sw_components = int(Tspan / (max_cadence*86400))
        sw_components = 120
        #log10_A_sw = parameter.Uniform(-10, 1)
        #gamma_sw = parameter.Uniform(-4, 4)
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

    if "spgw" == n or "spgwc" == n or "spgwcm" == n or "spgwc_18" == n or "spgwc_wide" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
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
                band_components = 120
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
                band_components = 120
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

        if "spgwc" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')
        
        elif "spgwc_wide" == n:
            log10_A_gw = parameter.Uniform(-20,-11)('log10_A_gw')
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

        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -12)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=120, Tspan=Tspan, name='gwb')
        s += gw


    if "free_spgw" == n or "free_spgwc" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_models.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120

        wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_values_SGWBWN_checked.json"))
        wnkeys = list(wn_json.keys())
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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


        if "free_spgwc" == n or "free_spgw" == n:
            #J1909 timespan
            Tspan_pta = 140723541.0264778
            
            crn = blocks.common_red_noise_block(psd='spectrum', prior='log-uniform', Tspan = Tspan_pta, components=30, orf=None, name='gw', delta_val=0)
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
        sw_components = 120

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

        pmev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_models.json"))
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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
                s += rn

            if pm =="DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 7)
                chrom_gp_idx = parameter.Uniform(0,7)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 120
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
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
                s += chrom

            if pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 7)
                chrom_gp_idx = parameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 120
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

            if pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 120
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 120
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 120
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 120
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
            sw_components = 120
            log10_A_sw = parameter.Uniform(-10, 1)
            gamma_sw = parameter.Uniform(-4, 4)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
            sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

            sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

            s += sw
        
    if "mpta_pm" == n:

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
            
        
    if "mpta_pm_gauss_ecorr" == n:

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
            s += gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=selection)

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
                s += rn

            if pm == "RNWIDE" or ( pm == "RN" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_red = parameter.Uniform(-20, -11)
                gamma_red = parameter.Uniform(0, 14)
                pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
                s += rn

            if pm =="DM":
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm
            
            if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_dm = parameter.Uniform(-20, -11)
                gamma_dm = parameter.Uniform(0, 14)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 7)
                chrom_gp_idx = parameter.Uniform(0,7)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 120
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
                log10_A_chrom_prior = parameter.Uniform(-20, -11)
                gamma_chrom_prior = parameter.Uniform(0, 7)
                chrom_gp_idx = parameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = 120
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

            if pm == "BL":
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 120
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BLWIDE" or ( pm == "BL" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 120
                bpl = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnl = gp_signals.FourierBasisGP(bpl, components=band_components,
                                            selection=low_freq, name='low_band_noise')
                s += bnl

            if pm == "BH":
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 7)
                band_components = 120
                bph = utils.powerlaw(log10_A=log10_A_bn, gamma=gamma_bn)
                bnh = gp_signals.FourierBasisGP(bph, components=band_components,
                                            selection=high_freq, name='high_band_noise')
                s += bnh

            if pm == "BHWIDE" or ( pm == "BH" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
                log10_A_bn = parameter.Uniform(-20, -11)
                gamma_bn = parameter.Uniform(0, 14)
                band_components = 120
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
        sw_components = 120

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
            sw_components = 120

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
            sw_components = 120

            log10_A_sw = parameter.Uniform(-10, 1)
            gamma_sw = parameter.Uniform(-4, 4)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
            sw_basis = createfourierdesignmatrix_solar_dm(nmodes=high_comps, Tspan=Tspan)

            sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

            s += sw



    if "smbhb" == n or "smbhb_const" == n or "smbhb_wn" == n or "smbhb_wn_all" == n or "smbhb_const_wn" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120

        wn_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_WN_values.json"))
        wnkeys = list(wn_json.keys())
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]
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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb" == n or "smbhb_wn" == n or "smbhb_wn_all" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')
        
        if "smbhb_const" == n or "smbhb_const_wn" == n:
            log10_A_gw = parameter.Constant(-14.25)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')

        if "smbhb_wn" == n or "smbhb_const_wn" == n:
            efac = parameter.Uniform(0.1,1.5)
            if "t2equad" in wnmodels or "tnequad" in wnmodels:
                equad = parameter.Uniform(-10,-5)
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef
                eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
                s += eq
            else:
                ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
                s += ef

            if "ecorr" in wnmodels:
                ecorr = parameter.Uniform(-10,-5)
                ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
                s += ec

        if "smbhb_wn_all" == n:
            efac = parameter.Uniform(0.1,1.5)
            equad = parameter.Uniform(-10,-5)
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
            s += ef
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
            s += eq

            ecorr = parameter.Uniform(-10,-5)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            s += ec

        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump


        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=120, Tspan=Tspan, name='gwb')
        s += gw

    if "smbhb_frank_pp10" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_10/MPTA_noise_models_PP10.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120
        wn_json = json.load(open(noisefile))
        keys = list(ev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb_frank_pp10" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')


        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump


        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw


    if "smbhb_frank_pp8" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120
        wn_json = json.load(open(noisefile))
        keys = list(ev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb_frank_pp8" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')


        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump


        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw


    if "smbhb_frank_psradd10" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/MPTA_noise_models_PSRADD10.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120
        wn_json = json.load(open(noisefile))
        keys = list(ev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb_frank_psradd10" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')


        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump


        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw



    if "smbhb_frank_psradd8" == n:
        ev_json = json.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/MPTA_noise_models_PSRADD8.json"))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120
        wn_json = json.load(open(noisefile))
        keys = list(ev_json.keys())
        wnkeys = list(wn_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb_frank_psradd8" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')


        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump


        gpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gw = gp_signals.FourierBasisGP(spectrum=gpl, components=30, Tspan=Tspan, name='gwb')
        s += gw

    if "planet_checker" == n:
        # !!! Not finished !!! #
        ev_json = json.load(open(modelfile))
        keys = list(ev_json.keys())
        # Get list of models
        psrmodels = [ psr_model for psr_model in keys if pulsar in psr_model ][0].split("_")[1:]
        high_comps = 120
        wn_json = json.load(open(noisefile))
        wnkeys = list(wn_json.keys())
        # Get list of wn models
        wnmodels = [ wn_model.split("_")[-1] for wn_model in wnkeys if pulsar in wn_model ]

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
                rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
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
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
                s += dm

            if pm == "CHROM":
                if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
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
                elif i+1 == len(psrmodels):
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
                log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                idx_chrom1yr = parameter.Uniform(0, 14)
                wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                s += chrom1yr

            if pm == "CHROMBUMP":
                chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
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

        if "smbhb_frank_psradd8" == n:
            log10_A_gw = parameter.Uniform(-18,-12)('log10_A_gw')
            gamma_gw = parameter.Constant(4.33)('gamma_gw')


        for i, n in enumerate(noise):
            if "extra_red" == n:
                if len([ pn for pn in psrmodels if "RN" in pn ]) == 0:
                    log10_A_red = parameter.Uniform(-20, -11)
                    gamma_red = parameter.Uniform(0, 7)
                    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
                    rn = gp_signals.FourierBasisGP(spectrum=pl, components=high_comps, Tspan=Tspan)
                    s += rn

        for i, n in enumerate(noise):
            if "extra_chrom_annual" == n:
                if len([ pn for pn in psrmodels if "CHROMANNUAL" in pn ]) == 0:
                    log10_Amp_chrom1yr = parameter.Uniform(-20, -4)
                    phase_chrom1yr = parameter.Uniform(0, 2*np.pi)
                    idx_chrom1yr = parameter.Uniform(0, 14)
                    wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_chrom1yr, phase=phase_chrom1yr, idx=idx_chrom1yr)
                    chrom1yr = deterministic_signals.Deterministic(wf, name="chrom1yr")
                    s += chrom1yr
        
        for i, n in enumerate(noise):
            if "extra_chrom_gauss_bump" == n:
                if len([ pn for pn in psrmodels if "CHROMBUMP" in pn ]) == 0:
                    chrom_gauss_bump = dm_gaussian_bump(tmin, tmax, idx="vary", sigma_min=604800, sigma_max=tmax[0]-tmin[0], log10_A_low=-10, log10_A_high=-1, name='chrom_bump')
                    s += chrom_gauss_bump





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

elif sampler =="hyper":
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
    
    N = int(4e5)
    sampler = hyper_model.setup_sampler(outdir=outDir, resume=True)

    sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

    os.system("touch "+outDir+"/finished")


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

    outDir=header_dir+'/{0}_{1}'.format(pulsar,results_dir)
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
    nlive=nlive
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



