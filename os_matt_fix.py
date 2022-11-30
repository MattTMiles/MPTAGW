import numpy as np
import pickle
import json
import glob
import os

import matplotlib.pyplot as plt

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import gp_signals

from enterprise.signals import utils
from enterprise.signals import gp_bases
from enterprise.signals import gp_priors
from enterprise.signals import parameter

from enterprise_extensions import model_utils, blocks
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from enterprise.signals  import selections
from enterprise.signals import white_signals
from enterprise.signals import deterministic_signals
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import time
import h5py

def weightedavg(rho,sig):
    weights, avg=0.,0.
    for r,s in zip(rho,sig):
        weights += 1./(s*s)
        avg += r/(s*s)

    return avg/weights, np.sqrt(1./weights)

def bin_crosscor(zeta,zi,rho,sig):

    rho_avg,sig_avg=np.zeros(len(zeta)), np.zeros(len(zeta))

    for i,z in enumerate(zeta[:-1]):
        myrhos, mysigs = [], []
        for x,r,s in zip(xi,rho,sig):
            if x>=z and x < (z+10):
                myrhos.append(r)
                myrhos.append(s)
        rho_avg[i],sig_avg[i]=weightavg(myrhos,mysigs)
    return rho_avg, sig_avg

def get_HD_curve(zeta):
    coszeta=np.cos(zeta*np.pi/180.)
    xip=(1.-coszeta)/2.
    HD=3.*(1./3.+xip*(np.log(xip) -1./6.))

    return HD/2.



no_selection=selections.Selection(selections.no_selection)
by_backend = selections.Selection(selections.by_backend)

def low_frequencies(freqs):
    """Selection for obs frequencies <=960MHz"""
    return dict(zip(['low'], [freqs <= 1284]))

def high_frequencies(freqs):
    """Selection for obs frequencies >=2048MHz"""
    return dict(zip(['high'], [freqs > 1284]))

low_freq = selections.Selection(low_frequencies)
high_freq = selections.Selection(high_frequencies)

psrlist = None

datadir='/fred/oz005/users/rshannon/correct_toas_3/'

parfiles=sorted(glob.glob(datadir+'*.par'))
timfiles=sorted(glob.glob(datadir+'*.tim'))

psrs = []
ephemeris =  'DE440'
#ephemeris = "DE438"
#ephemeris = "DE421"


for p, t in zip(parfiles,timfiles):
    if not p.startswith("J2322+2057"):
        psr=Pulsar(p,t,ephem=ephemeris)
        print(psr)
        psrs.append(psr)
        time.sleep(1)



noisefile='/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/equad_check/MPTA_ALL_NOISE_EQUAD_CHECKED.json'

with open(noisefile, 'r') as f:
    noisedict=json.load(f)

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

Tspan=model_utils.get_tspan(psrs)

efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()

models = []

crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=4,gamma_val=4.33, name='gw')

for pulsar in psrs:


    tm=gp_signals.TimingModel()
    s = tm

    #s += blocks.white_noise_block(vary=False,inc_ecorr=True,select=no_selection)
    #mn = white_signals.MeasurementNoise(selection=by_backend)
    #equad = white_signals.TNEquadNoise(selection=by_backend)
    #ecorr = gp_signals.EcorrBasisModel(selection=by_backend)

    if pulsar.name+"_KAT_MKBF_log10_t2equad" in noisedict.keys():
        ef = white_signals.MeasurementNoise(efac=efac,log10_t2equad=equad, selection=by_backend)
        s += ef
    else:
        ef = white_signals.MeasurementNoise(efac=efac, selection=by_backend)
        s += ef
    
    #if pulsar.name+"_KAT_MKBF_log10_ecorr" in noisedict.keys():
    #    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=by_backend)
    #    s += ec

    max_cadence = 60  # days
    components = int(Tspan / (max_cadence*86400))

    freqs = np.linspace(1/Tspan,30/Tspan,30)

    
    ev_json = json.load(open('/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/equad_check/MPTA_ALL_NOISE_EQUAD_CHECKED.json'))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar.name in psr_model ][0].split("_")[1:]
    
    if "RN" in psrmodels or "RED" in psrmodels:
        log10_A_red = parameter.Constant()
        gamma_red = parameter.Constant()
        pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
        s += rn
    '''
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
    
    '''
    s += crn

    models.append(s(pulsar))

bayesephem=False
if bayesephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                    model='setIII_1980')
    models.append(eph)



#log10_A = parameter.Uniform(-20, -11)
#gamma = parameter.Uniform(0, 7)

#log10_A_dm = parameter.Uniform(-20, -10)
#gamma_dm = parameter.Uniform(0, 7)

#gw_log10_A = parameter.Uniform(-18, -14)('gw_log10_A')
#gw_gamma = parameter.Constant(13./3)('gw_gamma')

#dm_basis = utils.createfourierdesignmatrix_dm(modes=freqs)
#dm_pl = utils.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
#dm_gp = gp_signals.BasisGP(priorFunction=dm_pl, basisFunction=dm_basis, name='dm_gp')#components=30,Tspan=Tspan)

#pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
#rn = gp_signals.FourierBasisGP(spectrum=pl, modes=freqs)#components=30,Tspan=Tspan)


#gw_pl = utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma)
#gw = gp_signals.FourierBasisGP(spectrum=gw_pl, modes=freqs[:10], name='gw')

#crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=4,gamma_val=4.33, name='gw')
#rn= blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=10)

#s = tm + crn + mn + equad
#s = tm + crn + mn + equad +ecorr 
#s = tm +  rn + crn + mn + equad   


pta = signal_base.PTA(models)

pta.set_default_params(noisedict)
'''
raise a
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

cov = np.diag(np.ones(ndim) * 0.01**2)

outdir = "/fred/oz002/users/mmiles/MPTA_GW/bayesEphem/"
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, 
                outDir=outdir, resume=True)

N = int(1e4)  # This will have to be played around with a bit
x0 = np.hstack([p.sample() for p in pta.params])
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )
'''



ostat = opt_stat.OptimalStatistic(psrs, pta=pta, orf='hd', bayesephem=True)
ostat_dip = opt_stat.OptimalStatistic(psrs, pta=pta, orf='dipole',bayesephem=True)
ostat_mono = opt_stat.OptimalStatistic(psrs, pta=pta, orf='monopole',bayesephem=True)

#with open('./MPTA_WN_models_2.json', 'r') as f:
#    ml_params = json.load(f)

with open('/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/equad_check/MPTA_ALL_NOISE_EQUAD_CHECKED.json', "r") as f:
    ml_params = json.load(f)

#ml_params = 

xi, rho, sig, OS, OS_sig = ostat.compute_os(params=ml_params)
print(OS, OS_sig, OS/OS_sig)

_, _, _, OS_dip, OS_sig_dip = ostat_dip.compute_os(params=ml_params)
print(OS_dip, OS_sig_dip, OS_dip/OS_sig_dip)

_, _, _, OS_mono, OS_sig_mono = ostat_mono.compute_os(params=ml_params)
print(OS_mono, OS_sig_mono, OS_mono/OS_sig_mono)


idx=np.argsort(xi)

xi_sorted = xi[idx]
rho_sorted = rho[idx]
sig_sorted = sig[idx]


npsr=np.shape(psrs)[0]

gof=np.zeros(npsr)
rho2=np.zeros(npsr)
sig2=np.zeros(npsr)

rho_expected=4e-30*get_HD_curve(xi)


i=0
for ii in range(npsr-1):
    for jj in range(ii+1,npsr):
        #print(i, ii,jj)
        gof[ii] += ((rho[i]-rho_expected[i])/sig[i])**2
        gof[jj] += ((rho[i]-rho_expected[i])/sig[i])**2
        rho2[ii] += (rho[i]-rho_expected[i])**2
        rho2[jj] += (rho[i]-rho_expected[i])**2
        sig2[ii] += sig[i]**2
        sig2[jj] += sig[i]**2
        #print(ii,jj, rho[i]/sig[i],rho[i],sig[i],xi[i])
        i=i+1


for i in range(npsr):
    print(i, gof[i], rho2[i],sig2[i])





#npairs=120
npairs = 140

xi_mean = []
xi_err = []
rho_avg = []
sig_avg = []

i=0

while i < len(xi_sorted):
    xi_mean.append(np.mean(xi_sorted[i:npairs+i]))
    xi_err.append(np.std(xi_sorted[i:npairs+i]))

    r,s=weightedavg(rho_sorted[i:npairs+i],sig_sorted[i:npairs+i])
    rho_avg.append(r)
    sig_avg.append(s)

    i+= npairs

xi_mean=np.array(xi_mean)
xi_err=np.array(xi_err)


(_, caps, _) = plt.errorbar(xi_mean*180/np.pi, rho_avg, xerr=xi_err*180/np.pi, yerr=sig_avg, marker='o', ls='', 
                                    color='0.1', fmt='o', capsize=4, elinewidth=1.2)


zeta=np.linspace(0.01,180,100)

HD=get_HD_curve(zeta)

plt.plot(zeta, OS*HD, ls='--', label='Hellings-Downs', color='C0', lw=1.5)
plt.plot(zeta, zeta*0.0+OS_mono, ls='--', label='Monopole', color='C1', lw=1.5)
plt.plot(zeta, OS_dip*np.cos(zeta*np.pi/180), ls='--', label='Dipole', color='C2', lw=1.5)

plt.xlim(0, 180)
plt.ylim(-1e-29, 1e-29)
plt.ylabel(r'$\hat{A}^2 \Gamma_{ab}(\zeta)$')
plt.xlabel(r'$\zeta$ (deg)')

plt.legend(loc=4)

plt.tight_layout()
plt.show()

