import numpy as np
import pickle
import json
import glob
import os

import matplotlib.pyplot as plt

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as eparameter
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
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import time
import h5py
import dill


#import pint
#from pint.models import *

#import pint.fitter
#from pint.residuals import Residuals
#from pint.toa import get_TOAs
#import pint.logging
#import pint.config

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
        rho_avg[i],sig_avg[i]=weightedavg(myrhos,mysigs)
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

#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"
# psrlist=[ x.strip("\n") for x in open(psr_list).readlines() ]
datadir='/fred/oz002/users/mmiles/MPTA_GW/SMBHB/active_partim/'

parfiles=sorted(glob.glob(datadir+'*.par'))
timfiles=sorted(glob.glob(datadir+'*.tim'))
# if psrlist is not None:
#     parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
#     timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]

# timfiles = [ tim  for tim in timfiles if "all" not in tim ]
# timfiles = [ tim  for tim in timfiles if "off" not in tim ]
#timfiles = [ tim  for tim in timfiles if "_red" not in tim ]
#parfiles = [ par  for par in parfiles if "_red" not in par ]

psrs = []
ephemeris =  'DE440'
#ephemeris = "DE438"
#ephemeris = "DE421"

for p, t in zip(parfiles,timfiles):

    print(p)
    psr=Pulsar(p,t,ephem=ephemeris)
    #print(psr)
    if psr.name != "J1902-5105":
    #if psr.name != "J1902-5105" and psr.name !="J1431-5740" and psr.name != "J1643-1224" and psr.name != "J1708-3506" and psr.name != "J1802-2124":
        # if psr.name != "J1708-3506":
        #     if psr.name != "J2236-5527":
        psrs.append(psr)



noisefile = '/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/MPTA_WN_models.json'

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

efac = eparameter.Constant()
equad = eparameter.Constant()
ecorr = eparameter.Constant()

models = []

crn = blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=30,gamma_val=4.33, name='gw')

#s = gp_signals.TimingModel()
#s += blocks.white_noise_block(vary=False, inc_ecorr=True, select='backend')
#s += blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=120)
#s += blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, 
#                                   components=120, gamma_val=4.33, name='gw')


for pulsar in psrs:

    #tm = gp_signals.TimingModel()
    #s = tm
    s = gp_signals.TimingModel()
    #s += blocks.white_noise_block(vary=False,inc_ecorr=True,select=no_selection)
    #mn = white_signals.MeasurementNoise(selection=by_backend)
    #equad = white_signals.TNEquadNoise(selection=by_backend)
    #ecorr = gp_signals.EcorrBasisModel(selection=by_backend)

    if pulsar.name+"_KAT_MKBF_log10_tnequad" in noisedict.keys():
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=by_backend)
        s += eq
    
    ef = white_signals.MeasurementNoise(efac=efac, selection=by_backend)
    s += ef
    
    if pulsar.name+"_basis_ecorr_KAT_MKBF_log10_ecorr" in noisedict.keys():
        #ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=by_backend)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,selection=by_backend)
        s += ec
            

    max_cadence = 60  # days
    #components = int(Tspan / (max_cadence*86400))
    components = 120
    freqs = np.linspace(1/Tspan,30/Tspan,30)

    #s += blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=120)
    #s += blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, 
    #                               components=120, gamma_val=4.33, name='gw')
    
    ev_json = json.load(open('/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/MPTA_noise_models.json'))
    keys = list(ev_json.keys())
    # Get list of models
    psrmodels = [ psr_model for psr_model in keys if pulsar.name in psr_model ][0].split("_")[1:]

    # log10_A_red = eparameter.Uniform(-20, -11)
    # gamma_red = eparameter.Uniform(0, 7)
    # pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
    # rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
    # s += rn

    high_comps = 120
    for i, pm in enumerate(psrmodels):
        if pm =="DM":
            if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_dm = eparameter.Uniform(-20, -11)
                gamma_dm = eparameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                s += dm
            elif i+1 == len(psrmodels):
                log10_A_dm = eparameter.Uniform(-20, -11)
                gamma_dm = eparameter.Uniform(0, 7)
                dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=high_comps,option="powerlaw")
                s += dm 
        
        if pm == "DMWIDE" or ( pm == "DM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
            log10_A_dm = eparameter.Uniform(-20, -11)
            gamma_dm = eparameter.Uniform(0, 14)
            dm = dm_noise(log10_A=log10_A_dm,gamma=gamma_dm,Tspan=Tspan,components=120,option="powerlaw")
            s += dm

        # if pm == "RN":
        #     log10_A_red = eparameter.Uniform(-20, -11)
        #     gamma_red = eparameter.Uniform(0, 7)
        #     pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
        #     rn = gp_signals.FourierBasisGP(spectrum=pl, components=120, Tspan=Tspan)
        #     s += rn

        if pm == "CHROM":
            if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_chrom_prior = eparameter.Uniform(-20, -11)
                gamma_chrom_prior = eparameter.Uniform(0, 7)
                chrom_gp_idx = eparameter.Uniform(0,7)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom
            elif i+1 == len(psrmodels):
                log10_A_chrom_prior = eparameter.Uniform(-20, -11)
                gamma_chrom_prior = eparameter.Uniform(0, 7)
                chrom_gp_idx = eparameter.Uniform(0,7)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom

        if pm == "CHROMWIDE" or ( pm == "CHROM" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
            log10_A_chrom_prior = eparameter.Uniform(-20, -11)
            gamma_chrom_prior = eparameter.Uniform(0, 14)
            chrom_gp_idx = eparameter.Uniform(0,14)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 120
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
            s += chrom

        if pm == "CHROMCIDX":
            if ( i+1 < len(psrmodels) and psrmodels[i+1] != "WIDE" ):
                log10_A_chrom_prior = eparameter.Uniform(-20, -11)
                gamma_chrom_prior = eparameter.Uniform(0, 7)
                chrom_gp_idx = eparameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom
            elif i+1 == len(psrmodels):
                log10_A_chrom_prior = eparameter.Uniform(-20, -11)
                gamma_chrom_prior = eparameter.Uniform(0, 7)
                chrom_gp_idx = eparameter.Constant(4)
                chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
                idx = chrom_gp_idx
                components = high_comps
                chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                        idx=idx)
                chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')
                s += chrom

        if pm == "CHROMCIDXWIDE" or ( pm == "CHROMCIDX" and ( i+1 < len(psrmodels) and psrmodels[i+1] == "WIDE" ) ):
            log10_A_chrom_prior = eparameter.Uniform(-20, -11)
            gamma_chrom_prior = eparameter.Uniform(0, 14)
            chrom_gp_idx = eparameter.Constant(4)
            chrom_model = utils.powerlaw(log10_A=log10_A_chrom_prior, gamma=gamma_chrom_prior)
            idx = chrom_gp_idx
            components = 120
            chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components,
                                                                    idx=idx)
            chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_wide_gp')
            s += chrom

        if pm == "SW":
            n_earth = eparameter.Uniform(0, 20)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            Tspan = psr.toas.max() - psr.toas.min()
            max_cadence = 60
            sw_components = 120

            log10_A_sw = eparameter.Uniform(-10, 1)
            gamma_sw = eparameter.Uniform(-4, 4)
            sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
            sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)

            sw = mean_sw + gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

            s += sw
        
        if pm == "SWDET":
            n_earth = eparameter.Uniform(0, 20)
            deter_sw = solar_wind(n_earth=n_earth)
            mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

            sw = mean_sw

            s += sw
        
    if "SW" not in psrmodels and "SWDET" not in psrmodels:
        n_earth = eparameter.Constant(4)
        deter_sw = solar_wind(n_earth=n_earth)
        mean_sw = deterministic_signals.Deterministic(deter_sw, name='n_earth')

        sw = mean_sw

        s += sw

    s += crn

    models.append(s(pulsar))


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

#pta = signal_base.PTA([s(p) for p in psrs])

pta = signal_base.PTA(models)

pta.set_default_params(noisedict)

#pta = dill.load(open("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_CRN_ER.pkl","rb"))

ostat = opt_stat.OptimalStatistic(psrs, pta=pta, orf='hd', bayesephem=False)
ostat_dip = opt_stat.OptimalStatistic(psrs, pta=pta, orf='dipole',bayesephem=False)
ostat_mono = opt_stat.OptimalStatistic(psrs, pta=pta, orf='monopole',bayesephem=False)

#with open('./MPTA_WN_models_2.json', 'r') as f:
#    ml_params = json.load(f)

#with open('/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/MPTA_noise_values_allpulsars.json', "r") as f:
#with open('/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/MPTA_noise_values_allpulsars_incRED.json', "r") as f:
with open('/fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/MPTA_noise_values_allpulsars_ER_SGWBconst.json', "r") as f:
    ml_params = json.load(f)

# chain_file = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_masterChain_0206.npy"
# marg_chain = np.load(chain_file)
# pars_file = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_run_1/pars.txt"

# parlist = list(open(pars_file))
# parlist = [ par.strip("\n") for par in parlist ]

# marg_no_gam = np.delete(marg_chain,-6,1)
# parlist.remove("gw_gamma")

# mxi, mrho, msig, mOS, mOS_SNR = ostat.compute_noise_marginalized_os(marg_no_gam, param_names=parlist, N=1000)
# print(mOS, mOS_SNR)


xi, rho, sig, OS, OS_sig = ostat.compute_os(params=ml_params)
print(OS, OS_sig, OS/OS_sig)

_, _, _, OS_dip, OS_sig_dip = ostat_dip.compute_os(params=ml_params)
print(OS_dip, OS_sig_dip, OS_dip/OS_sig_dip)

_, _, _, OS_mono, OS_sig_mono = ostat_mono.compute_os(params=ml_params)
print(OS_mono, OS_sig_mono, OS_mono/OS_sig_mono)

#ximult, rhomult, sigmult, OSmult, OS_sigmult = opt_stat.OptimalStatistic.compute_multiple_corr_os(params=ml_params, psd='powerlaw', fgw=None, correlations=['dipole', 'hd'])
#print(OSmult, OS_sigmult, OSmult/OS_sigmult)

# mrho_med = np.median(rho, axis=0)
# msig_med = np.median(sig, axis=0)

idx=np.argsort(xi)

xi_sorted = xi[idx]
rho_sorted = rho[idx]
sig_sorted = sig[idx]


npsr=np.shape(psrs)[0]

gof=np.zeros(npsr)
rho2=np.zeros(npsr)
sig2=np.zeros(npsr)

rho_expected=2.51e-29*get_HD_curve(xi)

f = open("crosscorr_noisemarg.dat", "w")

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
        f.write("{0} {1} {2} {3} {4} {5} \n".format(psrs[ii].name,psrs[jj].name, rho[i]/sig[i],rho[i],sig[i],xi[i]))
        i=i+1

f.close()

for i in range(npsr):
    print(psrs[i].name, gof[i], rho2[i],sig2[i])





#npairs=120
npairs = 300

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
plt.title("OS: {0:.2E}; SNR: {1:.2f}".format(OS, OS/OS_sig))
plt.plot(zeta, OS*HD, ls='--', label='Hellings-Downs', color='C0', lw=1.5)
plt.plot(zeta, zeta*0.0+OS_mono, ls='--', label='Monopole', color='C1', lw=1.5)
plt.plot(zeta, OS_dip*np.cos(zeta*np.pi/180), ls='--', label='Dipole', color='C2', lw=1.5)

plt.xlim(0, 180)
#plt.ylim(-1e-29, 1e-29)
plt.ylabel(r'$\hat{A}^2 \Gamma_{ab}(\zeta)$')
plt.xlabel(r'$\zeta$ (deg)')

plt.legend(loc=4)

plt.tight_layout()
#plt.show()
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/SMBHB/OS_runs/OS_new_models_noER_noRED_SGWBconst_iter_noJ1902.png")
plt.clf()
