#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
import seaborn as sbs
import random


def gw_freespec_all():
    chain_gws = []
    chain_all = []
    chaindirs = glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_FREESPEC/*ER_run*")
    random.shuffle(chaindirs)
    for chaindir in chaindirs:
        chainfile = chaindir+"/chain_1.txt"
        print("Loading "+chainfile+"... ")
        if not os.path.exists(chaindir+"/burnt_gw_chain.npy"):
            try:
                likelihood = os.popen("cat "+chaindir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
                print(likelihood)
                if not likelihood == "-inf":
                    chain = np.loadtxt(chainfile).squeeze()
                    if len(chain) > 20000:
                        chain_all.append(chain.T)
                        lenchain = len(chain)
                        burnoff = 0.5*lenchain
                        burnt_chain = chain[int(burnoff):,:]
                        
                        burnt_chain_last = burnt_chain[:,-35:]
                        burnt_chain_spec = burnt_chain_last[::50,:30]
                        np.save(chaindir+"/burnt_gw_chain", burnt_chain_spec)
                        #burnt_chain_all.append(burnt_chain_spec.T)
            except:
                print(chainfile+" didn't work")
            continue

    burnt_chain_all = []
    for chaindir in glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_FREESPEC/*ER_run*"):
        try:
            #likelihood = os.popen("cat "+chaindir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
            #print(likelihood)
            #if not likelihood == "-inf":
            b_chain = np.load(chaindir+"/burnt_gw_chain.npy")
            b_chain_trim = b_chain
                #if b_chain.shape[1] > 75000:
                    #chain_gws.append(b_chain[:, int(0.9*len(b_chain[0])):])
                
            burnt_chain_all.append(b_chain_trim)
            print("Collected chain: "+chaindir)
        except:
            print(chaindir+" doesn't have burnt chain") 



    burntchainall = np.vstack(burnt_chain_all)

    pdfs = [ np.histogram(burntchainall[:,bc_i], bins=25, density=True, range=(-9,-4)) for bc_i in range(len(burntchainall.T)) ]

    bins = np.linspace(-9, -4, 25)
    resampled = [ np.random.choice(bins, size=1000, p=pdf[0]/np.sum(pdf[0])) for pdf in pdfs]

    resamp_array = np.array(resampled)

    #resamp_array = burntchainall
    T = 122448047.42001152
    Tyear = 3.8828021125067073704

    f_xaxis = np.linspace(1,30,30)
    freal = f_xaxis/T
    fextend = np.logspace(np.log10(1/(5*T)), np.log10(2e-7), 600)
    
    pwl = np.sqrt((10**-14.29)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(4.3333-3) * fextend**(-4.3333) * freal[0])
    
    fig = plt.figure(figsize=(15,8))
    axes = fig.add_subplot(111)
    df = np.median(np.diff(freal))
    widths = df
    parts = axes.violinplot(resamp_array.T, positions=freal, widths=widths, bw_method="scott", showextrema=False, points=1000)
    for pc in parts["bodies"]:
        pc.set_facecolor("#8FC7B6")
        pc.set_edgecolor("#012319")
        pc.set_alpha(1)
        pc.set_linewidth(2)
    axes.set_xscale("log")
    axes.set_ylim(-9, -6)
    axes.set_xlim(3e-9)
    #sbs.violinplot(burntchainall.T, positions=f_xaxis, ax = axes)
    axes.plot(fextend, np.log10(pwl),linestyle="-", color="#003f5c",lw=3, label = "Factorised likelihood result:\n$\log_{10}A = -14.29; \gamma=4.333$")
    axes.legend(fontsize=15)
    axes.set_xlabel("Frequency (Hz)", fontsize=15)
    axes.set_ylabel(r"$\log_{10}(\rho/\mathrm{s})$", fontsize=15)
    axes.tick_params(axis="both", labelsize=15)

    #fig = corner.corner(chaingw.T,bins=30,labels=["gamma","amp"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84])
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_FREESPEC/CRN_PL_FREESPEC.png")
    fig.clf()
    #np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_GW_updated",chaingw)
    #np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_all_updated",chainall)
    np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_FREESPEC/CRN_ER_freebins_updated",burntchainall)

gw_freespec_all()