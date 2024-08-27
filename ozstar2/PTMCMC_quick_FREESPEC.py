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


def gw_freespec_all():
    chain_gws = []
    chain_all = []
    burnt_chain_all = []
    for chaindir in sorted(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/PTA_FREESPEC/*ER_run*"))[:2]:
        chainfile = chaindir+"/chain_1.txt"
        print("Loading "+chainfile+"... ")
        try:
            chain = np.loadtxt(chainfile, skiprows=90000).squeeze()
            if len(chain) > 1:
                chain_all.append(chain.T)
                lenchain = len(chain)
                # burnoff = 0.9*lenchain
                # burnt_chain = chain[-2000:,:]
                
                burnt_chain_last = chain[::10,-34:]
                burnt_chain_spec = burnt_chain_last[:,:30]
                
                burnt_chain_all.append(burnt_chain_spec.T)
        except:
            print(chainfile+" didn't work")
            continue


    #chaingw = np.hstack(chain_gws)
    #chainall = np.hstack(chain_all)
    burntchainall = np.hstack(burnt_chain_all)

    T = 140660000
    Tyear = 3.8828021125067073704

    f_xaxis = np.linspace(1,30,30)
    freal = f_xaxis/T

    pwl = np.sqrt((10**-14.18)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(3.41-3) * freal**(-3.41) * freal[0])
    
    df = np.median(np.diff(freal))

    fig = plt.figure(figsize=(10,5))
    axes = fig.add_subplot(111)
    #axes.violinplot(burntchainall.T, positions=f_xaxis)
    sbs.violinplot(burntchainall.T, positions=f_xaxis, ax = axes)
    axes.plot(f_xaxis, np.log10(pwl),linestyle="-", color="black",label = "CRN MAP Values:\n$\log_{10}A = -14.18; \gamma=3.41$")


    #fig = corner.corner(chaingw.T,bins=30,labels=["gamma","amp"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84])
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/PTA_FREESPEC/CRN_PL_FREESPEC.png")

    #np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_GW_updated",chaingw)
    #np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_all_updated",chainall)
    #np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_burntall_updated",burntchainall)

gw_freespec_all()