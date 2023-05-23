#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:53:00 2019

@author: dreardon

Runs basic white, red, and DM noise model for all pulsars in datadir
"""

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner


def chain_corner_from_dir(dirname, parstxt):
    chainfile = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/" + dirname + '/chain_1.0.txt'

    if not os.path.exists(chainfile):
        
        return print("chain_1.0.txt doesn't exist in this directory")

    print("loading {}".format(chainfile))
    chain_i = np.loadtxt(chainfile).squeeze()
    f = open(parstxt, "r")
    pars = list(f.readlines())
    f.close()
    '''
    pp = model_utils.PostProcessing(chain_i, pars)
    pp.plot_trace()
    plt.savefig(chainfile.replace('.txt', '{}_trace.png'.format(dirname)))
    plt.close()
    print("Made {}".format(chainfile.replace('.txt', '{}_trace.png'.format(dirname))))
    '''
    chain_gamma = chain_i[:,-6]
    chain_amp = chain_i[:,-5]
    chain_gw = np.vstack([chain_gamma,chain_amp])
    pp = model_utils.PostProcessing(chain_gw.T, [pars[-2],pars[-1]], burn_percentage=0)
    pp.plot_trace()
    plt.savefig(chainfile.replace('.txt', '{}_GW_trace.png'.format(dirname)))
    plt.close()
    print("Made {}".format(chainfile.replace('.txt', '{}_GW_trace.png'.format(dirname))))


    '''
    chain = chain_i[:, :-4]
    indices = np.array([True for p in pars])
    corner_file_label = ''
    chain_corner = chain[:, indices]
    fig = corner.corner(chain_corner, bins=30, labels=pars,
                        quantiles=(0.16, 0.84), show_titles=True)
    
    for ax in fig.axes:
        xlab = ax.get_xlabel()
        ylab = ax.get_ylabel()
        ti = ax.get_title()
        ax.set_title(ti, fontsize=9)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
    
    figsavename = dirname + '/{}_corner'.format(dirname) + \
        corner_file_label + '.png'
    print(figsavename)
    plt.savefig(figsavename, dpi=300, bbox_inches='tight')
    plt.close()
    '''
chain_corner_from_dir(sys.argv[1], "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/" + sys.argv[1]+"/pars.txt")

