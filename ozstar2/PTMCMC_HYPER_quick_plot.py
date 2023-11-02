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
import glob


def chain_corner_from_dir(dirname, parstxt):
    chainfile = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER/" + dirname + '/chain_1.txt'
    chaindir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER/" + dirname+"/"
    
    if not os.path.exists(chainfile):
        
        return print("chain_1.txt doesn't exist in this directory")

    print("loading {}".format(chainfile))
    likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")
    if likelihood == "-inf":
        os.system("rm -rf /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER/" + dirname)
        print("Removed: "+dirname)
    else:
        lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
        burn = int(0.5*int(lenchain))
        chain_i = np.loadtxt(chainfile, skiprows =burn).squeeze()
        thinned_chain = chain_i[::10,:]
        np.save(chaindir+"/thinned_chain_all", thinned_chain)
        f = open(parstxt, "r")
        pars = list(f.readlines())
        f.close()
        
        # pp = model_utils.PostProcessing(chain_i, pars, burn_percentage=0)
        # pp.plot_trace()
        # plt.savefig(chainfile.replace('.txt', '{}_trace.png'.format(dirname)))
        # plt.close()
        # print("Made {}".format(chainfile.replace('.txt', '{}_trace.png'.format(dirname))))
        
        chain_gamma = chain_i[:,-7]
        chain_amp = chain_i[:,-6]
        chain_gw = np.vstack([chain_gamma,chain_amp])
        pp = model_utils.PostProcessing(chain_gw[:,::10].T, [pars[-2],pars[-3]], burn_percentage=0.6)
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
chain_corner_from_dir(sys.argv[1], "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER/" + sys.argv[1]+"/pars.txt")
