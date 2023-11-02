

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob


def chain_corner_from_dir(pulsar, parstxt):
    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PM_WN/" + pulsar
    chainfiles = glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PM_WN/" + pulsar + '/MPTA_PM_*/chain_1.txt')

    chains = []

    for chainfile in chainfiles:
        if not os.path.exists(chainfile):
            print("chain_1.txt doesn't exist in this directory")
        else:
            print("loading {}".format(chainfile))
            chain_i = np.loadtxt(chainfile).squeeze()
            if len(chain_i) > 1000:
                burnt_chain = chain_i[int(0.8*len(chain_i)):]
                chains.append(burnt_chain)
                f = open(parstxt, "r")
                pars = list(f.readlines())
                f.close()
    
    chain_i = np.vstack(chains)

    pp = model_utils.PostProcessing(chain_i, pars, burn_percentage=0)
    pp.plot_trace()
    plt.savefig(pulsar_dir+"/"+pulsar+"_trace.png")
    plt.close()
    print("Made {}".format(pulsar_dir+"/"+pulsar+"_trace.png"))
    
    
    chain = chain_i[:, :-4]
    indices = np.array([True for p in pars])
    corner_file_label = ''
    chain_corner = chain[:, indices]
    fig = corner.corner(chain_corner, bins=30, labels=pars,
                        quantiles=(0.16, 0.84), title_quantiles=(0.16,0.5,0.84), show_titles=True)
    
    for ax in fig.axes:
        xlab = ax.get_xlabel()
        ylab = ax.get_ylabel()
        ti = ax.get_title()
        ax.set_title(ti, fontsize=9)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
    
    figsavename = pulsar_dir + '/{}_corner'.format(pulsar) + \
        corner_file_label + '.png'
    print(figsavename)
    plt.savefig(figsavename, dpi=300, bbox_inches='tight')
    plt.close()

    np.save(pulsar_dir+"/"+pulsar+"_burnt_chain", chain)
    
chain_corner_from_dir(sys.argv[1], "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PM_WN/" + sys.argv[1]+"/pars.txt")
