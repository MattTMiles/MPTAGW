

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob


def chainmaker(pulsar):
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
                chains.append(chain_i)
    
    chain_i = np.vstack(chains)
    chain = chain_i[:, :-4]

    np.save(pulsar_dir+"/"+pulsar+"_total_chain", chain)
    
chainmaker(sys.argv[1])
