

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob


def chain_corner_from_dir(pulsar, parstxt):
    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/" + pulsar
    chainfiles = glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/SPGWC/" + pulsar + '/SPGWC_*/chain_1.txt')



    for chainfile in chainfiles:
        if not "WIDE" in chainfile:
            if not os.path.exists(chainfile):
                
                return print("chain_1.txt doesn't exist in this directory")

            print("loading {}".format(chainfile))
            chain_i = np.loadtxt(chainfile).squeeze()
            burnt_chain = chain_i[int(0.8*len(chain_i)):]
            f = open(parstxt, "r")
            pars = list(f.readlines())
            f.close()
        
            chain_dir = chainfile.strip("chain_1.txt")
            pp = model_utils.PostProcessing(burnt_chain, pars, burn_percentage=0)
            pp.plot_trace()
            plt.savefig(chain_dir+"/"+pulsar+"_trace.png")
            plt.close()
            print("Made {}".format(chain_dir+"/"+pulsar+"_trace.png"))


        
    
chain_corner_from_dir(sys.argv[1], sys.argv[1]+"/pars.txt")
