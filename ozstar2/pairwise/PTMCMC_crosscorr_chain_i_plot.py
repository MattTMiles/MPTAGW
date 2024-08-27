

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
from scipy.signal import correlate
import os

def acor(arr):
    arr -= np.mean(arr)
    auto_correlation = correlate(arr, arr, mode='full')
    auto_correlation = auto_correlation[auto_correlation.size//2:]
    auto_correlation /= auto_correlation[0]
    indices = np.where(auto_correlation<0.5)[0]
    if len(indices)>0:
        if indices[0] == 0:
            indices[0] = 1
        return indices[0]
    else:
        return 1


def chain_corner_from_dir(pair, parstxt):
    pulsar_dir = pair
    chainfiles = glob.glob(pair + '/'+pair+'*/chain_1.txt')



    for chainfile in chainfiles:
        if not os.path.exists(chainfile):
            
            return print("chain_1.txt doesn't exist in this directory")

        print("loading {}".format(chainfile))
        chain_i = np.loadtxt(chainfile).squeeze()
        lenchain = len(chain_i)
        if lenchain > 1000:
            burnt_chain = chain_i[int(0.5*lenchain):, :]
            f = open(parstxt, "r")
            pars = list(f.readlines())
            f.close()
            ind = np.argwhere(['gw_bins_log10_A' in p for p in pars]).squeeze()
            thin = acor(burnt_chain[:, ind])
            burnt_chain = burnt_chain[::thin, :]

            chain_dir = chainfile.strip("chain_1.txt")
            pp = model_utils.PostProcessing(burnt_chain, pars, burn_percentage=0)
            pp.plot_trace()
            plt.savefig(chain_dir+"/"+pair+"_trace.png")
            plt.close()
            print("Made {}".format(chain_dir+"/"+pair+"_trace.png"))

            os.system("cp "+parstxt+" "+pulsar_dir+"/")
        
    
chain_corner_from_dir(sys.argv[1], sys.argv[1]+"/"+sys.argv[1]+"_1/pars.txt")
