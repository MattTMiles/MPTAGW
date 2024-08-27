
import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
from scipy.signal import correlate

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



def chain_corner_from_dir(dirname, parstxt):
    chainfile = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/PTA_RUN/" + dirname + '/chain_1.txt'
    chaindir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/PTA_RUN/" + dirname+"/"
    
    if not os.path.exists(chainfile):
        
        return print("chain_1.txt doesn't exist in this directory")

    print("loading {}".format(chainfile))
    likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")
    if likelihood == "-inf":
        #os.system("rm -rf /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/PTA_RUN/" + dirname)
        print("Removed: "+dirname)
    else:
        lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
        if int(lenchain) > 10000:
            #burn = int(0*int(lenchain))
            burn = 50000
            chain_i = np.loadtxt(chainfile, skiprows =int(burn)).squeeze()
            f = open(chaindir+"/"+parstxt, "r")
            pars = list(f.readlines())
            f.close()
            if not np.any(np.isinf(chain_i)):
                #ind = np.argwhere(['gw_log10_A' in p or "gwb_log10_A" in p for p in pars]).squeeze()
                ind = np.argwhere(["gw_gamma" in p for p in pars]).squeeze()
                thin = acor(chain_i[:, ind])
                print('autocorrelation length = {}'.format(thin))
                thinned_chain = chain_i[::5, :]
                #np.save(chaindir+"/thinned_chain_all", thinned_chain_all)
        

        
        pp = model_utils.PostProcessing(thinned_chain, pars, burn_percentage=0)
        pp.plot_trace()
        plt.savefig(chainfile.replace('.txt', '{}_trace.png'.format(dirname)))
        plt.close()
        print("Made {}".format(chainfile.replace('.txt', '{}_trace.png'.format(dirname))))
        
        chain_gamma = chain_i[:,-7]
        chain_amp = chain_i[:,-6]
        chain_gw = np.vstack([chain_gamma,chain_amp])
        pp = model_utils.PostProcessing(chain_gw[:,::5].T, [pars[-3],pars[-2]], burn_percentage=0)
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
chain_corner_from_dir(sys.argv[1], "pars.txt")
