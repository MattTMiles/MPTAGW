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
    pair_dir = pair
    
    try:
        quickcheck = np.load(pair + '/' + pair + '_burnt_chain.npy')
        if len(quickcheck) > 20000:
            return print(pair+" has enough samples")
        else:
            os.system('rm '+pair + '/' + pair + '_burnt_chain.npy')
    except:
        print("No burnt chain yet")
    
    chainfiles = glob.glob(pair + '/'+pair+'_*/chain_1.txt')

    os.system("touch "+pair_dir+"/happening_now")

    chains = []
    
    parstxt = glob.glob(pair + '/'+pair+'_*/pars.txt')[0]
    
    for chainfile in chainfiles:
        chaindir = chainfile.rstrip("/chain_1.txt")
        f = open(parstxt, "r")
        pars = list(f.readlines())
        f.close()
        pars = [ p for p in pars if "nmodel" not in p ]
        if not os.path.exists(chainfile):
            print("chain_1.txt doesn't exist in this directory")
        else:
            likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")
            if likelihood == "-inf":
                os.system("rm -rf " + chaindir)

                print("Removed: "+chaindir)
            else:
                try:
                    print("loading {}".format(chainfile))
                    chain_i = np.loadtxt(chainfile).squeeze()
                    if len(chain_i) > 8000:
                        print("{}".format(len(chain_i)))
                        lenchain = len(chain_i)
                        burnt_chain = chain_i[int(0.5*lenchain):, :]
                        ind = np.argwhere(['gw_bins_log10_A' in p for p in pars]).squeeze()
                        thin = acor(burnt_chain[:, ind])
                        burnt_chain = burnt_chain[::thin, :]
                        burnt_chain = burnt_chain[:, :len(pars)]
                        chains.append(burnt_chain)
                except ValueError:
                    print("Chain stalled")
                    

    try:
        chain_i = np.vstack(chains)

        pp = model_utils.PostProcessing(chain_i, pars, burn_percentage=0)
        pp.plot_trace()
        plt.savefig(pair_dir+"/"+pair+"_trace.png")
        plt.close()
        print("Made {}".format(pair_dir+"/"+pair+"_trace.png"))
        
        
        chain = chain_i
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
        
        figsavename = pair_dir + '/{}_corner'.format(pair) + \
            corner_file_label + '.png'
        print(figsavename)
        plt.savefig(figsavename, dpi=300, bbox_inches='tight')
        plt.close()

        np.save(pair_dir+"/"+pair+"_burnt_chain", chain)
    except:
        print("Data not there for: "+pair)

    os.system("rm "+pair_dir+"/happening_now")
    
chain_corner_from_dir(sys.argv[1], sys.argv[1]+"/"+sys.argv[1]+"_1/pars.txt")
