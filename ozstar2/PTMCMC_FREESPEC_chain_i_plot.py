

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob

import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
import gc
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


def chain_corner_from_dir(pulsar, parstxt):
    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/FREESPEC/" + pulsar
    chainfiles = glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/FREESPEC/" + pulsar + '/FREESPEC*/chain_1.txt')


    if not os.path.exists(pulsar_dir+"/FREESPEC.png"):
        for chainfile in chainfiles:
            if not "WIDE" in chainfile:
                if not os.path.exists(chainfile):
                    
                    return print("chain_1.txt doesn't exist in this directory")

                print("loading {}".format(chainfile))
                lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
                burn = int(0.5*int(lenchain))
                burnt_chain = np.loadtxt(chainfile, skiprows = burn).squeeze()
                f = open(parstxt, "r")
                pars = list(f.readlines())
                f.close()
            
                chain_dir = chainfile.strip("chain_1.txt")
                pp = model_utils.PostProcessing(burnt_chain, pars, burn_percentage=0)
                pp.plot_trace()
                plt.savefig(chain_dir+"/"+pulsar+"_trace.png")
                plt.close()
                print("Made {}".format(chain_dir+"/"+pulsar+"_trace.png"))

        #for chainfile in chainfiles:
            #if not os.path.exists(pulsar_dir+"/thinned_chain_all.npy"):
                likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")
                chaindir = chainfile.rstrip("/chain_1.txt")
                if likelihood == "-inf":
                    os.system("rm -rf " + chaindir)
                    print("Removed: "+chaindir)
                else:
                    try:
                        f = open(chaindir+"/pars.txt", "r")
                        pars = list(f.readlines())
                        f.close()
                        lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
                        if int(lenchain) > 1000:
                            burn = int(0.5*int(lenchain))
                            chain_i = np.loadtxt(chainfile, skiprows =burn).squeeze()
                            if not np.any(np.isinf(chain_i)):
                                ind = np.argwhere(['gw_log10_rho_0' in p for p in pars]).squeeze()
                                thin = acor(chain_i[:, ind])
                                print('autocorrelation length = {}'.format(thin))
                                thinned_chain_all = chain_i[::thin, :]
                                np.save(chaindir+"/thinned_chain_all", thinned_chain_all)

                            else:
                                os.system("rm -rf " + chaindir)
                                print("Removed: "+chaindir)
                    except:
                        print(chaindir+" didn't work")

        chain_alls = []
        for chaindir in glob.glob(pulsar_dir+"/FREESPEC*"):
            try:
                likelihood = os.popen("cat "+chaindir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
                print(likelihood)
                if not likelihood == "-inf":
                    b_chain_all = np.load(chaindir+"/thinned_chain_all.npy")
                    chain_alls.append(b_chain_all)
                    print("Collected chain: "+chaindir)
            except:
                print(chaindir+" doesn't have burnt chain") 

        chainall = np.vstack(chain_alls)
        chainall = chainall[:,:-5]
        np.save(pulsar_dir+"/thinned_chain_combined", chainall)
        
        T = 140723541.0264778
        Tyear = 3.8828021125067073704

        f_xaxis = np.linspace(1,30,30)
        freal = f_xaxis/T
        fextend = np.logspace(np.log10(1/(5*T)), np.log10(2e-7), 600)
        
        pwl = np.sqrt((10**-14.25)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(4.3333-3) * fextend**(-4.3333) * freal[0])
        pwl2 = np.sqrt((10**-14.25)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(2.75-3) * fextend**(-2.75) * freal[0])
        pwl3 = np.sqrt((10**-14)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(2.75-3) * fextend**(-2.75) * freal[0])
        fig = plt.figure(figsize=(15,8))
        axes = fig.add_subplot(111)
        df = np.median(np.diff(freal))
        widths = df
        parts = axes.violinplot(chainall[:,-30:], positions=freal, widths=widths, bw_method="scott", showextrema=False, points=1000)
        for pc in parts["bodies"]:
            pc.set_facecolor("#8FC7B6")
            pc.set_edgecolor("#012319")
            pc.set_alpha(1)
            pc.set_linewidth(2)
        axes.set_xscale("log")
        axes.set_ylim(-9)
        axes.set_xlim(3e-9)
        #sbs.violinplot(burntchainall.T, positions=f_xaxis, ax = axes)
        axes.plot(fextend, np.log10(pwl),linestyle="-", color="#003f5c",lw=3, label = "$\log_{10}A = -14.25; \gamma=4.333$")
        axes.plot(fextend, np.log10(pwl2),linestyle="-", color="red",lw=3, label = "$\log_{10}A = -14.25; \gamma=2.75$")
        axes.plot(fextend, np.log10(pwl3),linestyle="-", color="tab:orange",lw=3, label = "$\log_{10}A = -14; \gamma=2.75$")

        axes.legend(fontsize=15)
        axes.set_xlabel("Frequency (Hz)", fontsize=15)
        axes.set_ylabel(r"$\log_{10}(\rho/\mathrm{s})$", fontsize=15)
        axes.tick_params(axis="both", labelsize=15)
        fig.savefig(pulsar_dir+"/FREESPEC.png")
        
    
chain_corner_from_dir(sys.argv[1], sys.argv[1]+"/pars.txt")
