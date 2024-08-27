

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
    pulsar_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/" + pulsar
    chainfiles = glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_ptmcmc/SMBHB_WN/" + pulsar + '/SMBHB_ER*/chain_1.txt')
    
    chain_alls = []
    
    chainfiles = [ cf for cf in chainfiles if "SMBHB_ER" in cf ]
    if not os.path.exists(pulsar_dir+"/SMBHB_ER_WN_corner.png"):
        for chainfile in chainfiles:
            if not "WIDE" in chainfile:
                if not os.path.exists(chainfile):
                    
                    return print("chain_1.txt doesn't exist in this directory")

                print("loading {}".format(chainfile))
                lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
                burn = int(0.2*int(lenchain))
                #burn=1000
                burnt_chain = np.loadtxt(chainfile, skiprows = burn).squeeze()
                f = open(parstxt, "r")
                pars = list(f.readlines())
                f.close()
                ind = np.argwhere(['log10_A_gw' in p for p in pars]).squeeze()
                #thin = acor(burnt_chain[:, ind])
                #burnt_chain = burnt_chain[::5, :]
                chain_dir = chainfile.strip("chain_1.txt")
                chaindir = chain_dir
                # pp = model_utils.PostProcessing(burnt_chain, pars, burn_percentage=0)
                # pp.plot_trace()
                # plt.savefig(chain_dir+"/"+pulsar+"_trace.png")
                # plt.close()
                # print("Made {}".format(chain_dir+"/"+pulsar+"_trace.png"))

        #for chainfile in chainfiles:
            #if not os.path.exists(pulsar_dir+"/thinned_chain_all.npy"):
                likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")
                #chaindir = chainfile.rstrip("/chain_1.txt")
                if likelihood == "-inf":
                    os.system("rm -rf " + chaindir)
                    print("Removed: "+chaindir)
                else:
                    try:
                        np.save(chaindir+"/thinned_chain_all_ER", burnt_chain)
                        chain_alls.append(burnt_chain)
                        #f = open(chaindir+"/pars.txt", "r")
                        #pars = list(f.readlines())
                        #f.close()
                        #lenchain = os.popen("cat "+chainfile+" | wc -l").read().strip("\n")
                        # if int(lenchain) > 1000:
                        #     burn = int(0.5*int(lenchain))
                        #     chain_i = np.loadtxt(chainfile, skiprows =burn).squeeze()
                        #     if not np.any(np.isinf(chain_i)):
                        #         ind = np.argwhere(['log10_A_gw' in p for p in pars]).squeeze()
                        #         thin = acor(chain_i[:, ind])
                        #         print('autocorrelation length = {}'.format(thin))
                        #         thinned_chain_all = chain_i[::thin, :]
                        #         np.save(chaindir+"/thinned_chain_all", thinned_chain_all)

                        #     else:
                        #         os.system("rm -rf " + chaindir)
                        #         print("Removed: "+chaindir)
                    except:
                        print(chaindir+" didn't work")


        # for chaindir in glob.glob(pulsar_dir+"/SMBHB*"):
        #     try:
        #         likelihood = os.popen("cat "+chaindir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
        #         print(likelihood)
        #         if not likelihood == "-inf":
        #             b_chain_all = np.load(chaindir+"/thinned_chain_all.npy")
        #             chain_alls.append(b_chain_all)
        #             print("Collected chain: "+chaindir)
        #     except:
        #         print(chaindir+" doesn't have burnt chain") 

        chainall = np.vstack(chain_alls)
        chainall = chainall[:,:-5]
        np.save(pulsar_dir+"/thinned_chain_combined_ER", chainall)
        
        # ptrim = [ p.lstrip(pulsar+"_").rstrip("\n") for p in pars ]
        # ptrimwn = [ "EFAC" if p == "KAT_MKBF_efac" else "EQUAD" if p == "KAT_MKBF_log10_tnequad" else "ECORR" if p == "KAT_MKBF_log10_ecorr" else p for p in ptrim ]
        # #pfinal = [ p if p != "nmodel" for p in ptrimwn ]
        # ptrimwn.remove("nmodel")
        # fig = corner.corner(chainall,bins=30,labels=list(ptrimwn),quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], smooth=False, smooth1d=False, plot_contours=True, plot_datapoints=False, plot_density=True, levels=[0.393, 0.864, 0.989], fill_contours=False, color="teal", label_kwargs={"fontsize":14}, figsize=(19.20,10.80))
        # fig.savefig(pulsar_dir+"/SMBHB_ER_WN_corner.png")
        # fig.clf()
        # print("Made "+pulsar_dir+"/SMBHB_ER_WN_corner.png")
        
    
chain_corner_from_dir(sys.argv[1], sys.argv[1]+"/pars_ER.txt")
