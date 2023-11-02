import os
import json
import sys
import numpy as np
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt
import corner
import glob
import gc


def gw_corner_all():
    chain_gws = []
    chain_all = []
    burnt_chain_all = []
    for chaindir in glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_MISSPEC_RED_DM/CRN_ER_run*"):
        chainfile = chaindir+"/chain_1.txt"
        print("Loading "+chainfile+"... ")
        if not os.path.exists(chaindir+"/burnt_gw_chain.npy"):
            try:
                lenchain = os.popen("cat "+chaindir+"/chain_1.txt | wc -l").read().strip("\n")
                lenchain = int(lenchain)
                if lenchain > 10000:
                    #chain_all.append(chain.T)
                    #lenchain = len(chain)
                    burnoff = 0.6*lenchain
                    burnt_chain = np.loadtxt(chainfile, skiprows=int(burnoff)).squeeze()
                    #burnt_chain = chain[int(burnoff):,:]
                    burnt_chain_all.append(burnt_chain.T)
                    chain_gamma = burnt_chain[:,-7]
                    chain_amp = burnt_chain[:,-6]
                    chain_gw = np.vstack([chain_gamma,chain_amp])
                    #chain_gws.append(chain_gw)
                    np.save(chaindir+"/burnt_gw_chain", chain_gw)
                    gc.collect()

            except:
                print(chainfile+" didn't work")
                continue
        else:
            print("burnt chain already there")

    for chaindir in glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_MISSPEC_RED_DM/CRN_ER_run*"):
        try:
            likelihood = os.popen("cat "+chaindir+"/chain_1.txt | awk 'END{print $(NF-2)}'").read().strip("\n")
            print(likelihood)
            if not likelihood == "-inf":
                b_chain = np.load(chaindir+"/burnt_gw_chain.npy")
                #if b_chain.shape[1] > 75000:
                    #chain_gws.append(b_chain[:, int(0.9*len(b_chain[0])):])
                
                chain_gws.append(b_chain)
                print("Collected chain: "+chaindir)
        except:
            print(chaindir+" doesn't have burnt chain") 

    chaingw = np.hstack(chain_gws)
    #chainall = np.hstack(chain_all)
    #burntchainall = np.hstack(burnt_chain_all)

    fig = corner.corner(chaingw.T,bins=30,labels=["$\gamma_\mathrm{CRN}$","$\log_{10}\mathrm{A_{CRN}}$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], smooth=True, smooth1d=True, plot_contours=True, plot_datapoints=False, plot_density=True, levels=[0.393, 0.864, 0.989], fill_contours=False, color="teal", label_kwargs={"fontsize":14}, figsize=(19.20,10.80))
    corner.overplot_lines(fig, xs=(13/3, None), color="grey", linestyle="--")
    sgwb_line = plt.plot(0,1, linestyle = "--", color="grey") 
    fig.legend(sgwb_line, [r"$\gamma_\mathrm{SGWB}=13/3$"])
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_MISSPEC_RED_DM/MISSPEC_CRN_PL_ER_GW_trimmed_corner_3sigma.png")
    fig.clf()

    np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HYPER_MISSPEC_RED_DM/CRN_ER_GW_updated",chaingw)

gw_corner_all()