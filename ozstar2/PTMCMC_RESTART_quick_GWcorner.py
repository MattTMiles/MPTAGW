

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
    for chaindir in glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_ER_run*"):
        chainfile = chaindir+"/chain_1.txt"
        print("Loading "+chainfile+"... ")
        if not os.path.exists(chaindir+"/burnt_gw_chain.npy"):
            try:
                chain = np.loadtxt(chainfile).squeeze()
                if len(chain) > 1:
                    chain_all.append(chain.T)
                    lenchain = len(chain)
                    burnoff = 0.8*lenchain
                    burnt_chain = chain[int(burnoff):,:]
                    burnt_chain_all.append(burnt_chain.T)
                    chain_gamma = chain[:,-6]
                    chain_amp = chain[:,-5]
                    chain_gw = np.vstack([chain_gamma,chain_amp])
                    #chain_gws.append(chain_gw)
                    np.save(chaindir+"/burnt_gw_chain", chain_gw)
                    gc.collect()

            except:
                print(chainfile+" didn't work")
                continue
        else:
            print("burnt chain already there")

    for chaindir in glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_ER_run*"):
        try:
            b_chain = np.load(chaindir+"/burnt_gw_chain.npy")
            chain_gws.append(b_chain)
            print("Collected chain: "+chaindir)
        except:
            print(chaindir+" doesn't have burnt chain") 

    chaingw = np.hstack(chain_gws)
    chainall = np.hstack(chain_all)
    burntchainall = np.hstack(burnt_chain_all)

    fig = corner.corner(chaingw.T,bins=30,labels=["gamma","amp"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84])
    fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_PL_ER_GW_corner.png")

    np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_ER_GW_updated",chaingw)
    np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_ER_all_updated",chainall)
    np.save("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/RESTART/CRN_ER_burntall_updated",burntchainall)

gw_corner_all()