# This script attempts to reweight the CRN likelihood to discover the HD posterior

from __future__ import division

import os, glob, json, pickle, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
import enterprise.constants as const
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
from enterprise_extensions.blocks import common_red_noise_block
import corner
import multiprocessing
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

sys.path.insert(0, '/home/mmiles/soft/enterprise_warp/')
#sys.path.insert(0, '/fred/oz002/rshannon/enterprise_warp/')
from enterprise_warp import bilby_warp

from matplotlib.patches import Patch
import bilby
import argparse
import time
import faulthandler
import dill
import random
from tqdm import tqdm
import corner
import gc
from scipy.special import logsumexp

output_num = sys.argv[1]
outdir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_noauto_reweighted/HD_noauto_RW_{0}".format(output_num)

try:
    os.mkdir(outdir)
except:
    pass

pta_CRN_file = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_CRN_ER.pkl"
pta_HD_file = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/pta_objects/PTA_PL_NOAUTO_AUTO_ER.pkl"

#pta_CRN = dill.load(open(pta_CRN_file,"rb"))
pta_HD = dill.load(open(pta_HD_file,"rb"))

posterior_CRN_file = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/CRN_ER_all_updated.npy"
posterior_CRN = np.load(posterior_CRN_file)

likelihood_CRN = posterior_CRN[-3,:]

select_range = range(len(likelihood_CRN))

N = len(likelihood_CRN)

try:
    Ns_all = list(np.load(outdir+"/Ns_all.npy"))
except:
    Ns_all = []
    pass

try:
    likelihood_HD = list(np.load(outdir+"/likelihood_HD.npy"))
except:
    likelihood_HD = []
    pass

try:
    weights = list(np.load(outdir+"/weights.npy"))
except:
    weights = []
    pass

try:
    partial_CRN_post = list(np.load(outdir+"/partial_CRN_post.npy"))
except:
    partial_CRN_post = []
    pass

try:
    likelihood_CRN_partial = list(np.load(outdir+"/likelihood_CRN.npy"))
except:
    likelihood_CRN_partial = []
    pass

def marginalise_new_params(post_CRN_Ns, pta_HD):

    cc_amp_prior = np.linspace(-18,-11,10)
    cc_gamma_prior = np.linspace(0,7,10)
    
    post_to_use = post_CRN_Ns[:-4]

    integrand= []
    for i, cc_amp in enumerate(cc_amp_prior):
        for j, cc_gam in enumerate(cc_gamma_prior):
            post_to_calc = np.concatenate((post_to_use,np.array([cc_gam,cc_amp])))
            likelihood_HD = pta_HD.get_lnlikelihood(post_to_calc)
            integrand.append(likelihood_HD)
            print("likelihood calced for {} amp and {} gamma".format(i,j))

    return logsumexp(integrand)


with tqdm(total=N, position=0, leave=True) as pbar:
    for i in tqdm(range(N), position=0, leave=True):
        Ns = random.choice(select_range)
        if not Ns in Ns_all:
            Ns_all.append(Ns)
            partial_CRN_post.append(posterior_CRN[:,Ns])

            likelihood_HD.append(marginalise_new_params(posterior_CRN[:,Ns],pta_HD))
            #likelihood_HD_i = pta_HD.get_lnlikelihood(posterior_CRN[:,Ns])
            #likelihood_HD.append(likelihood_HD_i)
            likelihood_CRN_partial.append(likelihood_CRN[Ns])

            #weight_i = likelihood_HD_i/likelihood_CRN[Ns]
            #weights.append(weight_i)
            
            pbar.update()

        if len(likelihood_HD) > 1:
            if len(likelihood_HD) % 10 == 0:


                weights = likelihood_HD - likelihood_CRN_partial
                p = np.exp(weights)
                p /= np.sum(p)
                
                

                Ns_all_save = np.array(Ns_all)
                np.save(outdir+"/Ns_all",Ns_all_save)

                likelihood_HD_save = np.array(likelihood_HD) 
                np.save(outdir+"/likelihood_HD",likelihood_HD_save)

                weights_save = np.array(weights)
                np.save(outdir+"/weights",weights_save)
                mean_weight = np.mean(weights)

                partial_CRN_post_save = np.array(partial_CRN_post)
                np.save(outdir+"/partial_CRN_post",partial_CRN_post_save)

                posterior_HD_save = (weights*partial_CRN_post_save.T)*mean_weight
                np.save(outdir+"/posterior_HD_RW",posterior_HD_save)

                p_CRN_gwamp =  partial_CRN_post_save[:,-5]
                #p_HD_noauto_gwamp = posterior_HD_save[-5,:]
                
                p_HD_noauto_gwamp = np.random.choice(partial_CRN_post_save[:,-5], size=30000, p=p)

                sigw = np.std(weights)
                n_eff = len(p_HD_noauto_gwamp)/(1+((sigw/mean_weight)**2))
                efficiency = n_eff/len(p_HD_noauto_gwamp)

                plt.hist(p_CRN_gwamp, bins=50, histtype="stepfilled",label="CRN",alpha=0.5,color="C0",lw=2, density=True)
                plt.hist(p_HD_noauto_gwamp, bins=50, histtype="stepfilled",label="HD reweighted",alpha=0.5,color="C1",lw=2, density=True)
                plt.legend()
                plt.title("Reweighting efficiency: {:.3f}; Nsamples: {}".format(efficiency, len(p_HD_gwamp)))
                plt.xlabel(r"$\log_{10} A$")
                plt.ylabel("PDF")
                plt.savefig(outdir+"/GW_amp_comparison")
                plt.clf()
                gc.collect()

            # if len(weights) % 100 == 0:
            #     p_CRN_gwgamma = partial_CRN_post_save[:,-6]
            #     p_HD_gwgamma = posterior_HD_save[-6,:]

            #     GW_HD_post = np.vstack([p_HD_gwgamma, p_HD_gwamp])
            #     GW_CRN_post = np.vstack([p_CRN_gwgamma, p_CRN_gwamp])

            #     fig = corner.corner(GW_CRN_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "CRN",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
            #     corner.corner(GW_HD_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "HD_reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
            #     clr = ["blue", "orange"]
            #     labels_legend = ["CRN", "HD reweighted"]
            #     patches = [Patch(facecolor=clr[i],label=labels_legend[i], alpha=0.5) for i in range(len(labels_legend))]
            #     legend_elements = patches
            #     fig.legend(handles = legend_elements, fontsize=12)
            #     #fig.legend(["CRN", "HD reweighted"])
            #     fig.savefig(outdir+"/GW_HD_CRN_comparison.png")
            #     fig.clf()
            #     gc.collect()


# #Final saving and plotting if it gets through all the samples

# likelihood_HD_save = np.array(likelihood_HD) 
# np.save(outdir+"/likelihood_HD",likelihood_HD_save)

# weights_save = np.array(weights)
# np.save(outdir+"/weights",weights_save)
# mean_weight = np.mean(weights)

# partial_CRN_post_save = np.array(partial_CRN_post)
# np.save(outdir+"/partial_CRN_post",partial_CRN_post_save)

# posterior_HD_save = (weights*partial_CRN_post_save.T)*mean_weight
# np.save(outdir+"/posterior_HD_RW",posterior_HD_save)

# p_CRN_gwamp =  partial_CRN_post_save[:,-5]
# p_HD_gwamp = posterior_HD_save[-5,:]

# sigw = np.std(weights)
# n_eff = len(p_HD_gwamp)/(1+((sigw/mean_weight)**2))
# efficiency = n_eff/len(p_HD_gwamp)

# plt.hist(p_CRN_gwamp, bins=50, histtype="stepfilled",label="CRN",alpha=0.5,color="C0",lw=2, density=True)
# plt.hist(p_HD_gwamp, bins=50, histtype="stepfilled",label="HD reweighted",alpha=0.5,color="C1",lw=2, density=True)
# plt.legend()
# plt.title("Reweighting efficiency: {:.3f}; Nsamples: {}".format(efficiency, len(p_HD_gwamp)))
# plt.xlabel(r"$\log_{10} A$")
# plt.ylabel("PDF")
# plt.savefig(outdir+"/GW_amp_comparison")
# plt.clf()

# p_CRN_gwgamma = partial_CRN_post_save[:,-6]
# p_HD_gwgamma = posterior_HD_save[-6,:]

# GW_HD_post = np.vstack([p_HD_gwgamma, p_HD_gwamp])
# GW_CRN_post = np.vstack([p_CRN_gwgamma, p_CRN_gwamp])

# fig = corner.corner(GW_CRN_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "CRN",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
# corner.corner(GW_HD_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "HD_reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
# clr = ["blue", "orange"]
# labels_legend = ["CRN", "HD reweighted"]
# patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
# legend_elements = patches
# fig.legend(handles = legend_elements, fontsize=12)
# #fig.legend(["CRN", "HD reweighted"])
# fig.savefig(outdir+"/GW_HD_CRN_comparison.png")
# fig.clf()




                        


