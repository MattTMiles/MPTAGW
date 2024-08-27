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

parser = argparse.ArgumentParser(description="Likelihood reweighting master script.")

parser.add_argument("-post_file", dest="crn_chain", help="CRN chain file to use.")
parser.add_argument("-new_pta", dest="new_pta", help="New PTA pickle file to extract likelihoods")
parser.add_argument("-old_pta", dest="old_pta", help="Old PTA pickle file to extract prior")
parser.add_argument("-outdir", dest="outdir", help="Output directory")
args = parser.parse_args()

crn_chain_path = str(args.crn_chain)
new_pta_path = str(args.new_pta)
old_pta_path = str(args.old_pta)
outdir = str(args.outdir)

try:
    os.mkdir(outdir)
except:
    print("Output directory is made")


pta_HD = dill.load(open(new_pta_path,"rb"))
pta_CRN = dill.load(open(old_pta_path, "rb"))

posterior_CRN_file = crn_chain_path
try:
    posterior_CRN = np.load(posterior_CRN_file)
except:
     os.system("touch "+outdir+"/broken_posterior")
     sys.exit(1)
# try:
#     posterior_CRN = np.loadtxt(posterior_CRN_file)
# except ValueError:
#     os.system("touch "+outdir+"/broken_posterior")
#     sys.exit(1)

quickcheck = [ i for i, p in enumerate(pta_CRN.param_names) if p in pta_HD.param_names ]
quickcheck = quickcheck + [-4,-3,-2,-1]

likelihood_CRN = posterior_CRN[:,-3]

idx = [i for i, arr in enumerate(likelihood_CRN) if np.isfinite(arr).all()]

posterior_CRN = posterior_CRN[idx, :]
likelihood_CRN = likelihood_CRN[idx]
lnprior_old = pta_CRN.get_lnprior(posterior_CRN[0,:])

posterior_CRN = posterior_CRN[::5, quickcheck]
likelihood_CRN = likelihood_CRN[::5]
select_range = range(len(likelihood_CRN))

N = len(likelihood_CRN)


lnprior_new = pta_HD.get_lnprior(posterior_CRN[0,:])

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


with tqdm(total=N, position=0, leave=True) as pbar:
    for i in tqdm(range(N), position=0, leave=True):
        Ns = random.choice(select_range)
        if not Ns in Ns_all:
            Ns_all.append(Ns)
            likelihood_HD_i = float(pta_HD.get_lnlikelihood(posterior_CRN[Ns,:]))

            weight_i = np.exp(likelihood_HD_i +lnprior_new - likelihood_CRN[Ns] - lnprior_old)
            #if weight_i > 1e-100:

            partial_CRN_post.append(posterior_CRN[Ns,:])

            likelihood_HD.append(likelihood_HD_i)

            weights.append(weight_i)
            
            pbar.update()

        if len(weights) > 4:
            if len(weights) % 200 == 0:
                
                #weights = np.array(weights)

                Ns_all_save = np.array(Ns_all)
                np.save(outdir+"/Ns_all",Ns_all_save)

                
                likelihood_HD_save = np.array(likelihood_HD)
                likelihood_HD_save = likelihood_HD_save[weights < (np.median(weights) + (10*np.std(weights)))]
                np.save(outdir+"/likelihood_HD",likelihood_HD_save)

                
                weights_save = np.array(weights)
                weights_save = weights_save[weights < (np.median(weights) + (10*np.std(weights)))]
                np.save(outdir+"/weights",weights_save)
                mean_weight = np.mean(weights_save)

                
                partial_CRN_post_save = np.array(partial_CRN_post)
                partial_CRN_post_save = partial_CRN_post_save[weights < (np.median(weights) + (10*np.std(weights)))]
                np.save(outdir+"/partial_CRN_post",partial_CRN_post_save)

                p_CRN_gwCorr = partial_CRN_post_save[:,-5]
                
                sigw = np.std(weights_save)
                n_eff = len(weights_save)/(1+((sigw/mean_weight)**2))
                efficiency = n_eff/len(weights_save)

                plt.hist(p_CRN_gwCorr, bins=50, histtype="stepfilled",label="Old PTA Corr",alpha=0.25,color="C0",lw=2, density=True)
                plt.hist(p_CRN_gwCorr, weights=weights_save, bins=50, histtype="stepfilled",label="New PTA reweighted",alpha=0.25,color="C1",lw=2, density=True)
                plt.legend()
                plt.title(r"Reweighting efficiency: {:.3f}; Ns: {}; $\mu_w$: {:.2f}".format(efficiency, len(weights_save), mean_weight))
                plt.xlabel(r"Correlation")
                plt.ylabel("PDF")
                plt.savefig(outdir+"/GW_Corr_comparison")
                plt.clf()
                gc.collect()

            if len(weights) % 1000 == 0:
                p_CRN_gwAmp = partial_CRN_post_save[:,-6]
                p_CRN_gwCorr = partial_CRN_post_save[:,-5]
                
                #temp_weights = weights/np.max(weights)
                p_HD_gwCorr = np.random.choice(p_CRN_gwCorr,size=len(weights_save), p=weights_save/np.sum(weights_save))
                p_HD_gwAmp = np.random.choice(p_CRN_gwAmp,size=len(weights_save), p=weights_save/np.sum(weights_save))

                GW_HD_post = np.vstack([p_HD_gwCorr, p_HD_gwAmp])
                GW_CRN_post = np.vstack([p_CRN_gwCorr, p_CRN_gwAmp])

                fig = corner.corner(GW_CRN_post.T,bins=30,labels=[r"Correlation",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "CRN",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
                corner.corner(GW_HD_post.T,bins=30,labels=[r"Correlation",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "HD_reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
                clr = ["blue", "orange"]
                labels_legend = ["Old PTA", "New PTA reweighted"]
                patches = [Patch(facecolor=clr[i],label=labels_legend[i], alpha=0.5) for i in range(len(labels_legend))]
                legend_elements = patches
                fig.legend(handles = legend_elements, fontsize=12)
                #fig.legend(["CRN", "HD reweighted"])
                fig.savefig(outdir+"/Corr_new_old_comparison.png")
                fig.clf()
                np.save(outdir+"/HD_post", GW_HD_post)
                gc.collect()


#Final saving and plotting if it gets through all the samples


likelihood_HD_save = np.array(likelihood_HD)
likelihood_HD_save = likelihood_HD_save[weights < (np.median(weights) + (10*np.std(weights)))]
np.save(outdir+"/likelihood_HD",likelihood_HD_save)


weights_save = np.array(weights)
weights_save = weights_save[weights < (np.median(weights) + (10*np.std(weights)))]
np.save(outdir+"/weights",weights_save)
mean_weight = np.mean(weights_save)


partial_CRN_post_save = np.array(partial_CRN_post)
partial_CRN_post_save = partial_CRN_post_save[weights < (np.median(weights) + (10*np.std(weights)))]
np.save(outdir+"/partial_CRN_post",partial_CRN_post_save)

p_CRN_gwAmp = partial_CRN_post_save[:,-6]
p_CRN_gwCorr = partial_CRN_post_save[:,-5]

samplesize = len(weights_save)

p_HD_gwCorr = np.random.choice(p_CRN_gwCorr,size=samplesize, p=weights_save/np.sum(weights_save))
p_HD_gwAmp = np.random.choice(p_CRN_gwAmp,size=samplesize, p=weights_save/np.sum(weights_save))

sigw = np.std(weights_save)
n_eff = len(weights_save)/(1+((sigw/mean_weight)**2))
efficiency = n_eff/len(weights_save)

plt.hist(p_CRN_gwCorr, bins=50, histtype="stepfilled",label="Old PTA Corr",alpha=0.25,color="C0",lw=2, density=True)
plt.hist(p_CRN_gwCorr, weights=weights_save, bins=50, histtype="stepfilled",label="New PTA reweighted",alpha=0.25,color="C1",lw=2, density=True)
plt.legend()
plt.title(r"Reweighting efficiency: {:.3f}; Ns: {}; $\mu_w$: {:.2f}".format(efficiency, len(weights_save), mean_weight))
plt.xlabel(r"Correlation")
plt.ylabel("PDF")
plt.savefig(outdir+"/GW_Corr_comparison")
plt.clf()



GW_HD_post = np.vstack([p_HD_gwCorr, p_HD_gwAmp])
GW_CRN_post = np.vstack([p_CRN_gwCorr, p_CRN_gwAmp])

fig = corner.corner(GW_CRN_post.T,bins=30,labels=[r"Correlation",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "Old PTA",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
corner.corner(GW_HD_post.T,bins=30,labels=[r"Correlation",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "New PTA reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
clr = ["blue", "orange"]
labels_legend = ["Old PTA", "New PTA reweighted"]
patches = [Patch(facecolor=clr[i],label=labels_legend[i], alpha=0.5) for i in range(len(labels_legend))]
legend_elements = patches
fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.savefig(outdir+"/Corr_new_old_comparison.png")
fig.clf()
np.save(outdir+"/HD_post", GW_HD_post)
gc.collect()

os.system("touch "+outdir+"/finished_reweighting")


