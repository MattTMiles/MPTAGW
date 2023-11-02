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



for i, rw_dir in enumerate(glob.glob("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_reweighted/*HD*")):
    print(i)
    if i == 0:
        Ns_all = np.load(rw_dir+"/Ns_all.npy")
        HD_all = np.load(rw_dir+"/posterior_HD_RW.npy").T
        CRN_all = np.load(rw_dir+"/partial_CRN_post.npy")
    else:
        Ns_i = np.load(rw_dir+"/Ns_all.npy")
        HD_i = np.load(rw_dir+"/posterior_HD_RW.npy").T
        CRN_i = np.load(rw_dir+"/partial_CRN_post.npy")

        Ns_all = np.concatenate((Ns_all, Ns_i))
        HD_all = np.concatenate((HD_all, HD_i))
        CRN_all = np.concatenate((CRN_all, CRN_i))


#Plotting the combo if it gets through all the samples
nonrepeat = np.unique(Ns_all,return_index=True)[1]

p_CRN_gwamp =  CRN_all[:,-5]
p_HD_gwamp = HD_all[:,-5]

p_CRN_gwamp_unique = p_CRN_gwamp[nonrepeat]
p_HD_gwamp_unique = p_HD_gwamp[nonrepeat]

plt.hist(p_CRN_gwamp, bins=50, histtype="stepfilled",label="CRN",alpha=0.5,color="C0",lw=2, density=True)
plt.hist(p_HD_gwamp, bins=50, histtype="stepfilled",label="HD reweighted",alpha=0.5,color="C1",lw=2, density=True)
plt.legend()
plt.title("Total reweighting")
plt.xlabel(r"$\log_{10} A$")
plt.ylabel("PDF")
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_reweighted/GW_amp_comparison_total")
plt.clf()

plt.hist(p_CRN_gwamp_unique, bins=50, histtype="stepfilled",label="CRN",alpha=0.5,color="C0",lw=2, density=True)
plt.hist(p_HD_gwamp_unique, bins=50, histtype="stepfilled",label="HD reweighted",alpha=0.5,color="C1",lw=2, density=True)
plt.legend()
plt.title("Total reweighting, no repeats")
plt.xlabel(r"$\log_{10} A$")
plt.ylabel("PDF")
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_reweighted/GW_amp_comparison_total_unique")
plt.clf()

p_CRN_gwgamma = CRN_all[:,-6]
p_HD_gwgamma = HD_all[:,-6]

p_CRN_gwgamma_unique = p_CRN_gwgamma[nonrepeat]
p_HD_gwgamma_unique = p_HD_gwgamma[nonrepeat]

GW_HD_post = np.vstack([p_HD_gwgamma, p_HD_gwamp])
GW_CRN_post = np.vstack([p_CRN_gwgamma, p_CRN_gwamp])

GW_HD_post_unique = np.vstack([p_HD_gwgamma_unique, p_HD_gwamp_unique])
GW_CRN_post_unique = np.vstack([p_CRN_gwgamma_unique, p_CRN_gwamp_unique])

fig = corner.corner(GW_CRN_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "CRN",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
corner.corner(GW_HD_post.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "HD_reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
clr = ["blue", "orange"]
labels_legend = ["CRN", "HD reweighted"]
patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
legend_elements = patches
fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_reweighted/GW_HD_CRN_comparison_total.png")
fig.clf()


fig = corner.corner(GW_CRN_post_unique.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "CRN",color="blue", plot_density=True, hist_kwargs={"density":True},density=True)
corner.corner(GW_HD_post_unique.T,bins=30,labels=[r"$\gamma$",r"$\log_{10} A$"],quantiles=(0.16,0.84), show_titles=True,title_quantiles=[0.16,0.5,0.84], label = "HD_reweighted", color="orange", plot_density=True, hist_kwargs={"density":True},density=True, fig=fig)
clr = ["blue", "orange"]
labels_legend = ["CRN", "HD reweighted"]
patches = [Patch(facecolor=clr[i],label=labels_legend[i]) for i in range(len(labels_legend))]
legend_elements = patches
fig.legend(handles = legend_elements, fontsize=12)
#fig.legend(["CRN", "HD reweighted"])
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PTA_RUN/HD_reweighted/GW_HD_CRN_comparison_total_unique.png")
fig.clf()

                        


