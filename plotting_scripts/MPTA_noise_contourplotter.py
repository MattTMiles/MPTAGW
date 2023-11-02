import numpy as np
from numpy.random import multivariate_normal

import argparse
import astropy
import os,  os.path, sys
import glob
import chainconsumer
from chainconsumer import ChainConsumer
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

clr = [mcolors.CSS4_COLORS['mediumseagreen'],mcolors.CSS4_COLORS['firebrick'], mcolors.CSS4_COLORS['goldenrod'],mcolors.CSS4_COLORS['steelblue'],mcolors.CSS4_COLORS['palegreen'],mcolors.CSS4_COLORS['palevioletred'],mcolors.CSS4_COLORS['darkorange'],mcolors.CSS4_COLORS['cornflowerblue'],
       mcolors.CSS4_COLORS['salmon'],mcolors.CSS4_COLORS['lavender'],mcolors.CSS4_COLORS['lemonchiffon'],mcolors.CSS4_COLORS['darkcyan'],mcolors.CSS4_COLORS['olive'],mcolors.CSS4_COLORS['slategray'],mcolors.CSS4_COLORS['sienna'],mcolors.CSS4_COLORS['rebeccapurple']]

from astropy import units as u
from astropy.coordinates import SkyCoord
import corner
from matplotlib.lines import Line2D
import bilby
from sklearn.neighbors import KernelDensity

## Fix the plot colour to white
plt.rcParams.update({'axes.facecolor':'white'})

## Create conditions for the user
parser = argparse.ArgumentParser(description="Pulsar pair cross correlation enterprise noise run.")
parser.add_argument("-pulsar_list", dest="pulsar_list", help="Path to list of pulsars to run on", required = False)
parser.add_argument("-noise", type = str.lower, nargs="+",dest="noise", help="Which noise terms are plotted. Takes arguments as '-noise efac red dm' etc.", \
    choices={"efac", "equad", "ecorr", "dm", "chrom", "red", "band", "sw","n_earth"})
parser.add_argument("-outdir", dest="outdir", help="Directory to write out to.", required = False)
parser.add_argument("-scale", dest="scale", help="Scales the posteriors for potentially better viewing.", required = False)
args = parser.parse_args()

pulsar_list = args.pulsar_list

noise = args.noise
outdir = args.outdir
scale = args.scale

pulsar_list = list(open(pulsar_list).readlines())

res_collect = []
res_pars = []

#data_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM_nlive600/"
data_dir = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_active_noise_models/MPTA_PM_PTMCMC/"

working_pulsars = []

# for pulsar in pulsar_list:
#     pulsar = pulsar.strip("\n")
#     try:
#         resfile = glob.glob(data_dir+"/"+pulsar+"*/*json")[0]
#         res = bilby.result.read_in_result(resfile)
#         res_post = res.posterior
#         res_collect.append(res_post)
#         working_pulsars.append(pulsar)
#     except:
#         print("Can't read result file for: "+pulsar)
#         continue

for pulsar in pulsar_list:
    pulsar = pulsar.strip("\n")
    try:
        chainfile = data_dir+"/"+pulsar+"/"+pulsar+"_burnt_chain.npy"
        res_post = np.load(chainfile)
        res_collect.append(res_post)
        pars = list(open(data_dir+"/"+pulsar+"/pars.txt").readlines())
        pars = [ p.strip("\n") for p in pars ]
        res_pars.append(pars)
        working_pulsars.append(pulsar)
    except:
        print("Can't read result file for: "+pulsar)
        continue


total_pars = []
total_chains = []
legend = []
#total_label = pulsar+"_"+"_".join(noise)
#cc = ChainConsumer()
corners = []
#fig, ax = plt.subplots()
line = []

nbins = 50

newxbins2d = np.linspace(-18,-11,nbins*10)
newybins2d = np.linspace(0,7,nbins*10)
j=0
for i, pulsar in enumerate(working_pulsars):
    pulsar = pulsar.strip("\n")
    print(pulsar)
    
    res_pulsar = res_collect[i]
    res_par = res_pars[i]

    parlabs = []

    for n in noise:
        for parlab in list(res_par):
            if n in parlab:
                parlabs.append(parlab)

    labels = []
    pulsar_chains = []
    for parlab in parlabs:
        total_pars.append(parlab)
        total_chains.append(res_pulsar[:,res_par.index(parlab)])
        pulsar_chains.append(res_pulsar[:,res_par.index(parlab)])

    post_labels = [ parname.replace(pulsar+"_", "") for parname in parlabs ]

    post_labels = [ p.replace("red_noise_gamma", r"$\gamma_{\rm Red}$") for p in post_labels]
    post_labels = [ p.replace("red_noise_log10_A", r"$\log_{10} A_{\rm Red}$") for p in post_labels]
    post_labels = [ p.replace("dm_gp_gamma", r"$\gamma_{\rm DM}$") for p in post_labels]
    post_labels = [ p.replace("dm_gp_log10_A", r"$\log_{10} A_{\rm DM}$") for p in post_labels]
    

    if pulsar == "J1909-3744":
        clrc = "red"
        alpha = 1
        zo=10
        legend += [pulsar]
        line += [Line2D([0], [0], label=pulsar, color=clrc)]
    elif pulsar == "J1903-7051":
        clrc = "green"
        alpha = 1
        zo=10
        legend += [pulsar]
        line += [Line2D([0], [0], label=pulsar, color=clrc)]

    else:
        clrc = "grey"
        alpha = 0.25
        zo=11
        #legend += [pulsar]
        #line += [Line2D([0], [0], label=pulsar, color=clrc)]
        #legend = pulsar
    # elif pulsar == "J0437-4715":
    #     clrc = "red"
    #     alpha = 1
    #     zo=10
    #     legend += [pulsar]
    #     line += [Line2D([0], [0], label=pulsar, color=clrc)]
    #     #legend = pulsar
    # elif pulsar == "J1713+0747":
    #     clrc = "orange"
    #     alpha = 1
    #     zo=10
    #     legend += [pulsar]
    #     line += [Line2D([0], [0], label=pulsar, color=clrc)]
    #     #legend = pulsar
    # elif pulsar == "J1643-1224":
    #     clrc = "blue"
    #     alpha = 1
    #     zo=10
    #     legend += [pulsar]
    #     line += [Line2D([0], [0], label=pulsar, color=clrc)]
    #     #legend = pulsar
    # elif pulsar == "J1741+1351":
    #     clrc = "aqua"
    #     alpha = 1
    #     zo=10
    #     legend += [pulsar]
    #     line += [Line2D([0], [0], label=pulsar, color=clrc)]
    #     #legend = pulsar
    # elif pulsar == "J1824-2452A":
    #     clrc = "brown"
    #     alpha = 1
    #     zo=10
    #     legend += [pulsar]
    #     line += [Line2D([0], [0], label=pulsar, color=clrc)]
    #     #legend = pulsar
    # else:
    #     clrc = "grey"
    #     alpha = 0.2
    #     zo=1
    #     #legend += ["PPTA Pulsar"]
    #     #legend = "PPTA Pulsar"
    

    if len(pulsar_chains) > 0:

        pulsar_chain_array = np.vstack(pulsar_chains)
        pulsar_chain_array = pulsar_chain_array[:,::10]
        kdeAMP = KernelDensity(bandwidth="scott", kernel="gaussian")
        kdeAMP.fit(pulsar_chain_array[1,:][:, None])
        logprob = kdeAMP.score_samples(newxbins2d[:, None])
        kde_eval_AMP = np.exp(logprob)

        #kde_eval_DMAMP = pDMA

        kdeGAM = KernelDensity(bandwidth="scott", kernel="gaussian")
        kdeGAM.fit(pulsar_chain_array[0,:][:, None])
        logprob = kdeGAM.score_samples(newybins2d[:, None])
        kde_eval_GAM = np.exp(logprob)

        X, Y = np.meshgrid(newxbins2d, newybins2d)
        bindiffAMP = newxbins2d[1] - newxbins2d[0]
        bindiffGAM = newybins2d[1] - newybins2d[0]

        adjusted_amp = kde_eval_AMP/(np.sum(kde_eval_AMP)*bindiffAMP)
        adjusted_gam = kde_eval_GAM/(np.sum(kde_eval_GAM)*bindiffGAM)

        #amp_sample = np.random.choice(newxbins2d,size=100000, p=adjusted_amp/np.sum(adjusted_amp))
        #gam_sample = np.random.choice(newybins2d,size=100000, p=adjusted_gam/np.sum(adjusted_gam))
        
        amp_sample = np.random.choice(newxbins2d,size=100000, p=kde_eval_AMP/np.sum(kde_eval_AMP))
        gam_sample = np.random.choice(newybins2d,size=100000, p=kde_eval_GAM/np.sum(kde_eval_GAM))

        pulsar_chain_array = np.vstack([gam_sample,amp_sample])

        #corner.hist2d(pulsar_chains[0], pulsar_chains[1], axes=ax, smooth=True, plot_datapoints=False, plot_density=False, no_fill_contours=True,axes_scale="log", levels=[0.68])
        s_fac = 1
        if j==0:
            figure = corner.corner(pulsar_chain_array.T, levels=[0.32], smooth=False, plot_datapoints=False, no_fill_contours=False, color = clrc, plot_density=True, smooth1d=False,density=True, hist2d_kwargs={"density":True})
            j=j+1
        else:
            corner.corner(pulsar_chain_array.T, fig=figure, levels=[0.32],smooth=False, plot_datapoints=False, no_fill_contours=False, color = clrc, plot_density=True, smooth1d=False,density=True, hist2d_kwargs={"density":True})
    #cc.add_chain(n_chain,parameters=post_labels, walkers=np.shape(n_chain)[0],num_eff_data_points=np.shape(n_chain)[0],num_free_params=np.shape(n_chain)[1], posterior=n_chain[:,-1])
    #cc.add_chain(n_chain,parameters=post_labels, walkers=np.shape(n_chain)[0],num_eff_data_points=np.shape(n_chain)[0],num_free_params=np.shape(n_chain)[1], posterior=n_chain[:,-1])
    #cc.plotter.plot(legend=False)

#total_chains = np.array(total_chinas)



post_labels = [r"$\gamma_{\rm Red}$", r"$\log_{10} A_{\rm Red}$"]

line += [Line2D([0], [0], label="MPTA Pulsar", color="grey")]
legend +=["MPTA Pulsar"]
#corner.hist2d(tc[:,0],tc[:,1],axes=ax, smooth=True, plot_datapoints=False, plot_density=False, no_fill_contours=True,contour_kwargs={"colors":"grey", "alpha":0.2},axes_scale="log", levels=[0.68])
# ax.set_xlabel(list(set(post_labels))[0])
# ax.set_ylabel(list(set(post_labels))[1])
# ax.set_ylim(-20,-11)
# ax.set_xlim(0,7)
#figure.legend(line,legend)
#leg = ax.get_legend()
#ax.legend(["trial1", "trial2"])
#plt.show()
figure.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/MPTA_red_noise_PTMCMC_no_KDE.png")




#cc.configure(max_ticks=5, tick_font_size=tfs, colors="grey", label_font_size=lfs, spacing=1.6, diagonal_tick_labels=True, contour_label_font_size=cfs, shade=False,
#                shade_alpha=0.5, linewidths=2.0, summary=True, sigma2d=False, usetex=False, smooth=True, plot_point=False, plot_hists=False, plot_contour=True)

#cc.configure(smooth=True, plot_contour=True)
'''
patches = [Patch(label=pulsar_list[i].strip("\n")) for i in range(len(labels2))]
legend_elements = patches

fig = cc.plotter.plot(figsize=fig_size, legend=False)
fig = cc.plotter.plot(legend=False)
fig.set_size_inches(2* fig.get_size_inches())
fig.align_ylabels()
fig.align_xlabels()
#fig.legend(handles = legend_elements, bbox_to_anchor=(0.8, 0.95), fontsize=18)
#title = plt.suptitle(total_label, fontsize = 20)
fig.tight_layout()
plt.show()

#if outdir:
#    fig.savefig(outdir+"/"+total_label+"_"+"_".join(labels2)+".png")
#else:
#    fig.savefig(total_label+"_"+"_".join(labels2)+".png")
'''
    

    

    


    

