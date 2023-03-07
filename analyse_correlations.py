#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:01:00 2023

@author: dreardon
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from correlations_utils import get_psrnames, get_pairs, hd_orf, dipole, \
    plot_violin, anis_orf, anis_basis

"""
Define Matplotlib settings
"""
from matplotlib import rc
import matplotlib
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams.update({'font.size': 14})

figsize=(2.0*3.3,1.5*3.3)


"""
Load pulsar names, pair names, positions and correlation data
"""
psrnames = get_psrnames()
pairs = get_pairs(psrnames)
datadir = '/Users/dreardon/Desktop/ppta_correlations/corr_chains'
with open('positions.json', 'r') as f:
    positions = json.load(f)
scrambles = np.load('scrambles_ppta.npz')

corr_hd_all = {}  # fixed amplitude
corr_hd_all_imp = {}  # importance sampled
from scipy.interpolate import interp1d
y_flat = np.linspace(-0.99, 0.99, 256)
for i in range(1, 436): # 435 pulsar pairs
    pair = pairs[str(1 + (int(i) - 1) % 435)]
    pdf_1d_imp = np.load("corr_chains/{}_corr_hd_importance_more1713.npz".format(i))
    pdf_1d = np.load("corr_chains/{}_corr_hd_importance_fixA_0.1.npz".format(i))
    x_data_imp = pdf_1d_imp["arr_1"].squeeze()
    y_data_imp = pdf_1d_imp["arr_0"].squeeze()
    x_data = pdf_1d["arr_1"].squeeze()
    y_data = pdf_1d["arr_0"].squeeze()

    f = interp1d(x_data, y_data, kind='linear', assume_sorted=True)
    corr_hd_all['_'.join(pair)] = f(y_flat)
    f = interp1d(x_data_imp, y_data_imp, kind='linear', assume_sorted=True)
    corr_hd_all_imp['_'.join(pair)] = f(y_flat)


# Define sky separations
nseps = 8
vals = np.linspace(0, 180, nseps+1)
halfdiff = np.mean(np.diff(vals))/2
vals = (vals - halfdiff)[1:]
seps = {}
for ibin in range(1, nseps+1):
    seps["bin_{}".format(ibin)] = vals[ibin-1]


# plot ORF
pos = positions
orf = []
orf_aniso = []
angseps = []
for i in range(1, 436): # 435 pulsar pairs
    pair = pairs[str(1 + (int(i) - 1) % 435)]
    psr1 = pair[0]
    psr2 = pair[1]
    pos1 = pos[psr1]
    pos2 = pos[psr2]
    angsep = np.arccos(np.dot(pos1, pos2)) * 180/np.pi
    orf_val = hd_orf(np.array([angsep]))[0]

    psrs_pos = np.array([pos[k] for k in pos.keys()]).squeeze()
    # Location of the pulsars [phi, theta]
    psr_locs = []
    for p in psrs_pos:
        x=p[0]
        y=p[1]
        z=p[2]
        phi = np.arctan2(y, x)
        theta = np.arccos(z)

        # calculate backwards to check:
        if abs(x - np.cos(phi)*np.sin(theta)) > 1e-10:
            raise ValueError("x incorrect")
        if abs(y - np.sin(phi)*np.sin(theta)) > 1e-10:
            raise ValueError("y incorrect")
        if abs(z - np.cos(theta)) > 1e-10:
            raise ValueError("z incorrect")
        psr_locs.append([phi, theta])

    psr_locs = np.array(psr_locs).squeeze()

    lmax = 2  # number of spherical modes. l is 180/theta, where theta = sqrt(\Delta\Omega), and \Delta\Omega =
    # clm are the weights, requiring a physical prior
    basis = anis_basis(psr_locs, lmax, nside=4)
    np.random.seed(1)
    params = np.random.uniform(low=-5, high=5, size=(lmax + 1) ** 2 - 1 )
    #params = [-1.96, -0.16, -0.5]
    orf_aniso_val = anis_orf(pos1, pos2, params,
                              anis_basis=basis, psrs_pos=psrs_pos, lmax=lmax)

    orf.append(orf_val)
    angseps.append(angsep)
    orf_aniso.append(orf_aniso_val)

plt.scatter(angseps, orf)
plt.scatter(angseps, orf_aniso)

plt.show()

import sys
sys.exit()

"""
Loop through Nscramble sky scrambles
"""
Nscramble = 1
plot = True
tag = 'with1713'
ts_scrambles = []
ts_scrambles_curn = []
corr_original = {}
for ns in range(0, Nscramble):

    npos = len(psrnames)
    new_pos = {}
    for nump, psr in enumerate(psrnames):
        # get new sky position
        #ra = np.random.uniform(low=0, high=2*np.pi)
        #dec = np.arccos(np.random.uniform(low=-1, high=1)) - np.pi/2
        #new_pos[psr] = np.array(SkyCoord(ra=ra*u.rad, dec=dec*u.rad).cartesian.xyz)
        theta = scrambles['thetas'][ns, nump]
        phi = scrambles['phis'][ns, nump]
        new_pos[psr] = np.array([np.cos(phi)*np.sin(theta),
                                 np.sin(phi)*np.sin(theta),
                                 np.cos(theta)])


    # Use original data if not scrambling
    pos = positions if Nscramble == 1 else new_pos

    # Look through pairs and make plots
    kde = {}
    pdf = {}
    ptot = {}

    orf_bins_total = {}
    orf_bins_total_imp = {}
    likelihood_hd = {}
    likelihood_hd_imp = {}
    likelihood_curn = {}
    likelihood_mono = {}
    likelihood_curn_imp = {}
    likelihood_null = {}
    likelihood_hd_global = 0
    likelihood_hd_global_imp = 0
    likelihood_mono_global = 0
    likelihood_dipole_global = 0
    likelihood_curn_global = 0
    likelihood_curn_global_imp = 0
    likelihood_null_global = 0
    bf_hd = {}
    bf_hd_imp = {}
    n_bins = {}
    angsep_array = {}
    orf_val_array = {}
    n_tot = 0

    # y_flat = np.linspace(-1, 1, 256)
    null_prob = np.ones(np.shape(y_flat))
    null_prob /= np.sum(null_prob)
    null_prob /= np.mean(np.diff(y_flat))

    """
    Loop through all pulsar pairs
    """
    numerator = 0
    denominator_sum1 = 0
    denominator_sum2 = 0

    for i in range(1, 436): # 435 pulsar pairs

        pair = pairs[str(1 + (int(i) - 1) % 435)]

        psr1 = pair[0]
        psr2 = pair[1]

        #if '1713' in psr1 or '1713' in psr2:
        #    tag = 'no1713'
        #    continue
        #if '1744' in psr1 or '1744' in psr2:
        #    continue
        #if '1603' in psr1 or '1603' in psr2:
        #    continue
        #if '1713' in psr1 or '1713' in psr2:
        #    continue
        #if '1600' in psr1 or '1600' in psr2:
        #    continue

        corr_hd = np.array(corr_hd_all['_'.join(pair)])  # bw=0.1, refecting correlations but not amplitude. Slice taken at -14.69
        corr_hd_imp = np.array(corr_hd_all_imp['_'.join(pair)])
        # y_flat = np.linspace(-1, 1, 256)

        # calculate angular separation and ORF values
        pos1 = pos[psr1]
        pos2 = pos[psr2]
        angsep = np.arccos(np.dot(pos1, pos2)) * 180/np.pi
        orf_val = hd_orf(np.array([angsep]))[0]

        angsep_array['_'.join(pair)] = angsep
        orf_val_array['_'.join(pair)] = orf_val

        dipole_val = dipole(np.array([angsep]))[0]

        # Append to likelihoods for psr1
        try:
            likelihood_hd[psr1] += np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_mono[psr1] += np.log(corr_hd[np.argmin(np.abs(y_flat - 1))])
            likelihood_curn[psr1] += np.log(corr_hd[np.argmin(np.abs(y_flat))])
            likelihood_null[psr1] += np.log(null_prob[0])

            likelihood_hd_imp[psr1] += np.log(corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_curn_imp[psr1] += np.log(corr_hd_imp[np.argmin(np.abs(y_flat))])
        except KeyError:
            likelihood_hd[psr1] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))]))
            likelihood_mono[psr1] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat - 1))]))
            likelihood_curn[psr1] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat))]))
            likelihood_null[psr1] = np.log(null_prob[0])

            likelihood_hd_imp[psr1] = np.copy(np.log(corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))]))
            likelihood_curn_imp[psr1] = np.copy(np.log(corr_hd_imp[np.argmin(np.abs(y_flat))]))

        # Append to likelihoods for psr2
        try:
            likelihood_hd[psr2] += np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_mono[psr2] += np.log(corr_hd[np.argmin(np.abs(y_flat - 1))])
            likelihood_curn[psr2] += np.log(corr_hd[np.argmin(np.abs(y_flat))])
            likelihood_null[psr2] += np.log(null_prob[0])

            likelihood_hd_imp[psr2] += np.log(corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_curn_imp[psr2] += np.log(corr_hd_imp[np.argmin(np.abs(y_flat))])
        except KeyError:
            likelihood_hd[psr2] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))]))
            likelihood_mono[psr2] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat - 1))]))
            likelihood_curn[psr2] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat))]))
            likelihood_null[psr2] = np.log(null_prob[0])

            likelihood_hd_imp[psr2] = np.copy(np.log(corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))]))
            likelihood_curn_imp[psr2] = np.copy(np.log(corr_hd_imp[np.argmin(np.abs(y_flat))]))

        ibin = np.argmin(np.abs(vals - angsep)) + 1
        try:
            orf_bins_total["bin_{}".format(ibin)] *= corr_hd
            orf_bins_total_imp["bin_{}".format(ibin)] *= corr_hd_imp
            n_bins["bin_{}".format(ibin)] +=1
        except KeyError:
            orf_bins_total["bin_{}".format(ibin)] = corr_hd
            orf_bins_total_imp["bin_{}".format(ibin)] = corr_hd_imp
            n_bins["bin_{}".format(ibin)] = 1

        try:
            likelihood_hd_global += np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_curn_global += np.log(corr_hd[np.argmin(np.abs(y_flat))])
            likelihood_mono_global += np.log(corr_hd[np.argmin(np.abs(y_flat - 1))])
            likelihood_dipole_global += np.log(corr_hd[np.argmin(np.abs(y_flat - dipole_val))])
            likelihood_null_global += np.log(null_prob[0])

            likelihood_hd_global_imp += np.log(corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))])
            likelihood_curn_global_imp += np.log(corr_hd_imp[np.argmin(np.abs(y_flat))])
        except KeyError:
            likelihood_hd_global = corr_hd[np.argmin(np.abs(y_flat - orf_val))]
            likelihood_curn_global = corr_hd[np.argmin(np.abs(y_flat))]
            likelihood_mono_global = corr_hd[np.argmin(np.abs(y_flat - 1))]
            likelihood_dipole_global = corr_hd[np.argmin(np.abs(y_flat - dipole_val))]
            likelihood_null_global = np.log(null_prob[0])

            likelihood_hd_global_imp = corr_hd_imp[np.argmin(np.abs(y_flat - orf_val))]
            likelihood_curn_global_imp = corr_hd_imp[np.argmin(np.abs(y_flat))]

    if plot:
        figsize2=(2.4*3.3,1.5*3.3)
        fig, ax1 = plt.subplots(1, 1, figsize=figsize2)
        ax2 = ax1.twinx()

        values = []
        for bini in range(1, nseps+1):
            #values.append(n_bins['bin_{}'.format(bini)])
            try:
                values.append(n_bins['bin_{}'.format(bini)])
            except KeyError:
                n_bins['bin_{}'.format(bini)] = 0
                values.append(n_bins['bin_{}'.format(bini)])
        edges = np.append(np.array(list(seps.values()) - halfdiff), 180)

        ax2.fill_between(np.insert(np.array(list(seps.values()) + halfdiff),0, 0),
                         np.insert(np.array(values),0,values[0]),
                         y2=-np.ones(len(values)+1),
                         step="pre", alpha=0.15, color='k')

        for k in orf_bins_total.keys():
            orf_bins_total[k] /= np.sum(orf_bins_total[k])
            orf_bins_total[k] /= np.mean(np.diff(y_flat))
            # np.save('/Users/dreardon/Desktop/ppta_correlations/{}.npy'.format(k), orf_bins_total[k]/np.sum(orf_bins_total[k]))
            #draws = np.random.choice(y_flat, p=orf_bins_total_imp[k]/np.sum(orf_bins_total_imp[k]), size=100000)
            #ax = plot_violin(ax1, seps[k], draws, width=np.mean(np.diff(vals)), colour='magenta', alpha=1)

            draws = np.random.choice(y_flat, p=orf_bins_total[k]/np.sum(orf_bins_total[k]), size=100000)
            ax = plot_violin(ax1, seps[k], draws, width=np.mean(np.diff(vals)), alpha=0.9)


            ibin = int(k.split('_')[-1])

        theta = np.linspace(0, 180, 1000)
        orf = hd_orf(theta)
        ax1.plot(theta, orf, color='k', linewidth=2)
        #for p in orf_val_array.keys():
        #    if '0437' in p:
        #        ax1.scatter(angsep_array[p], orf_val_array[p],
        #                    c='darkorange', marker='x', zorder=10)
        ax1.set_ylim([-1, 1])
        ax2.set_ylim([0, round(max(n_bins.values())+10, -1)])
        ax2.set_ylabel('Number of pulsar pairs')
        plt.xlim([0, 180])
        plt.tight_layout()
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.set_frame_on(False)
        plt.savefig('corr_plots/corr_total_{}.png'.format(tag))
        plt.savefig('corr_plots/corr_total_{}.pdf'.format(tag))
        plt.show()
        plt.close()

    # inds = np.argsort(list(bf_hd.values()))
    # for i in range(0, len(np.flip(np.array(list(bf_hd.keys()))[inds]))):
    #     print(np.flip(np.array(list(bf_hd.keys()))[inds])[i],
    #           np.flip(np.array(list(bf_hd.values()))[inds])[i])

    # print('')
    # bf_hd_psr = np.array(list(likelihood_hd.values())) - np.array(list(likelihood_null.values()))
    # inds = np.argsort(bf_hd_psr)
    # for i in range(0, len(np.flip(np.array(list(bf_hd.keys()))[inds]))):
    #     print(np.flip(np.array(list(bf_hd.keys()))[inds])[i],
    #           np.flip(bf_hd_psr[inds])[i])

    # print('')
    # bf_curn_psr = np.array(list(likelihood_curn.values())) - np.array(list(likelihood_null.values()))
    # inds = np.argsort(bf_curn_psr)
    # for i in range(0, len(np.flip(np.array(list(bf_hd.keys()))[inds]))):
    #     print(np.flip(np.array(list(bf_hd.keys()))[inds])[i],
    #           np.flip(bf_curn_psr[inds])[i])


    ts = (likelihood_hd_global - likelihood_curn_global)
    # ts_curn = 2*(likelihood_hd_global - likelihood_curn_global)
    print(" ")
    print('HD test statistic')
    print(ns, ts)

    if ns >= 1:
        ts_scrambles.append(ts)
    # ts_scrambles_curn.append(ts_curn)


#np.save("likelihood_ratios_more1713_fixA_{}.npy".format(tag), ts_scrambles)
# ts_scrambles = np.array(ts_scrambles)
ts_scrambles = np.load("likelihood_ratios_more1713_fixA_{}.npy".format(tag))
#ts_scrambles = np.array(ts_scrambles)

plt.figure(figsize=figsize)
plt.hist(np.log10(np.exp(ts_scrambles)), bins=25, color='darkgreen', lw=0.8, ec='k', log=False, density=False)
yl=plt.ylim()
#plt.plot([2.881695971811496, 2.881695971811496], yl, 'k--', linewidth=2)
plt.plot([np.log10(np.exp(2.2860087657149393)), np.log10(np.exp(2.2860087657149393))], yl, 'k--', linewidth=2)
#plt.plot([np.log10(np.exp(1.9489000987995837)), np.log10(np.exp(1.9489000987995837))], yl, 'k--', linewidth=2)

plt.ylim(yl)
#plt.ylabel(r'PDF')
plt.ylabel(r'Number of randomised skies, $N_{\rm sky}$')
plt.xlabel(r'$\Delta  \log_{10} \mathcal{L}$')
plt.savefig('scrambles_{}.png'.format(tag))
plt.savefig('scrambles_{}.pdf'.format(tag))
plt.show()

#print(len(np.array(ts_scrambles)[(ts_scrambles > 2.881695971811496)]) / len(np.array(ts_scrambles)[(ts_scrambles < 2.881695971811496)]) * 100)
print(len(np.array(ts_scrambles)[(ts_scrambles > 2.2860087657149393)]) / len(np.array(ts_scrambles)[(ts_scrambles < 2.2860087657149393)]) * 100)
#print(len(np.array(ts_scrambles)[(ts_scrambles > 1.9489000987995837)]) / len(np.array(ts_scrambles)[(ts_scrambles < 1.9489000987995837)]) * 100)


# plt.hist(ts_scrambles_curn, bins=30)
# yl=plt.ylim()
# plt.plot([6.126481254140344, 6.126481254140344], yl)
# plt.ylim(yl)
# plt.ylabel('N')
# plt.xlabel('Test statistic')
# plt.show()

# print(len(np.array(ts_scrambles_curn)[(ts_scrambles_curn > 6.126481254140344)]) / len(np.array(ts_scrambles_curn)[(ts_scrambles_curn < 6.126481254140344)]) * 100)


bf_psrs = []
bf_psrs_mono = []
for psr in likelihood_hd.keys():
    bf_psrs.append(likelihood_hd[psr] - likelihood_curn[psr])
    bf_psrs_mono.append(likelihood_mono[psr] - likelihood_curn[psr])
bf_psrs = np.array(bf_psrs)
bf_psrs_mono = np.array(bf_psrs_mono)

inds = np.flip(np.argsort(bf_psrs).squeeze())
for i in range(0, 30):
    print(np.array(list(likelihood_hd.keys()))[inds][i], bf_psrs[inds][i])

print("")

inds = np.flip(np.argsort(bf_psrs_mono).squeeze())
for i in range(0, 30):
    print(np.array(list(likelihood_mono.keys()))[inds][i], bf_psrs_mono[inds][i])







