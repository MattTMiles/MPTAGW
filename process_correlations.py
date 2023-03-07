#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:01:00 2023

@author: dreardon
"""

import bilby
import numpy as np
import matplotlib.pyplot as plt
from correlations_utils import *
from KDEpy import FFTKDE
from scipy.signal import savgol_filter


from matplotlib import rc
import matplotlib
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams.update({'font.size': 16})

psrnames = ['J0030+0451',
            'J0125-2327',
            'J0437-4715',
            'J0613-0200',
            'J0614-3329',
            'J0711-6830',
            'J0900-3144',
            'J1017-7156',
            'J1022+1001',
            'J1024-0719',
            'J1045-4509',
            'J1125-6014',
            'J1446-4701',
            'J1545-4550',
            'J1600-3053',
            'J1603-7202',
            'J1643-1224',
            'J1713+0747',
            'J1730-2304',
            'J1744-1134',
            'J1832-0836',
            'J1857+0943',
            'J1902-5105',
            'J1909-3744',
            'J1933-6211',
            'J1939+2134',
            'J2124-3358',
            'J2129-5721',
            'J2145-0750',
            'J2241-5236']

corr_hd_all = {}
datadir = '/Users/dreardon/Desktop/ppta_correlations/corr_chains'
pairs = get_pairs(psrnames)
for i in range(1, 436): # 435 pulsar pairs
    pair = pairs[str(1 + (int(i) - 1) % 435)]
    corr_hd_all['_'.join(pair)] = np.load("corr_chains/{}_corr_hd.npy".format(i))

import json, random

ts_scrambles = []

bad_list = []


#new_pos = {}
#for i, k in enumerate(pos.keys()):
#    new_pos[k] = np.random.uniform(low=-1, high=1, size=3)

for _ in range(0, 1):

    with open('positions.json', 'r') as f:
        pos = json.load(f)

    npos = len(np.array(list(pos.values())))
    inds = np.linspace(0, npos-1, npos)
    random.shuffle(inds)
    new_pos = {}
    for i, k in enumerate(pos.keys()):
        #new_pos[k] = np.array(list(pos.values()))[int(inds[i])]
        new_pos[k] = np.random.uniform(low=-1, high=1, size=3)
    pos = new_pos


    # Look through pairs and make plots
    kde = {}
    pdf = {}
    ptot = {}

    orf_bins_total = {}
    numerator = 0
    denominator = 0
    numerator_68 = 0
    denominator_68 = 0
    donepair = {}
    likelihood_hd = {}
    likelihood_curn = {}
    likelihood_null = {}
    likelihood_hd_global = 0
    likelihood_mono_global = 0
    likelihood_dipole_global = 0
    likelihood_curn_global = 0
    likelihood_null_global = 0
    bf_hd = {}
    n_bins = {}
    n_tot = 0

    y_flat = np.linspace(-1, 1, 256)
    null_prob = np.ones(np.shape(y_flat))
    null_prob /= np.sum(null_prob)
    null_prob /= np.mean(np.diff(np.linspace(-1, 1, 256)))
    #null_prob = np.ones(np.shape(null_prob))

    nseps = 8
    vals = np.linspace(0, 180, nseps+1)
    halfdiff = np.mean(np.diff(vals))/2
    vals = (vals - halfdiff)[1:]
    seps = {}

    nseps2 = 7
    vals2 = np.array([1e-3, 30.0, 50.0, 80.0, 100.0, 120.0, 150.0, 180.0])
    halfdiff2 = np.mean(np.diff(vals2))/2
    vals2 = (vals2[1:] - np.diff(vals2)/2)
    seps2 = {}

    data_bins = np.load("bins_chain_total.npy")


    for ibin in range(1, nseps+1):
        seps["bin_{}".format(ibin)] = vals[ibin-1]
    for ibin in range(1, nseps2+1):
        seps2["bin_{}".format(ibin)] = vals2[ibin-1]

    for _ in range(0,1):

    #for psri in psrnames:
        orf_bins = {}
        for i in range(1, 436): # 435 pulsar pairs

            pair = pairs[str(1 + (int(i) - 1) % 435)]
            print(i, '_'.join(pair))

            psr1 = pair[0]
            psr2 = pair[1]

            psri='dummy2'

            #if ('0437' not in psr1) and ('0437' not in psr2):
            #    continue
            #if ('1909' not in psr1) and ('1909' not in psr2):
            #    continue

            #if '1600' in psr1 or '1600' in psr2:
            #    continue
            #if '2145' in psr1 or '2145' in psr2:
            #    continue
            #if '1545' in psr1 or '1545' in psr2:
            #    continue
            #if '0437' in psr1 or '0437' in psr2:
            #    continue

            #if '0437' in psr1 or '0437' in psr2:
            #    continue
            #if '1713' in psr1 or '1713' in psr2:
            #    continue

            if not '_'.join(pair) in donepair:
                donepair['_'.join(pair)] = False

            # corr_hd = np.array(corr_hd_all['_'.join(pair)])  # bw=0.1, refecting correlations but not amplitude. Slice taken at -14.69
            # y_flat = np.linspace(-1, 1, 256)

            """
            ind = -6  # corr coeff
            ind_amp = -7  # amplitude

            chain = np.load("corr_chains/{}.npy".format(i))

            chain_2d = chain[:, [ind_amp, ind]].squeeze()


            amp = chain_2d[:,0]
            corr = chain_2d[:,1]

            reweight_factor = 10**(amp) / (10**-14 - 10**-18) # to uniform

            a2 = (10**(amp * reweight_factor) * 10**-14)**2  # uniform in linear

            square_reweight = abs(10**56 * a2 * corr)

            cov = a2 * square_reweight * corr
            print(len(cov))

            medcov = np.median(cov)
            q16 = np.percentile(cov, q=16)
            q84 = np.percentile(cov, q=84)
            stdcov = np.std(cov)
            cov68 = (q84-q16)/2

            print(medcov, q16, q84, stdcov, cov68)

            plt.hist(cov, bins=1000, density=True)
            plt.title(' '.join(pair))

            kde = gaussian_kde(cov, bw_method=0.1)

            grid_coords = np.linspace(-1, 1, 1000)
            cov_hd = kde(grid_coords)

            cov_hd /= np.sum(cov_hd)
            cov_hd /= np.mean(np.diff(grid_coords))

            plt.plot(grid_coords, cov_hd)
            plt.xlim([-1, 1])
            plt.show()

            np.save("corr_chains/{}_cov_hd.npy".format(i), cov_hd)
            continue
            """



            # Find index corresponding to the correlation corefficient
            ind = -6  # corr coeff
            ind_amp = -7  # amplitude

            chain_name = "corr_chains/{}_more1713.npy".format(i)
            chain = np.load(chain_name)

            #if len(chain[:, ind]) <= 3000:
            #    print(i, pair, len(chain[:, ind]))
            # continue

            if not "fixA" in chain_name:
                # FOR TAKING ONLY SOME SAMPLES
                # chain_2d = chain[1::2, [ind_amp, ind]].squeeze()
                chain_2d = chain[:, [ind_amp, ind]].squeeze()

                """
                COMPUTE OPTIMAL STATISTIC
                """

                log10_amps = chain_2d[:,0]
                corrs = chain_2d[:,1]
                cov = (10**log10_amps) ** 2 * corrs

                # uniform in A**2 * Gam
                weights = abs(corrs * 2**(2*log10_amps + 1) * 5**(2*log10_amps) * np.log(10))

                # Uniform in A * Gam
                weights = abs(corrs * 10**log10_amps * np.log(10))


                vals = plt.hist(cov, bins=200, range=[-10e-30, 10e-30], weights=weights, density=True)
                plt.xlim([-10*10**-30, 10*10**-30])
                plt.title(' '.join(pair))

                densities = vals[0]
                edges = vals[1]
                centres = edges[:-1] + np.diff(edges)/2

                smooth_dense = savgol_filter(densities, 10, 1)
                plt.plot(centres, smooth_dense)
                plt.close()

                np.savez("corr_chains/{}_os.npz".format(i), smooth_dense, centres)
                print("saved", "corr_chains/{}_os.npz".format(i))

                continue

                print("Samples above -14.69: {}".format(len(np.argwhere(chain_2d[:, 0] >= -14.69))))
                if len(np.argwhere(chain_2d[:, 0] >= -14.69)) < 1000:
                    bad_list.append('_'.join(pair))
                    print("*** WARNING! TOO FEW SAMPLES! ***")

                #plt.subplots(1, 2, figsize=(12,6))
                #plt.subplot(1, 2, 1)
                #h, xedges, yedges, _ = plt.hist2d(chain_2d[:, 0], chain_2d[:, 1], bins=100, density=True, range=[[-18, -14],[-1, 1]])
                #plt.ylabel('Corr coeff')
                #plt.xlabel('log10 A')
                #plt.colorbar()
                #plt.title(' '.join(pair))
                #plt.tight_layout()


                chain_2d_reflect_corrs_pos = chain_2d.copy()
                chain_2d_reflect_corrs_pos[:, 1] = 2*(1) - chain_2d_reflect_corrs_pos[:, 1]
                chain_2d_reflect_corrs_neg = chain_2d.copy()
                chain_2d_reflect_corrs_neg[:, 1] = 2*(-1) - chain_2d_reflect_corrs_neg[:, 1]

                chain_mirror_corr = np.concatenate((chain_2d_reflect_corrs_neg, chain_2d, chain_2d_reflect_corrs_pos))

                chain_2d_reflect_amp_pos = chain_mirror_corr.copy()
                chain_2d_reflect_amp_pos[:, 0] = 2*(-14) - chain_2d_reflect_amp_pos[:, 0]
                chain_2d_reflect_amp_neg = chain_mirror_corr.copy()
                chain_2d_reflect_amp_neg[:, 0] = 2*(-18) - chain_2d_reflect_amp_neg[:, 0]

                chain_mirror = np.concatenate((chain_2d_reflect_amp_neg, chain_mirror_corr, chain_2d_reflect_amp_pos))
                #chain_mirror = np.concatenate((chain_2d_reflect_amp_neg, chain_mirror_corr))
                #chain_mirror = chain_mirror_corr

                #gkde_2d = gaussian_kde(np.transpose(chain_mirror))

                #p, bins, _ = plt.hist(chain[:, ind], range=(-1, 1), bins=100, density=True)
                #plt.xlim([-1, 1])
                #plt.close()
                #centres = bins[0:-1] + np.diff(bins)
                #gkde = gaussian_kde(chain[:, ind].squeeze(), bw_method=0.2)

                bw = 0.05

                data = chain_mirror.squeeze()
                grid_points = 1024  # Grid points in each dimension
                kde = FFTKDE(kernel='gaussian', bw=bw)
                grid, points = kde.fit(data).evaluate((grid_points, grid_points))
                # The grid is of shape (obs, dims), points are of shape (obs, 1)
                x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                # x is amplitudes
                # y is corr coeff
                z = points.reshape(grid_points, grid_points).T

                z[y<=-1, :] = 0  # Set the KDE to zero outside of the domain
                z[y>=1, :] = 0  # Set the KDE to zero outside of the domain
                z[:, x<=-18] = 0  # Set the KDE to zero outside of the domain
                z[:, x>=-14] = 0  # Set the KDE to zero outside of the domain
                z = z.squeeze() * 9  # multiply the kde to get integral of ~1

                indx = np.argwhere((x>=-18)*(x<=-14)).squeeze()
                indy = np.argwhere((y>=-1)*(y<=1)).squeeze()

                y = y[indy]
                x = x[indx]
                z = z[indy, :][:, indx]

                #plt.subplot(1, 2, 2)
                #plt.pcolormesh(x,y,z)
                #plt.colorbar()
                #plt.xlim([-18,-14])
                #plt.ylim([-1,1])

                #plt.ylabel('Corr coeff')
                #plt.xlabel('log10 A')

                np.savez("corr_chains/{}_pdf_importance_more1713.npz".format(i), x, y, z)

                # sys.exit()

                # x_flat = np.linspace(-18, -14, 20)
                # # x_flat = np.array([-14.69])
                # y_flat = np.linspace(-1, 1, 20)
                # xcentre = xedges[1:] - np.mean(np.diff(xedges))/2
                # ycentre = yedges[1:] - np.mean(np.diff(yedges))/2
                # x, y = np.meshgrid(xcentre, ycentre)
                # grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

                # z = kde(grid_coords.T)

                # z = z.reshape(20, 20)
                # z /= np.sum(z)
                # z /= np.mean(np.diff(xcentre))
                # z /= np.mean(np.diff(ycentre))

                # plt.subplot(3, 1, 2)
                # plt.title("KDE, bw={}".format(bw))
                # plt.pcolormesh(xedges, yedges, z)
                # plt.colorbar()
                # plt.ylabel('Corr coeff')
                # plt.xlabel('log10 A')


                # plt.subplot(3, 1, 3)
                # plt.pcolormesh(xedges, yedges, z - h.T)
                # plt.colorbar()
                # plt.title("Residual, standard deviation = {}".format(round(np.std(z - h.T), 2)))
                # plt.ylabel('Corr coeff')
                # plt.xlabel('log10 A')

                #plt.tight_layout()

                #plt.savefig('corr_plots/{}_2dhist.png'.format('_'.join(pair)))
                #plt.show()


                nsamp = 1000

                samples = np.array([])

                if nsamp > 1:
                    for _ in range(0, nsamp):
                        # draw an amplitude
                        amp = np.random.normal(loc=-14.69, scale=0.05)
                        indx = np.argmin(np.abs(x - amp)).squeeze()

                        pdf = z[:, indx]
                        pdf /= np.sum(pdf)

                        samps = np.random.choice(y, size=1000, p=pdf)
                        samples = np.concatenate((samples, samps))

                    samples_pos = samples.copy()
                    samples_pos = 2*(1) - samples_pos
                    samples_neg = samples.copy()
                    samples_neg = 2*(-1) - samples_neg

                    samples = np.concatenate((samples_neg, samples, samples_pos))

                    bw = 0.1

                    data = samples.squeeze()
                    grid_points = 4096  # Grid points in each dimension
                    kde = FFTKDE(kernel='gaussian', bw=bw)
                    grid, points = kde.fit(data).evaluate(grid_points)
                    # The grid is of shape (obs, dims), points are of shape (obs, 1)
                    y2 = np.unique(grid)

                    z2 = points.reshape(grid_points).T

                    z2[y2<=-1] = 0  # Set the KDE to zero outside of the domain
                    z2[y2>=1] = 0  # Set the KDE to zero outside of the domain
                    z2 = z2.squeeze() * 3  # multiply the kde to get integral of ~1

                    indy = np.argwhere((y2>=-1)*(y2<=1)).squeeze()
                    y2 = y2[indy]

                    corr_hd2 = z2[indy] / np.sum(z2[indy])

                indx = np.argmin(np.abs(x + 14.69)).squeeze()
                corr_hd = z[:, indx]
                corr_hd /= np.sum(corr_hd)

                # if ('1713' in psr1 or '1713' in psr2) and not '1909' in psr2:
                #    corr_hd = np.ones(np.shape(corr_hd))
                #corr_hd /= np.sum(corr_hd)

                corr_hd /= np.mean(np.diff(y))
                corr_hd2 /= np.mean(np.diff(y2))

                #plt.plot(y, corr_hd)
                #plt.plot(y2, corr_hd2)
                #yl = plt.ylim()
                #plt.ylim([0, yl[1]*1.1])
                #plt.xlim([-1, 1])
                #plt.savefig('corr_plots/{}_1dpdf.png'.format('_'.join(pair)))
                #plt.xlabel('Corr coeff')
                #plt.show()
                np.savez("corr_chains/{}_corr_hd_importance_more1713.npz".format(i), corr_hd2, y2)
                print("Saved:", "corr_chains/{}_corr_hd_importance_more1713.npz".format(i))
                print("")
                continue

            else: # fixA, so only a 1d chain
                chain_1d = chain[:, ind].squeeze()

                chain_1d_reflect_corrs_pos = chain_1d.copy()
                chain_1d_reflect_corrs_pos = 2 - chain_1d_reflect_corrs_pos
                chain_1d_reflect_corrs_neg = chain_1d.copy()
                chain_1d_reflect_corrs_neg = -2 - chain_1d_reflect_corrs_neg

                chain_mirror = np.concatenate((chain_1d_reflect_corrs_neg, chain_1d, chain_1d_reflect_corrs_pos))

                bws = [0.075, 0.1, 0.125, 0.15]

                for bw in bws:

                    data = chain_mirror.squeeze()
                    grid_points = 1024  # Grid points in each dimension

                    kde = FFTKDE(kernel='gaussian', bw=bw)
                    grid, points = kde.fit(data).evaluate(grid_points)
                    # The grid is of shape (obs, dims), points are of shape (obs, 1)
                    y = np.unique(grid)
                    # y is corr coeff
                    z = points.reshape(grid_points)

                    z[y<=-1] = 0  # Set the KDE to zero outside of the domain
                    z[y>=1] = 0  # Set the KDE to zero outside of the domain
                    z = z.squeeze() * 3  # multiply the kde to get integral of ~1

                    indy = np.argwhere((y>=-1)*(y<=1)).squeeze()

                    y = y[indy]
                    z = z[indy]

                    np.savez("corr_chains/{}_corr_hd_importance_fixA_{}.npz".format(i, bw), z, y)
                    print("Saved:", "corr_chains/{}_corr_hd_importance_fixA_{}.npz".format(i, bw))
                    print("")
                continue



            sigma_k = np.std(np.random.choice(np.linspace(-1, 1, 256), p=corr_hd/np.sum(corr_hd), size=10000))
            q_16 = np.percentile(np.random.choice(np.linspace(-1, 1, 256), p=corr_hd/np.sum(corr_hd), size=10000), q=16)
            q_84 = np.percentile(np.random.choice(np.linspace(-1, 1, 256), p=corr_hd/np.sum(corr_hd), size=10000), q=84)
            sigma_68 = np.abs(q_84 - q_16)/2

            pos1 = pos[psr1]
            pos2 = pos[psr2]

            angsep = np.arccos(np.dot(pos1, pos2)) * 180/np.pi
            orf_val = hd_orf(np.array([angsep]))[0]
            dipole_val = dipole(np.array([angsep]))[0]

            try:
                likelihood_hd[psri] += np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))])
                likelihood_curn[psri] += np.log(corr_hd[np.argmin(np.abs(y_flat))])
                likelihood_null[psri] += np.log(null_prob[0])
            except KeyError:
                likelihood_hd[psri] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))]))
                likelihood_curn[psri] = np.copy(np.log(corr_hd[np.argmin(np.abs(y_flat))]))
                likelihood_null[psri] = np.log(null_prob[0])

            ibin = np.argmin(np.abs(vals - angsep)) + 1
            try: orf_bins["bin_{}".format(ibin)] *= corr_hd
            except KeyError as e: orf_bins["bin_{}".format(ibin)] = corr_hd

            if not donepair['_'.join(pair)]:
                numerator += (sigma_k)**-2
                denominator += (sigma_k)**-4
                numerator_68 += (sigma_68)**-2
                denominator_68 += (sigma_68)**-4
                try:
                    orf_bins_total["bin_{}".format(ibin)] *= corr_hd
                    n_bins["bin_{}".format(ibin)] +=1
                except KeyError as e:
                    orf_bins_total["bin_{}".format(ibin)] = np.copy(corr_hd * null_prob)
                    n_bins["bin_{}".format(ibin)] = 1

                try:
                    likelihood_hd_global += np.log(corr_hd[np.argmin(np.abs(y_flat - orf_val))])
                    likelihood_curn_global += np.log(corr_hd[np.argmin(np.abs(y_flat))])
                    likelihood_mono_global += np.log(corr_hd[np.argmin(np.abs(y_flat - 1))])
                    likelihood_dipole_global += np.log(corr_hd[np.argmin(np.abs(y_flat - dipole_val))])
                    likelihood_null_global += np.log(null_prob[0])
                except KeyError:
                    likelihood_hd_global = corr_hd[np.argmin(np.abs(y_flat - orf_val))]
                    likelihood_curn_global = corr_hd[np.argmin(np.abs(y_flat))]
                    likelihood_mono_global = corr_hd[np.argmin(np.abs(y_flat - 1))]
                    likelihood_dipole_global = corr_hd[np.argmin(np.abs(y_flat - dipole_val))]
                    likelihood_null_global = np.log(null_prob[0])

                n_tot += 1
                donepair['_'.join(pair)] = True


            import sys
            sys.exit()

        # fig, ax = plt.subplots(1,1, figsize=(9,6))
        # for k in orf_bins.keys():
        #     orf_bins[k] /= np.sum(orf_bins[k])
        #     orf_bins[k] /= np.mean(np.diff(np.linspace(-1, 1, 256)))
        #     draws = np.random.choice(np.linspace(-1, 1, 256), p=orf_bins[k]/np.sum(orf_bins[k]), size=100000)
        #     ax = plot_violin(ax, seps[k], draws)


        # theta = np.linspace(0, 180, 1000)
        # orf = hd_orf(theta)
        # plt.plot(theta, orf, color='k', linewidth=3)
        # plt.ylim([-1, 1])
        # plt.savefig('corr_plots/{}.png'.format(psri))
        # plt.savefig('corr_plots/{}.pdf'.format(psri))
        # plt.close()

        #bf_hd[psri] = (likelihood_hd[psri] - likelihood_curn[psri])


    fig, ax1 = plt.subplots(1, 1, figsize=(9,6))
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
    #plt.stairs(values, edges=edges, linewidth=1, color='k')

    for k in orf_bins_total.keys():
        orf_bins_total[k] /= np.sum(orf_bins_total[k])
        orf_bins_total[k] /= np.mean(np.diff(np.linspace(-1, 1, 256)))
        np.save('/Users/dreardon/Desktop/ppta_correlations/{}.npy'.format(k), orf_bins_total[k]/np.sum(orf_bins_total[k]))
        draws = np.random.choice(np.linspace(-1, 1, 256), p=orf_bins_total[k]/np.sum(orf_bins_total[k]), size=10000)
        ax = plot_violin(ax1, seps[k], draws, width=np.mean(np.diff(vals)), alpha=0.9)

        ibin = int(k.split('_')[-1])

    # for k in seps2.keys():
    #     ibin = int(k.split('_')[-1])
    #     ax = plot_violin(ax1, seps2[k], data_bins[:, -(14-ibin-1)], width=np.mean(np.diff(vals2)), colour='crimson', alpha=0.5)


    theta = np.linspace(0, 180, 1000)
    orf = hd_orf(theta)
    ax1.plot(theta, orf, color='k', linewidth=3)
    ax1.set_ylim([-1, 1])
    ax2.set_ylim([0, round(max(n_bins.values()), -1)])
    ax2.set_ylabel('Number of pulsar pairs')
    plt.xlim([0, 180])
    plt.tight_layout()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    plt.savefig('corr_plots/corr_total.png')
    plt.savefig('corr_plots/corr_total.pdf')
    plt.show()
    plt.close()

    neff = numerator**2 / denominator
    neff_68 = numerator_68**2 / denominator_68
    print("Number of effective pulsar pairs, using standard deviation = {}".format(neff))
    print("Number of effective pulsar pairs, using 68% confidence = {}".format(neff_68))
    print(len(donepair.keys()))

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

    print(" ")
    print('HD log likelihood difference')
    print(likelihood_hd_global - likelihood_null_global)
    print(" ")
    print("HD S/N estimate")
    print(np.sqrt(2*(likelihood_hd_global - likelihood_null_global)))



    ts = 2*(likelihood_hd_global - likelihood_null_global)

    ts_scrambles.append(ts)

    new_pos.clear()
    kde.clear()
    pdf.clear()
    ptot.clear()
    orf_bins_total.clear()
    donepair.clear()
    likelihood_hd.clear()
    likelihood_curn.clear()
    likelihood_null.clear()
    bf_hd.clear()
    n_bins.clear()
    seps.clear()
    seps2.clear()
    orf_bins.clear()

ts_scrambles = np.load("likelihood_ratios.npy")
ts_scrambles = np.array(ts_scrambles)

plt.hist(ts_scrambles, bins=20)
yl=plt.ylim()
plt.plot([1, 1], yl)
plt.ylim(yl)
plt.ylabel('N')
plt.xlabel('Test statistic')

len(np.array(ts_scrambles)[(ts_scrambles > 1)]) / len(np.array(ts_scrambles)[(ts_scrambles < 1)]) * 100










