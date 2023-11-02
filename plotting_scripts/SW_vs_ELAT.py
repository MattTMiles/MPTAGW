# Plots the relationship between ELAT and SW

import numpy as np
import matplotlib.pyplot as plt

from enterprise.pulsar import Pulsar
import glob
import ephem
import datetime
import gc
import random
from sklearn.neighbors import KernelDensity


mpta_dir = "//fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ptmcmc/PM_WN/"

psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"

partim = "/fred/oz002/users/mmiles/MPTA_GW/partim/"

to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []

parfiles = sorted(glob.glob(partim + '/*par'))
timfiles = sorted(glob.glob(partim + '/*tim'))

elats = []
sw_posts = []

for psr in to_use:
    psrname = psr.strip("\n")
    psr_dir = mpta_dir + "/" + psrname
    parfile = [x for x in parfiles if x.split('/')[-1].split('.')[0] == psrname][0]
    timfile = [x for x in timfiles if x.split('/')[-1].split('.')[0] == psrname][0]

    pars = list(open(psr_dir+"/pars.txt").readlines())
    pars = [ p.strip("\n") for p in pars ]

    ref_epoch = ephem.Date(datetime.datetime(2000, 1, 1, 12, 0))

    if psrname+"_n_earth_n_earth" in pars:
        print(psrname)
        psr = Pulsar(parfile, timfile, ephem="DE440")
        
        
        ma = ephem.Equatorial(psr._raj, psr._decj, epoch=ephem.Date(ref_epoch))
        me = ephem.Ecliptic(ma)

        elat = me.lat
        elats.append(elat)
        
        result_psr = np.load(psr_dir+"/"+psrname+"_burnt_chain.npy")
        sw_index = pars.index(psrname+"_n_earth_n_earth")
        posts_SW = result_psr[:, sw_index]

        sw_posts.append(posts_SW)
        gc.collect()



elats = np.array(elats)
elats = elats*180/np.pi

sw_arrays = [ np.array(sw) for sw in sw_posts ]

fig = plt.figure(figsize=(15,10))
axes = fig.add_subplot(211)
axes2 = fig.add_subplot(212)
axes.violinplot(sw_arrays, positions=elats, widths=3, showextrema=False, points=50)
axes.axhline(4, color="grey", linestyle="--", alpha=0.5)
axes2.violinplot(sw_arrays, positions=elats, widths=3, showextrema=False, points=50)
axes2.axhline(4, color="grey", linestyle="--", alpha=0.5)
axes2.set_xlim(-15,15)
axes2.set_ylabel(r"$\mathrm{n_{\oplus}} (\mathrm{cm}^{-3})$")
axes.set_ylabel(r"$\mathrm{n_{\oplus}} (\mathrm{cm}^{-3})$")
axes2.set_xlabel("Ecliptic Latitude (degrees)")


fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/ELAT_vs_SW.png")
fig.clf()

# Bin the SW together so that it can be factorised
# Do it in step sizes of 15 then shorten down at -45 to a step of 10
# Then shorten again at -25 to a step of 5


elat_1_idxs =  np.where((-90 < elats) * (elats< -72.5))[0]
elat_2_idxs =  np.where((-72.5 < elats) * (elats < -57.5))[0]
elat_3_idxs =  np.where((-57.5 < elats) * (elats < -42.5))[0]
elat_4_idxs =  np.where((-42.5 < elats) * (elats < -32.5))[0]
elat_5_idxs =  np.where((-32.5 < elats) * (elats < -22.5))[0]
elat_6_idxs =  np.where((-22.5 < elats) * (elats < -17.5))[0]
elat_7_idxs =  np.where((-17.5 < elats) * (elats < -12.5))[0]
elat_8_idxs =  np.where((-12.5 < elats) * (elats < -7.5))[0]
elat_9_idxs = np.where((-7.5 < elats) * (elats < -2.5))[0]
elat_10_idxs =  np.where((-2.5 < elats) * (elats < 2.5))[0]
elat_11_idxs =  np.where((2.5 < elats) * (elats < 7.5))[0]
elat_12_idxs =  np.where((7.5 < elats) * (elats < 12.5))[0]
elat_13_idxs =  np.where((12.5 < elats) * (elats < 17.5))[0]
elat_14_idxs =  np.where((17.5 < elats) * (elats < 22.5))[0]
elat_15_idxs =  np.where((22.5 < elats) * (elats < 27.5))[0]
elat_16_idxs =  np.where((27.5 < elats) * (elats < 32.5))[0]
elat_17_idxs =  np.where((32.5 < elats) * (elats < 37.5))[0]
elat_18_idxs =  np.where((elats > 37.5))[0]

elat_1_sw = [ sw_arrays[idx] for idx in elat_1_idxs ]
elat_2_sw = [ sw_arrays[idx] for idx in elat_2_idxs ]
elat_3_sw = [ sw_arrays[idx] for idx in elat_3_idxs ]
elat_4_sw = [ sw_arrays[idx] for idx in elat_4_idxs ]
elat_5_sw = [ sw_arrays[idx] for idx in elat_5_idxs ]
elat_6_sw = [ sw_arrays[idx] for idx in elat_6_idxs ]
elat_7_sw = [ sw_arrays[idx] for idx in elat_7_idxs ]
elat_8_sw = [ sw_arrays[idx] for idx in elat_8_idxs ]
elat_9_sw = [ sw_arrays[idx] for idx in elat_9_idxs ]
elat_10_sw = [ sw_arrays[idx] for idx in elat_10_idxs ]
elat_11_sw = [ sw_arrays[idx] for idx in elat_11_idxs ]
elat_12_sw = [ sw_arrays[idx] for idx in elat_12_idxs ]
elat_13_sw = [ sw_arrays[idx] for idx in elat_13_idxs ]
elat_14_sw = [ sw_arrays[idx] for idx in elat_14_idxs ]
elat_15_sw = [ sw_arrays[idx] for idx in elat_15_idxs ]
elat_16_sw = [ sw_arrays[idx] for idx in elat_16_idxs ]
elat_17_sw = [ sw_arrays[idx] for idx in elat_17_idxs ]
elat_18_sw = [ sw_arrays[idx] for idx in elat_18_idxs ]

nbins = 20
SWspace = np.linspace(0,20,nbins*100)

for i, sw in enumerate(elat_1_sw):
    pSW, binSW, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p1 = (pSW + 1e-20)
        i = i+1
    else:
        p1 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_2_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')
    
    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p2 = (pSW + 1e-20)
        i = i+1
    else:
        p2 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_3_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p3 = (pSW + 1e-20)
        i = i+1
    else:
        p3 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_4_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p4 = (pSW + 1e-20)
        i = i+1
    else:
        p4 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_5_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p5 = (pSW + 1e-20)
        i = i+1
    else:
        p5 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_6_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p6 = (pSW + 1e-20)
        i = i+1
    else:
        p6 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_7_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p7 = (pSW + 1e-20)
        i = i+1
    else:
        p7 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_8_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p8 = (pSW + 1e-20)
        i = i+1
    else:
        p8 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_9_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p9 = (pSW + 1e-20)
        i = i+1
    else:
        p9 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_10_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p10 = (pSW + 1e-20)
        i = i+1
    else:
        p10 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_11_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p11 = (pSW + 1e-20)
        i = i+1
    else:
        p11 *= pSW
        i = i+1

for i, sw in enumerate(elat_12_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p12 = (pSW + 1e-20)
        i = i+1
    else:
        p12 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_13_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p13 = (pSW + 1e-20)
        i = i+1
    else:
        p13 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_14_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p14 = (pSW + 1e-20)
        i = i+1
    else:
        p14 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_15_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p15 = (pSW + 1e-20)
        i = i+1
    else:
        p15 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_16_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p16 = (pSW + 1e-20)
        i = i+1
    else:
        p16 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_17_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p17 = (pSW + 1e-20)
        i = i+1
    else:
        p17 *= (pSW + 1e-20)
        i = i+1

for i, sw in enumerate(elat_18_sw):
    pSW, _, _= plt.hist(sw, bins=nbins, range=(0, 20), density=True, alpha=0.6, histtype='step')

    kdevals = KernelDensity(bandwidth=1.0, kernel="gaussian")
    kdevals.fit(sw[:, None])
    logprob = kdevals.score_samples(SWspace[:, None])
    kde_eval = np.exp(logprob)
    pSW = kde_eval

    if i==0:
        p18 = (pSW + 1e-20)
        i = i+1
    else:
        p18 *= (pSW + 1e-20)
        i = i+1


#Now need to resample across the prior range for each FL bin



elats_revised = []

bindiff = binSW[1] - binSW[0]
p1adjust = p1/(np.sum(p1)*bindiff)
elats_revised.append(-81.25)
p2adjust = p2/(np.sum(p2)*bindiff)
elats_revised.append(-65)
p3adjust = p3/(np.sum(p3)*bindiff)
elats_revised.append(-50)
p4adjust = p4/(np.sum(p4)*bindiff)
elats_revised.append(-37.5)
p5adjust = p5/(np.sum(p5)*bindiff)
elats_revised.append(-27.5)
p6adjust = p6/(np.sum(p6)*bindiff)
elats_revised.append(-20)
p7adjust = p7/(np.sum(p7)*bindiff)
elats_revised.append(-15)
p8adjust = p8/(np.sum(p8)*bindiff)
elats_revised.append(-10)
p9adjust = p9/(np.sum(p9)*bindiff)
elats_revised.append(-5)
p10adjust = p10/(np.sum(p10)*bindiff)
elats_revised.append(0)
p11adjust = p11/(np.sum(p11)*bindiff)
elats_revised.append(5)
p12adjust = p12/(np.sum(p12)*bindiff)
elats_revised.append(10)
p13adjust = p13/(np.sum(p13)*bindiff)
elats_revised.append(15)
p14adjust = p14/(np.sum(p14)*bindiff)
elats_revised.append(20)
try:
    p15adjust = p15/(np.sum(p15)*bindiff)
    elats_revised.append(25)
except:
    pass
try:
    p16adjust = p16/(np.sum(p16)*bindiff)
    elats_revised.append(30)
except:
    pass
try:
    p17adjust = p17/(np.sum(p17)*bindiff)
    elats_revised.append(36.25)
except:
    pass
try:
    p18adjust = p18/(np.sum(p18)*bindiff)
    elats_revised.append(40)
except:
    pass

total_samples = []

resamp_N = 10000

p1sample = np.random.choice(SWspace, size=resamp_N, p=p1adjust/np.sum(p1adjust))
total_samples.append(p1sample)
p2sample = np.random.choice(SWspace, size=resamp_N, p=p2adjust/np.sum(p2adjust))
total_samples.append(p2sample)
p3sample = np.random.choice(SWspace, size=resamp_N, p=p3adjust/np.sum(p3adjust))
total_samples.append(p3sample)
p4sample = np.random.choice(SWspace, size=resamp_N, p=p4adjust/np.sum(p4adjust))
total_samples.append(p4sample)
p5sample = np.random.choice(SWspace, size=resamp_N, p=p5adjust/np.sum(p5adjust))
total_samples.append(p5sample)
p6sample = np.random.choice(SWspace, size=resamp_N, p=p6adjust/np.sum(p6adjust))
total_samples.append(p6sample)
p7sample = np.random.choice(SWspace, size=resamp_N, p=p7adjust/np.sum(p7adjust))
total_samples.append(p7sample)
p8sample = np.random.choice(SWspace, size=resamp_N, p=p8adjust/np.sum(p8adjust))
total_samples.append(p8sample)
p9sample = np.random.choice(SWspace, size=resamp_N, p=p9adjust/np.sum(p9adjust))
total_samples.append(p9sample)
p10sample = np.random.choice(SWspace, size=resamp_N, p=p10adjust/np.sum(p10adjust))
total_samples.append(p10sample)
p11sample = np.random.choice(SWspace, size=resamp_N, p=p11adjust/np.sum(p11adjust))
total_samples.append(p11sample)
p12sample = np.random.choice(SWspace, size=resamp_N, p=p12adjust/np.sum(p12adjust))
total_samples.append(p12sample)
p13sample = np.random.choice(SWspace, size=resamp_N, p=p13adjust/np.sum(p13adjust))
total_samples.append(p13sample)
p14sample = np.random.choice(SWspace, size=resamp_N, p=p14adjust/np.sum(p14adjust))
total_samples.append(p14sample)
try:
    p15sample = np.random.choice(SWspace, size=resamp_N, p=p15adjust/np.sum(p15adjust))
    total_samples.append(p15sample)
except:
    pass
try:
    p16sample = np.random.choice(SWspace, size=resamp_N, p=p16adjust/np.sum(p16adjust))
    total_samples.append(p16sample)
except:
    pass
try:
    p17sample = np.random.choice(SWspace, size=resamp_N, p=p17adjust/np.sum(p17adjust))
    total_samples.append(p17sample)
except:
    pass
try:
    p18sample = np.random.choice(SWspace, size=resamp_N, p=p18adjust/np.sum(p18adjust))
    total_samples.append(p18sample)
except:
    pass

elats_revised = np.array(elats_revised)

fig = plt.figure(figsize=(15,5))
axes = fig.add_subplot(111)
axes.axvline(-72.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-57.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-42.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-32.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-22.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-17.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-12.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-7.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(-2.5, color="grey", linestyle="-", alpha=0.25)
#axes.axvline(0, color="grey", linestyle="-", alpha=0.5)
axes.axvline(2.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(7.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(12.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(17.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(22.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(27.5, color="grey", linestyle="-", alpha=0.25)
axes.axvline(32.5, color="grey", linestyle="-", alpha=0.25)
#axes.axvline(35, color="grey", linestyle="-", alpha=0.25)

axes.violinplot(total_samples, positions=elats_revised, widths=3, showextrema=False, points=100)
axes.axhline(4, color="xkcd:burnt sienna", linestyle="--", alpha=0.5, linewidth=3)

axes.set_ylabel(r"$\mathrm{n_{\oplus}} (\mathrm{cm}^{-3})$", fontsize=14)
axes.set_xlabel(r"Ecliptic Latitude ($^{\circ}$)", fontsize=14)
axes.tick_params(axis="both", labelsize=12)
axes.set_xlim(-90, 40)
axes.set_xticks([-90, -72.5, -57.5, -42.5, -32.5, -22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 40])
#axes.set_xticklabels()
fig.tight_layout()
fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/ELAT_vs_SW_fact_like.png")
fig.clf()

