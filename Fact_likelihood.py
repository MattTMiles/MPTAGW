import numpy as np
import matplotlib.pyplot as plt
import bilby
import random
import corner
from scipy.stats import gaussian_kde
import glob
import pandas as pd
import json

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"
#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
stacked = []


i=0
for pulsar in to_use:
    psrname = pulsar.strip("\n")
    #if psrname != "J1811-2405" and psrname != "J1918-0642" and psrname != "J1737-0811":
    if psrname != "J00000":
    #if psrname != "J0711-6830" and psrname != "J0900-3144" and psrname != "J1017-7156" and psrname != "J1431-5740" and psrname != "J1547-5709" and psrname != "J1652-4838" and psrname != "J1708-3506" and psrname != "J1801-1417" and psrname != "J1802-2124" and psrname != "J2234+0944":
        print(psrname)
        try:
            #psr_SPGW = gw_dir + "/" + psrname + "/" + psrname + "_SPGW1000_ER"
            psr_SPGWC = glob.glob(gw_dir + "/" + psrname + "*SGWB/")[0]

            #result_SPGW = bilby.result.read_in_result(psr_SPGW+"/SPGW1000_ER_result.json")
            result_SPGWC = pd.read_json(json.load(open(glob.glob(psr_SPGWC+"/*final_res.json")[0])))

            #if not psrname+"_red_noise_log10_A" in result_SPGW.parameter_labels:
            #    posts_SPGW_A = result_SPGW.posterior["log10_A_gw"].values
            #    posts_SPGW_g = result_SPGW.posterior["gamma_gw"].values
            #else:
            #    posts_SPGW_A = result_SPGW.posterior[result_SPGW.posterior[psrname+"_red_noise_log10_A"] < -16.5]["log10_A_gw"].values
            #    posts_SPGW_g = result_SPGW.posterior[result_SPGW.posterior[psrname+"_red_noise_log10_A"] < -16.5]["gamma_gw"].values

            posts_SPGWC_A = result_SPGWC["log10_A_gw"].values

            #pdf_SPGW_A = np.histogram(posts_SPGW_A,bins=np.linspace(-18,-12,99),density=True)[0] + 1e-20
            #pdf_SPGW_g = np.histogram(posts_SPGW_g,bins=np.linspace(0,7,99),density=True)[0] + 1e-20
            
            pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,100),density=True)[0] + 1e-20
            #pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,24),density=True)[0] + 1e-20

            stacked.append(pdf_SPGWC_A)
            plt.figure(0)
            p, bins, patches = plt.hist(posts_SPGWC_A, bins=100, range=(-18, -12), density=True, alpha=0.6, histtype='step')
            #try:
            #except
            #pgam, binsgam, patchesgam = plt.hist(posts_SPGW_g, bins=100, range=(0, 7), density=True, alpha=0.6, histtype='step')
            #plt.figure(1)
            #p2d, xbins2d, ybins2d, im = plt.hist2d(posts_SPGW_A, posts_SPGW_g, bins=[20,20], range=([-18, -12],[0, 7]), density=True, alpha=0.6)
            #print(p2d)
            ind = np.argmax(p)
            centres = bins[0:-1] + np.diff(bins)
            print(centres[ind])

        except:
            continue
    
    #if np.any(p2d[:1,:]<=0.0001):
    #    print(psrname+" close to zero sample")
    #    continue
    
    #stacked = np.vstack(stacked)

    FWHM = 6
    #FWHM = 16
    FWHMalt = 2
    #FWHM2 = 
    sigma = fwhm2sigma(FWHM)
    sigmaalt = fwhm2sigma(FWHMalt)

    smoothed_vals = np.zeros(p.shape)
    x_vals = np.linspace(0,99,100)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[int(x_position)] = sum(p * kernel)



    #smoothed_valsg = np.zeros(pgam.shape)
    #x_vals = np.linspace(0,99,100)
    #for x_position in x_vals:
    #    kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
    #    kernel = kernel / sum(kernel)
    #    smoothed_valsg[int(x_position)] = sum(pgam * kernel)




    smoothed_valsC = np.zeros(pdf_SPGWC_A.shape)
    x_vals = np.linspace(0,98,99)
    #x_vals = np.linspace(0,22,23)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_valsC[int(x_position)] = sum(pdf_SPGWC_A * kernel)
        #smoothed_valsC[int(x_position)] = sum(p_total * kernel)
        #plt.plot()

    if i == 0:
        #fact_l_SPGW_A = pdf_SPGW_A
        #fact_l_SPGW_A = smoothed_vals
        #fact_l_SPGW_g = smoothed_valsg
        fact_l_SPGWC_A = smoothed_valsC
        fact_l_SPGWC_A_1 = smoothed_valsC
        fact_nosmooth = pdf_SPGWC_A
        p1 = (p + 1e-20)
    elif i==1:
        fact_l_SPGWC_A_2 = smoothed_valsC
        p2 = (p + 1e-20)
    elif i%2 == 0:
        #fact_l_SPGW_A *= pdf_SPGW_A
        #fact_l_SPGW_A *= smoothed_vals
        #fact_l_SPGW_A = np.histogram(posts_SPGW_A,bins=np.linspace(-18,-12,100),density=True)[0]
        #fact_l_SPGW_g *= smoothed_valsg
        #fact_l_SPGW_g = fact_l_SPGW_g/fact_l_SPGW_g.max()
        fact_l_SPGWC_A_1 *= smoothed_valsC
        fact_l_SPGWC_A *= smoothed_valsC
        fact_nosmooth *= pdf_SPGWC_A
        p1 *= (p + 1e-20)
    else:
        fact_l_SPGWC_A_2 *= smoothed_valsC
        fact_l_SPGWC_A *= smoothed_valsC
        fact_nosmooth *= pdf_SPGWC_A
        p2 *= (p + 1e-20)
    


    if i==0:
        p_total = (smoothed_vals + 1e-20)
        #p_totalgam = (smoothed_valsg + 1e-20)
        #p_total_2d = (p2d.T + 1e-20)
        #im_total = im
    else:
        p_total *= (smoothed_vals + 1e-20)
        #p_totalgam *= (smoothed_valsg + 1e-20)
        #p_total_2d *= (p2d.T + 1e-20)
        #im_total *= im
    i=i+1
'''
smoothed_vals_p1 = np.zeros(p1.shape)
x_vals = np.linspace(0,29,30)
#x_vals = np.linspace(0,22,23)
for x_position in x_vals:
    kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigmaalt ** 2))
    kernel = kernel / sum(kernel)
    smoothed_vals_p1[int(x_position)] = sum(p1 * kernel)

smoothed_vals_p2 = np.zeros(p2.shape)
x_vals = np.linspace(0,29,30)
#x_vals = np.linspace(0,22,23)
for x_position in x_vals:
    kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigmaalt ** 2))
    kernel = kernel / sum(kernel)
    smoothed_vals_p2[int(x_position)] = sum(p2 * kernel)
'''

#smoothed_valsC = np.zeros(p_total.shape)
#x_vals = np.linspace(0,len(p_total)-1,len(p_total))

#for x_position in x_vals:
#    kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
#    kernel = kernel / sum(kernel)
    #smoothed_valsC[int(x_position)] = sum(pdf_SPGWC_A * kernel)
#    smoothed_valsC[int(x_position)] = sum(p_total * kernel)

prod = np.prod(stacked,axis=0)

#prod = []
#for i in range(len(stacked[0,:])):
#    comb_int = np.prod(stacked[:,i])
#    prod.append(comb_int)

#prod = np.array(prod) 

#np.save("fact_l_SPGW_A",fact_l_SPGW_A)
#np.save("fact_l_SPGW_g",fact_l_SPGW_g)
#np.save(fact_l_SPGWC_A,"fact_l_SPGWC_A")

#plt.title("MPTA Factorised Likelihood: Power Law CRN search")
#plt.hist(fact_l_SPGW_A)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGW_A/fact_l_SPGW_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
#plt.plot(np.linspace(-18,-12,23),fact_l_SPGW_A/fact_l_SPGW_A.max(), color="xkcd:green",linewidth=3,label="MPTA CRN detection")
#plt.fill_between(np.linspace(-18,-12,23), fact_l_SPGW_A/fact_l_SPGW_A.max(),color="xkcd:green",alpha=0.2)
#plt.yscale("log")
#plt.xlabel(r"CRN: $log_{10}A_{CP}$")
#plt.ylabel("PDF")
#plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
#plt.ylim(10e-20,10)
#plt.legend()
#plt.title("FL: SPGW gamma")
#plt.bar(np.linspace(0,7,23),fact_l_SPGW_g,width=0.1)
'''
#plt.title("FL: SPGWC amp")
plt.title("MPTA Factorised Likelihood: Fixed Gamma PL CRN search")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
plt.plot(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(), color="xkcd:green",linewidth=3,label="Factorised Likelihood")
plt.fill_between(np.linspace(-18,-12,23), fact_l_SPGWC_A/fact_l_SPGWC_A.max(),color="xkcd:green",alpha=0.2)
plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
plt.ylim(10e-50,10)
plt.yscale("log")
plt.xlabel(r"CRN: $log_{10}A_{CP}$")
plt.ylabel("PDF")
plt.legend()
#plt.figure()
'''
'''
#plt.title("MPTA Factorised Likelihood: Fixed Gamma PL CRN search (linear)")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
plt.figure(figsize=(4,8))
plt.plot(np.linspace(-18,-12,93),fact_l_SPGWC_A/fact_l_SPGWC_A.max(), color="xkcd:green",linewidth=3,label="MPTA CRN detection")
plt.fill_between(np.linspace(-18,-12,93), fact_l_SPGWC_A/fact_l_SPGWC_A.max(),color="xkcd:green",alpha=0.2)
plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
plt.ylim(10e-5,1.05)
#plt.xlim(-15,-13.75)
plt.xticks(fontsize=16)
plt.yscale("linear")
plt.xlabel(r"CRN: $log_{10}A_{CP}$", fontsize=16)
plt.ylabel("PDF", fontsize=16)
#plt.legend()
#plt.tight_layout()
'''

#plt.title("MPTA Factorised Likelihood: Fixed Gamma PL CRN search (linear)")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
scalefac = fact_l_SPGWC_A/np.sum(fact_l_SPGWC_A)
amps = np.linspace(-18,-12,99)
diff = amps[1]-amps[0]
peakval = np.argmax(scalefac)
plt.figure(figsize=(19.20,10.80))
#plt.plot(np.linspace(-18,-12,99),fact_l_SPGWC_A/(np.sum(fact_l_SPGWC_A)*diff), color="xkcd:green",linewidth=3,label="Gaussian Kernel Smoothed MPTA CRN")
#plt.fill_between(np.linspace(-18,-12,99), fact_l_SPGWC_A/(np.sum(fact_l_SPGWC_A)*diff),color="xkcd:green",alpha=0.2)



bindiff = bins[1]-bins[0]


plt.stairs(p_total/(np.sum(p_total)*bindiff), bins, color='k', zorder=0, linewidth=2)

#plt.title()

#corner.hist2d(p_total/(np.sum(p_total)*bindiff), bins=bins, range=(-18, -12), smooth=True, smooth1d=True, labels=[r"$log_{10}A_{CP}$"])

#plt.axvline(-14.38,color="dimgray",linestyle="--",label="CRN = {:.2f}".format(-14.15))
plt.ylim(1e-11)
#plt.xlim(-15,-13.75)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.xlabel(r"CRN: $\mathrm{log}_{10}\mathrm{A}_\mathrm{CRN, FL}$", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/Amp_FL.png")
plt.clf()


#plt.tight_layout()
'''
plt.figure(figsize=(4,8))
#plt.plot(np.linspace(-18,-12,99),fact_l_SPGWC_A/(np.sum(fact_l_SPGWC_A)*diff), color="xkcd:green",linewidth=3,label="Gaussian Kernel Smoothed MPTA CRN")
#plt.fill_between(np.linspace(-18,-12,99), fact_l_SPGWC_A/(np.sum(fact_l_SPGWC_A)*diff),color="xkcd:green",alpha=0.2)

bindiffgam = binsgam[1]-binsgam[0]
plt.stairs(p_totalgam/(np.sum(p_totalgam)*bindiffgam), binsgam, color='k', zorder=0, linewidth=2, label = "Factorised likelihood MPTA CRN gamma")

#plt.axvline(-14.28,color="dimgray",linestyle="--",label="CRN = {:.2f}".format(-14.28))
#plt.ylim(10e-8)
#plt.xlim(-15,-13.75)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.yscale("log")
plt.xlabel(r"CRN: $\gamma_{CP}$", fontsize=20)
plt.ylabel("PDF", fontsize=20)
plt.legend(fontsize=16)

#adjusted_prob = p_total_2d/(np.sum(p_total_2d))
#adjusted_prob = p_total_2d/(np.sum(p_total_2d)*np.diff([xbins2d,ybins2d]).mean())

'''
'''
newxbins2d = np.linspace(-18,-12,100)
newybins2d = np.linspace(0,7,100)
#newxbins2d = newxbins2d[:-1]+(0.5*(newxbins2d[1] -newxbins2d[0]))
#newybins2d = newybins2d[:-1]+(0.5*(newybins2d[1] -newybins2d[0]))

#newbins2d = xbins2d + 0.5*(xbins2d[1]-xbins2d[0])

X, Y = np.meshgrid(newxbins2d,newybins2d)
#multix = np.array([newxbins2d]*30)
#multiy = np.array([newybins2d]*30)

bindiffgam = binsgam[1]-binsgam[0]
pgam_adjust = p_totalgam/(np.sum(p_totalgam)*bindiffgam)
p_adjust = p_total/(np.sum(p_total)*bindiff)

#data = np.vstack([p_adjust,pgam_adjust])
#kde = gaussian_kde(data)

#Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))
#plt.figure(figsize=(4,8))
#plt.imshow(Z.reshape(X.shape),
#           origin='lower', aspect='auto',
#           extent=[-18,-12, 0,7],
#           cmap='Blues')
#cb = plt.colorbar()
#cb.set_label("density")


ampsample = np.random.choice(newxbins2d,size=10000, p=p_adjust/np.sum(p_adjust))
gammasample = np.random.choice(newybins2d,size=10000, p=pgam_adjust/np.sum(pgam_adjust))
corner.corner(np.array([ampsample,gammasample]).T, bins=100, range=([-18, -12], [0, 7]), smooth=True, smooth1d=True, labels=[r"$log_{10}A_{CP}$", r"$\gamma_{CP}$"])

#plt.show()

#adjusted_prob = p_total_2d/(np.sum(p_total_2d)*np.diff([X,Y])[0].mean())
#corner.corner(np.array([X.flatten(),Y.flatten()]).T, bins=30, range=([-18, -12], [0, 7]), weights=adjusted_prob, smooth=True, smooth1d=True, labels=[r"$log_{10}A_{CP}$", r"$\gamma_{CP}$"])
#corner.corner(np.array([X.flatten(),Y.flatten()]).T, bins=20, range=([-18, -12], [0, 7]), weights=adjusted_prob.flatten(), labels=[r"$log_{10}A_{CP}$", r"$\gamma_{CP}$"])
'''

#fig, ax = plt.subplots(figsize=(4,8))
#ax.hist2d(p_total/(np.sum(p_total)*bindiff), p_totalgam/(np.sum(p_totalgam)*bindiffgam), bins=[bins,binsgam])
#ax.set_xlabel(r"CRN: $log_{10}A_{CP}$", fontsize=20)
#ax.set_ylabel(r"CRN: $\gamma_{CP}$", fontsize=20)
#ax.set_title("2D posterior")
'''


scalefac = fact_l_SPGWC_A_1/np.sum(fact_l_SPGWC_A_1)
amps = np.linspace(-18,-12,99)
diff = amps[1]-amps[0]
peakval = np.argmax(scalefac)
plt.figure(figsize=(4,8))
#plt.plot(np.linspace(-18,-12,99),fact_l_SPGWC_A_1/(np.sum(fact_l_SPGWC_A_1)*diff), color="xkcd:green",linewidth=3,label="Gaussian Kernel Smoothed MPTA CRN")
#plt.fill_between(np.linspace(-18,-12,99), fact_l_SPGWC_A_1/(np.sum(fact_l_SPGWC_A_1)*diff),color="xkcd:green",alpha=0.2)

plt.figure(figsize=(4,8))
bindiff = bins[1]-bins[0]
plt.stairs(p1/(np.sum(p1)*bindiff), bins, color='k', zorder=0, linewidth=2, label = "Factorised likelihood MPTA CRN (1/2)")

plt.axvline(-14.38,color="dimgray",linestyle="--",label="CRN = {:.2f}".format(-14.38))
plt.ylim(0.001)
#plt.xlim(-15,-13.75)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("linear")
plt.xlabel(r"CRN: $log_{10}A_{CP}$", fontsize=20)
plt.ylabel("PDF", fontsize=20)
plt.legend(fontsize=16)


#scalefac = smoothed_vals_p2/np.sum(smoothed_vals_p2)
amps = np.linspace(-18,-12,30)
diff = amps[1]-amps[0]
#peakval = np.argmax(scalefac)
plt.figure(figsize=(4,8))
#plt.plot(np.linspace(-18,-12,30),smoothed_vals_p2/(np.sum(smoothed_vals_p2)*diff), color="xkcd:green",linewidth=3,label="Gaussian Kernel Smoothed MPTA CRN")
#plt.fill_between(np.linspace(-18,-12,30), smoothed_vals_p2/(np.sum(smoothed_vals_p2)*diff),color="xkcd:green",alpha=0.2)

bindiff = bins[1]-bins[0]
plt.stairs(p2/(np.sum(p2)*bindiff), bins, color='k', zorder=0, linewidth=2, label = "Factorised likelihood MPTA CRN (2/2)")

#plt.axvline(amps[peakval],color="dimgray",linestyle="--",label="CRN = {:.2f}".format(amps[peakval]))
plt.axvline(-14.38,color="dimgray",linestyle="--",label="CRN = {:.2f}".format(-14.38))
plt.ylim(0.001)
#plt.xlim(-15,-13.75)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("linear")
plt.xlabel(r"CRN: $log_{10}A_{CP}$", fontsize=20)
plt.ylabel("PDF", fontsize=20)
plt.legend(fontsize=16)
'''







#plt.plot(np.linspace(-18,-12,99),fact_l_SPGWC_A_1/fact_l_SPGWC_A_1.max(), color="xkcd:green",linewidth=3,label="Gaussian Kernel Smoothed MPTA CRN")
#plt.fill_between(np.linspace(-18,-12,99), fact_l_SPGWC_A_1/fact_l_SPGWC_A_1.max(),color="xkcd:green",alpha=0.2)



#plt.plot(np.linspace(-18,-12,49),prod/prod.max(), color="xkcd:green",linewidth=3,label="MPTA CRN detection")
#plt.fill_between(np.linspace(-18,-12,49), prod/prod.max(),color="xkcd:green",alpha=0.2)
#plt.hist(prod/prod.max(), bins=100, range=(-18, -12), density=True, alpha=0.6, histtype='step')
#plt.stairs(prod / np.sum(prod) / np.mean(np.diff(bins)), bins, color='k', zorder=0, linewidth=2)
#plt.stairs(p_total / np.sum(p_total) / np.mean(np.diff(bins)), bins, color='k', zorder=0, linewidth=2)
#plt.stairs(p_total / p_total.max(), bins, color='k', zorder=0, linewidth=2, label = "Factorised likelihood MPTA CRN")

'''
plt.title("Sample 1")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
plt.plot(np.linspace(-18,-12,23),fact_l_SPGWC_A_1/fact_l_SPGWC_A_1.max(), color="xkcd:green",linewidth=3,label="Factorised Likelihood")
plt.fill_between(np.linspace(-18,-12,23), fact_l_SPGWC_A_1/fact_l_SPGWC_A_1.max(),color="xkcd:green",alpha=0.2)
plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
plt.ylim(10e-9,2)
plt.yscale("log")
plt.xlabel(r"CRN: $log_{10}A_{CP}$")
plt.ylabel("PDF")
plt.legend()
plt.figure()

plt.title("Sample 2")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
plt.plot(np.linspace(-18,-12,23),fact_l_SPGWC_A_2/fact_l_SPGWC_A_2.max(), color="xkcd:green",linewidth=3,label="Factorised Likelihood")
plt.fill_between(np.linspace(-18,-12,23), fact_l_SPGWC_A_2/fact_l_SPGWC_A_2.max(),color="xkcd:green",alpha=0.2)
plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
plt.ylim(10e-9,2)
plt.yscale("log")
plt.xlabel(r"CRN: $log_{10}A_{CP}$")
plt.ylabel("PDF")
plt.legend()
plt.figure()
'''
#plt.show()
