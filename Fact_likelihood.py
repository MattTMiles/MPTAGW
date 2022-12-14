import numpy as np
import matplotlib.pyplot as plt
import bilby
import random

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc_SPGWC_WN/live_400"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_trusted_noise_281022.txt"
#psr_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list_noJ1756.txt"
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))   
to_use = open(psr_list).readlines()
random.shuffle(to_use)
for i, pulsar in enumerate(to_use):
    psrname = pulsar.strip("\n")
    print(psrname)
    try:
        psr_SPGW = gw_dir + "/" + psrname + "_SPGW"
        psr_SPGWC = gw_dir + "/" + psrname + "_SPGWC"

        result_SPGW = bilby.result.read_in_result(psr_SPGW+"/SPGW_result.json")
        result_SPGWC = bilby.result.read_in_result(psr_SPGWC+"/SPGWC_result.json")

        posts_SPGW_A = result_SPGW.posterior["log10_A_gw"].values
        posts_SPGW_g = result_SPGW.posterior["gamma_gw"].values

        posts_SPGWC_A = result_SPGWC.posterior["log10_A_gw"].values

        pdf_SPGW_A = np.histogram(posts_SPGW_A,bins=np.linspace(-18,-12,24),density=True)[0] + 1e-20
        pdf_SPGW_g = np.histogram(posts_SPGW_g,bins=np.linspace(0,7,24),density=True)[0] + 1e-20
        
        #pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,94),density=True)[0] + 1e-20
        pdf_SPGWC_A = np.histogram(posts_SPGWC_A,bins=np.linspace(-18,-12,24),density=True)[0] + 1e-20
    except:
        continue
    
    FWHM = 6
    #FWHM = 24
    sigma = fwhm2sigma(FWHM)
    smoothed_vals = np.zeros(pdf_SPGW_A.shape)
    x_vals = np.linspace(0,22,23)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[int(x_position)] = sum(pdf_SPGW_A * kernel)

    smoothed_valsg = np.zeros(pdf_SPGW_g.shape)
    x_vals = np.linspace(0,22,23)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_valsg[int(x_position)] = sum(pdf_SPGW_g * kernel)

    smoothed_valsC = np.zeros(pdf_SPGWC_A.shape)
    #x_vals = np.linspace(0,92,93)
    x_vals = np.linspace(0,22,23)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_valsC[int(x_position)] = sum(pdf_SPGWC_A * kernel)
        #plt.plot()

    if i == 0:
        #fact_l_SPGW_A = pdf_SPGW_A
        fact_l_SPGW_A = smoothed_vals
        fact_l_SPGW_g = smoothed_valsg
        fact_l_SPGWC_A = smoothed_valsC
        fact_l_SPGWC_A_1 = smoothed_valsC
        fact_nosmooth = pdf_SPGWC_A
    elif i==1:
        fact_l_SPGWC_A_2 = smoothed_valsC
    elif i%2 == 0:
        #fact_l_SPGW_A *= pdf_SPGW_A
        #fact_l_SPGW_A *= smoothed_vals
        #fact_l_SPGW_A = np.histogram(posts_SPGW_A,bins=np.linspace(-18,-12,100),density=True)[0]
        #fact_l_SPGW_g *= smoothed_valsg
        #fact_l_SPGW_g = fact_l_SPGW_g/fact_l_SPGW_g.max()
        fact_l_SPGWC_A_1 *= smoothed_valsC
        fact_l_SPGWC_A *= smoothed_valsC
        fact_nosmooth *= pdf_SPGWC_A
    else:
        fact_l_SPGWC_A_2 *= smoothed_valsC
        fact_l_SPGWC_A *= smoothed_valsC
        fact_nosmooth *= pdf_SPGWC_A


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
plt.title("MPTA Factorised Likelihood: Fixed Gamma PL CRN search (linear)")
#plt.plot(fact_l_SPGWC_A)
#plt.bar(np.linspace(-18,-12,95),fact_l_SPGWC_A,width=0.1)
#plt.bar(np.linspace(-18,-12,23),fact_l_SPGWC_A/fact_l_SPGWC_A.max(),width=0.2,color="xkcd:green",alpha=0.3)
plt.plot(np.linspace(-18,-12,93),fact_l_SPGWC_A/fact_l_SPGWC_A.max(), color="xkcd:green",linewidth=3,label="MPTA CRN detection")
plt.fill_between(np.linspace(-18,-12,93), fact_l_SPGWC_A/fact_l_SPGWC_A.max(),color="xkcd:green",alpha=0.2)
plt.axvline(-14.5,color="dimgray",linestyle="--",label="IPTA result")
plt.ylim(10e-5,1.05)
plt.yscale("linear")
plt.xlabel(r"CRN: $log_{10}A_{CP}$")
plt.ylabel("PDF")
plt.legend()
plt.figure()
'''
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
plt.show()
