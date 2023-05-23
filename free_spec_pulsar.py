import numpy as np
import matplotlib.pyplot as plt
import bilby
import random

import seaborn as sns
import pandas as pd

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/SPGW"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/post_gauss_check.list"

#cross_corr_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/fixed_amp_500/"


psr_list = list(open(psr_list).readlines())

psr_list = [ x.strip("\n") for x in psr_list ]
#psr_list = ["J1918-0642"]
i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0

bin_number = 100

free_bins = []

#psr_list = ["J2241-5236"]
for psr in psr_list:
    print(psr)

    try:
        free_bins = []
        psr_FREE_SPGWC = gw_dir + "/" + psr + "/" + psr + "_FREE_SPGW600_ER"
        res_FREE_SPGWC = bilby.result.read_in_result(psr_FREE_SPGWC+"/FREE_SPGW600_ER_result.json")

        i=-1
        for parlab in res_FREE_SPGWC.parameter_labels:
            if parlab.startswith("gw"):
                i +=1
                posts = res_FREE_SPGWC.posterior[parlab].values
                free_bins.append(posts)
    
        
        newbins = np.linspace(-9, -4, 10)

        T = 122448047.42001152
        Tyear = 3.8828021125067073704

        f_xaxis = np.linspace(1,10,10)
        freal = f_xaxis/T

        pwl = np.sqrt((10**-14.26)**2 / 12.0 / np.pi**2 * (1/(86400*365.2425))**(4.3333-3) * freal**(-4.3333) * freal[0])

        fig = plt.figure()
        axes = fig.add_subplot(111)

        axes.violinplot(free_bins, positions=f_xaxis)
        axes.plot(f_xaxis, np.log10(pwl),linestyle="--",label = r"$\log_{10}A = -14.26; \gamma=4.333$")

        #axes.plot(f_xaxis,p_spec)
        axes.set_xlabel("1/T (3.88 yrs)")
        axes.set_ylabel(r"$\log_{10}(\rho/s)$")
        axes.set_xscale("log")
        #plt.show()
        plt.savefig(psr_FREE_SPGWC+"/"+psr+"_FREE_SPEC_10_bins.png")
    


    except:
        print("Not finished")
        continue


#p_total_7_max=np.max(p_total_7)

#p_total_7=np.exp(p_total_7-p_total_7_max)

#newbins = np.linspace(-1,1,bin_number)

#bindiff = bins[1]-bins[0]

#free_bins_norm = [ f/(np.sum(f)*bindiff) for f in free_bins ]

'''
p_total_1 = p_total_1/(np.sum(p_total_1)*bindiff)
p_total_2 = p_total_2/(np.sum(p_total_2)*bindiff)
p_total_3 = p_total_3/(np.sum(p_total_3)*bindiff)
p_total_4 = p_total_4/(np.sum(p_total_4)*bindiff)
p_total_5 = p_total_5/(np.sum(p_total_5)*bindiff)
if i_6 > 0:
    p_total_6 = p_total_6/(np.sum(p_total_6)*bindiff)
if i_7 > 0:
    p_total_7 = p_total_7/(np.sum(p_total_7)*bindiff)

bin_list = []

newbins = np.linspace(-9, -4, 30)

prob_free = [ np.random.choice(newbins, size=100, p=fp/np.sum(fp)) for fp in free_bins_norm ]

T = 122448047.42001152
Tyear = 3.8828021125067073704

f_xaxis = np.linspace(1,30,30)

fig = plt.figure()
axes = fig.add_subplot(111)

axes.violinplot(prob_free, positions=f_xaxis)

axes.set_xlabel("Hz")
axes.set_ylabel(r"$\log_{10}(\rho/s)$")
#axes.set_xscale()
plt.show()

angle_linspace = np.linspace(0,180,1000)
angle_linspace_rad = angle_linspace*(np.pi/180)
hd = 0.5 - (0.25*((1-np.cos(angle_linspace_rad))/2)) + ((1.5)*((1-np.cos(angle_linspace_rad))/2))*np.log((1-np.cos(angle_linspace_rad))/2)

#bin_list = [np.array(p_corr_1), np.array(p_corr_2), np.array(p_corr_3), np.array(p_corr_4), np.array(p_corr_5), np.array(p_corr_6), np.array(p_corr_7)]
bin_df = pd.DataFrame(bin_list)

angles = np.linspace(0, np.pi,plotting_bins)

fig = plt.figure()
axes = fig.add_subplot(111)

#plt.plot(angle_linspace_rad*(7/angle_linspace_rad.max()), hd, color="black", linestyle="--")
plt.plot(angle_linspace_rad, hd, color="black", linestyle="--")

#for i, post in enumerate(bin_list):
axes.violinplot(bin_list,positions=angles)
#ax = sns.violinplot(bin_list)

plt.xticks(ticks=[0,np.pi/6,2*np.pi/6,3*np.pi/6,4*np.pi/6,5*np.pi/6,6*np.pi/6], labels=["$0$","$30$","$60$","$90$","$120$","$150$","$180$"])
#axes.xaxis.set_ticklabels(ticklabels=["$0$","$30$","$60$","$90$","$120$","$150$","$180$"])

#sns.violinplot(bin_df, color="xkcd:aqua")
#plt.plot(angle_linspace_rad*(7/angle_linspace_rad.max()),hd, linestyle="--")

plt.show()
'''



    



