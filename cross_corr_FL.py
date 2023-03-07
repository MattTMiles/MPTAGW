import numpy as np
import matplotlib.pyplot as plt
import bilby
import random

import seaborn as sns
import pandas as pd

gw_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW"
psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/factorised_likelihood.list"

cross_corr_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/cross_corrs/fixed_amp_500"

pulsar_pair_list = "/fred/oz002/users/mmiles/MPTA_GW/pair_bins_new.txt"

pp_list = list(open(pulsar_pair_list).readlines())

pp_list = [ x.strip("\n") for x in pp_list ]

i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0

bin_number = 100

for pair in pp_list:
    pairname, pairbin = pair.split()
    if not "J1327" in pairname:
    #if not "J1327" in pairname or "J1902-5105" in pairname or "J2322-2650" in pairname or "J1036-8317" in pairname:
        print(pairname, pairbin)

        try:
            res_pair = bilby.result.read_in_result(cross_corr_dir + "/" + pairname + "/" + pairname + "_result.json")
            corr_posts = res_pair.posterior["gw_single_orf_bin"].values

            p, bins, patches = plt.hist(corr_posts, bins=bin_number, range=(-1, 1), density=True, alpha=0.6, histtype='step')
            

            #if pairbin == "3":
            #    p, bins, patches = plt.hist(corr_posts, bins=30, range=(-1, 1), density=True, alpha=0.6, histtype='step')

            ind = np.argmax(p)
            centres = bins[0:-1] + np.diff(bins)
            #print(centres[ind])



            if int(pairbin) == 1:
                if i_1 == 0:
                    #p_total_1 = (p + 1e-20)
                    p_total_1 = (p)
                    i_1 += 1
                else:
                    i_1 += 1
                    #p_total_1 *= (p + 1e-20)
                    p_total_1 *= (p)

            elif int(pairbin) == 2:
                if i_2 == 0:
                    p_total_2 = (p)
                    i_2 += 1
                else:
                    i_2 += 1
                    p_total_2 *= (p)

            elif int(pairbin) == 3:
                if i_3 == 0:
                    p_total_3 = (p)
                    i_3 += 1
                else:
                    i_3 += 1
                    p_total_3 *= (p)

            elif int(pairbin) == 4:
                if i_4 == 0:
                    p_total_4 = (p)
                    i_4 += 1
                else:
                    i_4 += 1
                    p_total_4 *= (p)

            elif int(pairbin) == 5:
                if i_5 == 0:
                    p_total_5 = (p)
                    i_5 += 1
                else:
                    i_5 += 1
                    p_total_5 *= (p)

            elif int(pairbin) == 6:
                if i_6 == 0:
                    p_total_6 = (p)
                    i_6 += 1
                else:
                    i_6 += 1
                    p_total_6 *= (p)

            elif int(pairbin) == 7:
                if i_7 == 0:
                    #p_total_7 = np.log(p)
                    p_total_7 = (p)
                    i_7 += 1
                else:
                    i_7 += 1
                    #p_total_7 += np.log(p )
                    p_total_7 *= (p)

        except:
            print("Not finished")
            continue


#p_total_7_max=np.max(p_total_7)

#p_total_7=np.exp(p_total_7-p_total_7_max)

newbins = np.linspace(-1,1,bin_number)

bindiff = bins[1]-bins[0]
p_total_1 = p_total_1/(np.sum(p_total_1)*bindiff)
p_total_2 = p_total_2/(np.sum(p_total_2)*bindiff)
p_total_3 = p_total_3/(np.sum(p_total_3)*bindiff)
p_total_4 = p_total_4/(np.sum(p_total_4)*bindiff)
p_total_5 = p_total_5/(np.sum(p_total_5)*bindiff)
p_total_6 = p_total_6/(np.sum(p_total_6)*bindiff)
p_total_7 = p_total_7/(np.sum(p_total_7)*bindiff)

p_corr_1 =  np.random.choice(newbins,size=100, p=p_total_1/np.sum(p_total_1))
p_corr_2 =  np.random.choice(newbins,size=100, p=p_total_2/np.sum(p_total_2))
p_corr_3 =  np.random.choice(newbins,size=100, p=p_total_3/np.sum(p_total_3))
p_corr_4 =  np.random.choice(newbins,size=100, p=p_total_4/np.sum(p_total_4))
p_corr_5 =  np.random.choice(newbins,size=100, p=p_total_5/np.sum(p_total_5))
p_corr_6 =  np.random.choice(newbins,size=100, p=p_total_6/np.sum(p_total_6))
p_corr_7 =  np.random.choice(newbins,size=100, p=p_total_7/np.sum(p_total_7))


angle_linspace = np.linspace(0,180,1000)
angle_linspace_rad = angle_linspace*(np.pi/180)
hd = 0.5 - (0.25*((1-np.cos(angle_linspace_rad))/2)) + ((1.5)*((1-np.cos(angle_linspace_rad))/2))*np.log((1-np.cos(angle_linspace_rad))/2)

bin_list = [np.array(p_corr_1), np.array(p_corr_2), np.array(p_corr_3), np.array(p_corr_4), np.array(p_corr_5), np.array(p_corr_6), np.array(p_corr_7)]
bin_df = pd.DataFrame(bin_list)

angles = np.linspace(0, np.pi,7)

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



    



