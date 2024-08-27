import numpy as np
import matplotlib.pyplot as plt
import bilby
import random

import seaborn as sns
import pandas as pd
import json
import os

psr_list = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt"

cross_corr_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/"

pulsar_pair_list = "/fred/oz002/users/mmiles/MPTA_GW/pair_bins_7.txt"

pp_list = list(open(pulsar_pair_list).readlines())

pp_list = [ x.strip("\n") for x in pp_list ]


i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0

bin_number = 40

scale_count = 0

for pair in pp_list:
    pairname, pairbin = pair.split()
    if not "J1024" in pairname or "J1216" in pairname or "J1327" in pairname:
    #if "J1909"  in pairname or "J2241" in pairname or "J1902" in pairname or "J1231" in pairname or "J1022" in pairname:
    #if "J1327" not in pairname and "J1902-5105" not in pairname and "J2322-2650" not in pairname and "J1036-8317" not in pairname and "J1024-0719" not in pairname:

        try:
            print(pairname, pairbin)
            res_pair = pd.read_json(json.load(open(cross_corr_dir + "/" + pairname + "/" + pairname + "_final_res.json")))
            #res_pair = bilby.result.read_in_result(cross_corr_dir + "/" + pairname + "/" + pairname + "_result.json")
            short_df = res_pair[(-14.6 < res_pair["gw_bins_log10_A"]) * (res_pair["gw_bins_log10_A"] < -14.0)]
            corr_posts = short_df["gw_single_orf_bin_0"].values

            p, bins, patches = plt.hist(corr_posts, bins=bin_number, range=(-1, 1), density=True, alpha=0.6, histtype='step')
            

            scalep = p/p.max()

            if not scalep.max() > 4*np.median(scalep):
                
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
                        if not 0 in p:
                            i_1 += 1
                            #p_total_1 *= (p + 1e-20)
                            p_total_1 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)


                elif int(pairbin) == 2:
                    if i_2 == 0:
                        p_total_2 = (p)
                        i_2 += 1
                    else:
                        if not 0 in p:
                            i_2 += 1
                            p_total_2 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)

                elif int(pairbin) == 3:
                    if i_3 == 0:
                        p_total_3 = (p)
                        i_3 += 1
                    else:
                        if not 0 in p:
                            i_3 += 1
                            p_total_3 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)

                elif int(pairbin) == 4:
                    if i_4 == 0:
                        p_total_4 = (p)
                        i_4 += 1
                    else:
                        if not 0 in p:
                            i_4 += 1
                            p_total_4 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)

                elif int(pairbin) == 5:
                    if i_5 == 0:
                        p_total_5 = (p)
                        i_5 += 1
                        #print(p_total_5)
                    else:
                        if not 0 in p:
                            i_5 += 1
                            p_total_5 *= (p)
                            #print(p_total_5)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)

                elif int(pairbin) == 6:
                    if i_6 == 0:
                        p_total_6 = (p)
                        i_6 += 1
                    else:
                        if not 0 in p:
                            i_6 += 1
                            p_total_6 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)

                elif int(pairbin) == 7:
                    if i_7 == 0:
                        #p_total_7 = np.log(p)
                        p_total_7 = (p)
                        i_7 += 1
                    else:
                        if not 0 in p:
                            i_7 += 1
                            #p_total_7 += np.log(p )
                            p_total_7 *= (p)
                        else:
                            print("!! WARNING: run is broken for "+pairname+" !!")
                            print("!! Removing "+pairname+" !!")
                            #os.system("rm -rf "+cross_corr_dir+"/"+pairname)
            
            else:
                print("!!!!! This might be wrong: "+pairname+" !!!!!")
                scale_count +=1
                os.system("touch "+pairname+"_suspect")
        except:
            print("Not finished")
            
            continue


#p_total_7_max=np.max(p_total_7)

#p_total_7=np.exp(p_total_7-p_total_7_max)

samplesize = 80

newbins = np.linspace(-1,1,bin_number)

bindiff = bins[1]-bins[0]
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

p_corr_1 =  np.random.choice(newbins,size=samplesize, p=p_total_1/np.sum(p_total_1))
bin_list += [np.array(p_corr_1)]
p_corr_2 =  np.random.choice(newbins,size=samplesize, p=p_total_2/np.sum(p_total_2))
bin_list += [np.array(p_corr_2)]
p_corr_3 =  np.random.choice(newbins,size=samplesize, p=p_total_3/np.sum(p_total_3))
bin_list += [np.array(p_corr_3)]
p_corr_4 =  np.random.choice(newbins,size=samplesize, p=p_total_4/np.sum(p_total_4))
bin_list += [np.array(p_corr_4)]
p_corr_5 =  np.random.choice(newbins,size=samplesize, p=p_total_5/np.sum(p_total_5))
bin_list += [np.array(p_corr_5)]
if i_6 > 0:
    p_corr_6 =  np.random.choice(newbins,size=samplesize, p=p_total_6/np.sum(p_total_6))
    bin_list += [np.array(p_corr_6)]
if i_7 > 0:
    p_corr_7 =  np.random.choice(newbins,size=samplesize, p=p_total_7/np.sum(p_total_7))
    bin_list += [np.array(p_corr_7)]


plotting_bins = 0
if i_1 > 0:
    plotting_bins +=1
if i_2 > 0:
    plotting_bins +=1
if i_3 > 0:
    plotting_bins +=1
if i_4 > 0:
    plotting_bins +=1
if i_5 > 0:
    plotting_bins +=1
if i_6 > 0:
    plotting_bins +=1
if i_7 > 0:
    plotting_bins +=1

angle_linspace = np.linspace(0,180,1000)
angle_linspace_rad = angle_linspace*(np.pi/180)
hd = 0.5 - (0.25*((1-np.cos(angle_linspace_rad))/2)) + ((1.5)*((1-np.cos(angle_linspace_rad))/2))*np.log((1-np.cos(angle_linspace_rad))/2)

#bin_list = [np.array(p_corr_1), np.array(p_corr_2), np.array(p_corr_3), np.array(p_corr_4), np.array(p_corr_5), np.array(p_corr_6), np.array(p_corr_7)]
bin_df = pd.DataFrame(bin_list)

angles = np.linspace(0, np.pi,plotting_bins)

print("scale count: {0}".format(scale_count))

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
plt.savefig("/fred/oz002/users/mmiles/MPTA_GW/PAIRWISE_NEW/pairwise_cross_corr_temp.png")
plt.show()



    



