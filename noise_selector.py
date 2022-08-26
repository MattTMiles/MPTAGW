import bilby
import os
import glob
import json
import sys
import numpy as np

pulsar_list = "/fred/oz002/users/mmiles/MPTA_GW/MPTA_pulsar_list.txt"
enterprise_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out"

pref_models_list = '/fred/oz002/users/mmiles/MPTA_GW/enterprise/preferred_model.json'

#pref = []
#for line in open(pref_models_list, "r").readlines():
#    line = line.strip("\n")
#    pref.

with open(pref_models_list) as json_file:
    data = json.load(json_file)

models = list(data)

noise_models = {}

for model in models:
    result = bilby.result.read_in_result(glob.glob(enterprise_dir+"/"+model+"/*json")[0])
    for parlab in result.parameter_labels:
        val = result.get_one_dimensional_median_and_error_bar(parlab).median
        if "dm_gp_log10_A" in parlab:
            val = val - np.log10(2.41e-16*1.4e9**2/np.sqrt(12*np.pi**2))
        noise_models[parlab] = val

with open("noise_values.json", "a+") as outfile:
    json.dump(noise_models, outfile, indent=4)

    