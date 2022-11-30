import bilby
import os
import glob
import json
import sys
import numpy as np
import argparse


active_dir = "/fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models"
noise_list = active_dir + "/MPTA_noise_models.txt"

parser = argparse.ArgumentParser(description="Evidence comparer and decider.")
parser.add_argument("-pulsars", dest="pulsars", nargs="+", help="List of pulsars to do evidence comparison on. Must be in a ascii file separated by lines.",required = True)
parser.add_argument("-outfile", dest="outfile", help="Path and name of the outfile (.json extension)",required = True)
args = parser.parse_args()

pulsar_list = args.pulsars
outfile = args.outfile

psrlist=None
if pulsar_list is not None and pulsar_list != "":
    if type(pulsar_list[0]) != list:
        psrlist=[ x.strip("\n") for x in open(str(pulsar_list[0])).readlines() ]
    else:
        psrlist = list(pulsar_list)

os.chdir(active_dir)

model_list = open(noise_list).readlines()

mpta_models = {}

for pulsar in psrlist:
    pulsar = pulsar.strip("\n")
    print(pulsar)
    
    psrdir = glob.glob(pulsar+"*")[0]

    res = bilby.result.read_in_result(glob.glob(psrdir+"/*json")[0]).log_evidence

    modelname = [ model for model in model_list if pulsar in model ][0].strip("\n")

    mpta_models[modelname] = res

with open(outfile,"a+") as outFile:
    json.dump(mpta_models,outFile,indent=4)

