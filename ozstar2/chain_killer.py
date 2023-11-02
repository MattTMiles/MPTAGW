# Quick code to kill a MCMC directory if it has stalled chains

import numpy as np 
import sys
import os

dirname = sys.argv[1]

chainfile = dirname + '/chain_1.txt'

if not os.path.exists(chainfile):
    
    print("chain_1.txt doesn't exist in this directory")

else:
    #chain = np.loadtxt(chainfile)
    likelihood = os.popen("cat "+chainfile+" | awk 'END{print $(NF-2)}'").read().strip("\n")

    # This will be dependent on if it's a hypermodel run or not, make sure to check this
    #if np.isinf(chain[-1][-3]):
    if likelihood == "-inf":
        # Gets rid of the stalled chain, the auto-submit sequence will remake this
        print("Removing chain directory: " + dirname)
        os.system("rm -rf " + dirname)
