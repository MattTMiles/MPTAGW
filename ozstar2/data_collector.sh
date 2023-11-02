#!/bin/bash

psr=$1
#for psr in $(ls -d J*); do 
echo $psr 
for dir in $(ls -d /fred/oz005/timing_processed/PTA/$psr/20*); do 
    if [[ -f $(echo $dir/*/*/decimated/*zap.*928chI.fluxcal.ar) ]]; then 
        cp $dir/*/*/decimated/*zap.*928chI.fluxcal.ar $psr/; 
    elif [[ -f $(echo $dir/*/*/decimated/*zap.*928chTS.fluxcal.ar) ]]; then
        cp $dir/*/*/decimated/*zap.*928chTS.fluxcal.ar $psr/;
    else
        echo "${dir} doesn't contain file"
        echo "${dir}" >> ${psr}/${psr}_missing.list
    fi; 
done; 
#done




