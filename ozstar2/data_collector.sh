#!/bin/bash

psr=$1
#for psr in $(ls -d J*); do 
echo $psr 
for dir in $(ls -d /fred/oz005/timing_processed/$psr/20*); do 
    if [[ -f $(echo $dir/*/decimated/*zap*928ch1p[123456789]t.ar) ]]; then 
        cp $dir/*/decimated/*zap*928ch1p[123456789]t.ar $psr/; 
    elif [[ -f $(echo $dir/*/decimated/*zap*928ch_1p_1t.ar) ]]; then
        cp $dir/*/decimated/*zap*928ch_1p_1t.ar $psr/;
    else
        echo "${dir} doesn't contain file"
        echo "${dir}" >> ${psr}/${psr}_missing.list
    fi; 
done; 
#done




