#!/bin/bash

psr=$1

#Collect data

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


cd $psr

#Change data into correct format and dlyfix

for arch in $(ls *ar);
do
    echo ${arch}
    pam -mpT ${arch}
    /fred/oz002/users/mmiles/dlyfix/dlyfix ${arch} -u . -o ${arch%.ar}.dly
done

rm J*ar
#Apply the correct dispersion measure to the data
for arch in $(ls J*dly); 
do 
    echo ${arch} 
    pdmp -g /ps -bf ${arch} 
    dm_temp=$(cat pdmp.best | head -n 9 |tail -n 1 | awk '{print($1)}')
    pam -md ${dm_temp} ${arch} --update_dm
    pam -D ${arch} -e dly_D
done

#Create grand data object

seed_dir="/fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated/"
psradd -PT $(psrstat -Q -c snr $(awk '{print $1}' ${seed_dir}/${psr}.tim | uniq | cut -d "/" -f 9 | grep J | rev | cut -f 2- -d '.' | rev | awk -v awkpsr=${psr} '{print "*"$0"*.dly_D"}') | sort -g -k 2 | tail -n $(echo $(( $(ls J*fluxcal.dly_D | wc -l) /1))) | awk '{print $1}') -o ${psr}_grand_T.dly_D



