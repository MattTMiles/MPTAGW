#!/bin/bash

for arch in $(ls *ar);
do
    num1=$(psrstat -c length $arch | awk -F\= '{print $NF}'); 
    num2=3000; 
    if [[ 1 -eq "$(echo "${num1} > ${num2}" |bc -l)" ]]; 
    then 
        echo "${arch}_too_big; splitting it";
        median=$(psredit -Qq -c length *fluxcal.dly | sort -n | awk '{arr[NR]=$1} END { if (NR%2==1) print arr[(NR+1)/2]; else print (arr[NR/2]+arr[NR/2+1])/2}')

        splitby=$((${num1%.*}/${median%.*}))
        subints=$(psredit -Qq -c nsubint ${arch})

        nsubint=$((($subints+$splitby+1)/$splitby))

        psrsplit -n ${nsubint} ${arch}
        rm ${arch}

        for subarch in $(ls ${arch%.ar}*ar);
        do
            pam --setnchn 32 ${subarch} -e 32chn
            rm ${subarch}
            pam -mT ${subarch%.ar}.32chn
            /fred/oz002/users/mmiles/dlyfix/dlyfix ${subarch%.ar}.32chn -u . -o ${subarch%.ar}.dly
            rm ${subarch%.ar}.32chn
        done
    else 
        pam --setnchn 32 ${arch} -e 32chn

        rm ${arch}

        pam -mT ${arch%.ar}.32chn

        /fred/oz002/users/mmiles/dlyfix/dlyfix ${arch%.ar}.32chn -u . -o ${arch%.ar}.dly

        rm ${arch%.ar}.32chn
    fi; 
done




