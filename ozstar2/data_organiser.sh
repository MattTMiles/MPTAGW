#!/bin/bash

for arch in $(ls *ar);
do
    num1=$(psrstat -c length $arch | awk -F\= '{print $NF}'); 
    num2=3000; 
    if [[ 1 -eq "$(echo "${num1} > ${num2}" |bc -l)" ]]; 
    then 
        echo "${arch}_too_big; splitting it";
        
        #splitby=$((${num1}/${num2}))
        
        #if [[ 1 -eq "$(echo "${splitby} > 1" |bc -l)" ]]; then

        psrsplit -n 1 ${arch}
        rm ${arch}

        for subarch in $(ls ${arch%.ar}*ar);
        do
            pam --setnchn 16 ${subarch} -e 16chn
            rm ${subarch}
            pam -mT ${subarch%.ar}.16chn
            /fred/oz002/users/mmiles/dlyfix/dlyfix ${subarch%.ar}.16chn -u . -o ${subarch%.ar}.dly
            rm ${subarch%.ar}.16chn
        done
    else 
        pam --setnchn 16 ${arch} -e 16chn

        rm ${arch}

        pam -mT ${arch%.ar}.16chn

        /fred/oz002/users/mmiles/dlyfix/dlyfix ${arch%.ar}.16chn -u . -o ${arch%.ar}.dly

        rm ${arch%.ar}.16chn
    fi; 
done




