#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE

ml conda

conda activate mpippcgw

i=0; 

ls /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/ >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/temp_corr_object.list

while [ $i > -1 ]; 
do   
    for psr1 in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt); 
    do
        echo $psr1
        if [[ ! "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/${psr1}_J1909-3744.pkl" == $(grep -w -m 1 /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/${psr1}_J1909-3744.pkl /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/temp_corr_object.list) ]] && [[ ! "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/J1909-3744_${psr1}.pkl" == $(grep -w -m 1 /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/J1909-3744_${psr1}.pkl /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/temp_corr_object.list) ]] && [[ ! "${psr1}" == "J1909-3744" ]]; then
            echo $psr1 J1909-3744
            python /home/mmiles/soft/GW/ozstar2/pairwise/cross_corr_pair_object_maker.py -pulsars ${psr1} J1909-3744 -partim /fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/ -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_WN_values.json -modelfile /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json -sampler pbilby -noise_search single_bin_cross_corr_er -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/
            echo ${psr1}_J1909-3744 >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pairwise_ready.list
            echo echo /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/pair_objects/${psr1}_J1909-3744.pkl >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE/temp_corr_object.list
        fi

    done

done 
