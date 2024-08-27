#!/bin/bash

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341

ml conda

conda activate mpippcgw



ls /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/*pkl >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/temp_corr_object.list



for psr1 in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt); 
do
    echo $psr1
    for psr2 in $(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/MPTA_pulsar_list.txt);
    do 
        
        if [[ ! "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${psr1}_${psr2}.pkl" == $(grep -w -m 1 /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${psr1}_${psr2}.pkl /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/temp_corr_object.list) ]] && [[ ! "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${psr2}_${psr1}.pkl" == $(grep -w -m 1 /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${psr2}_${psr1}.pkl /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/temp_corr_object.list) ]] && [[ ! "${psr1}" == "${psr2}" ]]; then
            echo $psr1 $psr2
            python /home/mmiles/soft/GW/ozstar2/pairwise/cross_corr_pair_object_maker.py -pulsars ${psr1} ${psr2} -partim /fred/oz002/users/mmiles/MPTA_GW/partim_frank/pp_8/ -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/PM_WN/MPTA_WN_values.json -modelfile /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/MPTA_noise_models_PP8.json -sampler ptmcmc -noise_search single_bin_cross_corr_er_altgamma -alt_dir /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/
            echo ${psr1}_${psr2} >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pairwise_ready.list
            echo /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/pair_objects/${psr1}_${psr2}.pkl >> /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/temp_corr_object.list
        fi
    done
done


rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/PAIRWISE_341/temp_corr_object.list