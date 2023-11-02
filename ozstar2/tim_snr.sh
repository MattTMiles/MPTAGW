#!/bin/bash

seed_dir="/fred/oz002/users/mmiles/MPTA_GW/partim_august23/"
target_dir="/fred/oz002/users/mmiles/MPTA_GW/partim_august23_snr10/partim_updated/"

snr_cut=10

psr=$1

#for psr in $(cd ${seed_dir} && ls *tim); do
echo ${psr}
pat -jp -A FDM -C "chan rcvr snr length subint" -f "tempo2 IPTA" -P -s /fred/oz002/users/mmiles/MPTA_GW/portraits/2D.${psr}.notebook_version.ar $(awk '{print $1}' ${seed_dir}/${psr}.tim | uniq | cut -d "/" -f4 | grep J | rev | cut -f 2- -d '.' | rev | awk -v awkpsr=${psr} '{print "/fred/oz002/users/mmiles/MPTA_GW/data_august23/"awkpsr"/"$0"*"}')  >> ${target_dir}/${psr}.tim
#done
echo "made timfile"

#for tim in $(cd ${target_dir} && ls *tim); do 
    #echo $tim 
awk -v snrcut=${snr_cut} -F " " '$29>snrcut' ${target_dir}/${psr}.tim >> ${target_dir}/temp.tim 
mv ${target_dir}/temp.tim ${target_dir}/${psr}.tim
echo "trimmed timfile"

sed -i '1s/^/FORMAT 1\n/' ${target_dir}/${psr}.tim
echo "finished"

