#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMANNUAL_ER/SMBHB_WN_CHROMANNUAL_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMANNUAL_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMANNUAL_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMANNUAL_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMANNUAL_ER "smbhb_wn extra_chrom_annual extra_red" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMANNUAL_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN/SMBHB_WN_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN" == $(grep -w -m 1 ^${psr}_SMBHB_WN /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_WN /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN "smbhb_wn" 400
    echo "rerunning ${psr}_SMBHB_WN" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_ALL/SMBHB_WN_ALL_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_ALL" == $(grep -w -m 1 ^${psr}_SMBHB_WN_ALL /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_WN_ALL /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_ALL "smbhb_wn_all" 400
    echo "rerunning ${psr}_SMBHB_WN_ALL" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi



# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMBUMP_ER/SMBHB_WN_CHROMBUMP_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMBUMP_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMBUMP_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMBUMP_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMBUMP_ER "smbhb_wn extra_red extra_chrom_gauss_bump" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMBUMP_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER/SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER "smbhb_wn extra_red extra_chrom_annual extra_chrom_gauss_bump" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_CHROMBUMP/SMBHB_CHROMBUMP_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMBUMP" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMBUMP /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMBUMP /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_CHROMBUMP "efac_c equad_c ecorr_gauss_c smbhb extra_chrom_gauss_bump" 400
    echo "rerunning ${psr}_SMBHB_CHROMBUMP" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_CHROMBUMP_CHROMANNUAL/SMBHB_CHROMBUMP_CHROMANNUAL_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMBUMP_CHROMANNUAL" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMBUMP_CHROMANNUAL /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMBUMP_CHROMANNUAL /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_CHROMBUMP_CHROMANNUAL "efac_c equad_c ecorr_gauss_c smbhb extra_chrom_annual extra_chrom_gauss_bump" 400
    echo "rerunning ${psr}_SMBHB_CHROMBUMP_CHROMANNUAL" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_CHROMANNUAL/SMBHB_CHROMANNUAL_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMANNUAL" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMANNUAL /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMANNUAL /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_CHROMANNUAL "efac_c equad_c ecorr_gauss_c smbhb extra_chrom_annual " 400
    echo "rerunning ${psr}_SMBHB_CHROMANNUAL" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi
