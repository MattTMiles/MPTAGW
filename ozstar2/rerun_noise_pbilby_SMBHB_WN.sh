#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMANNUAL_ER/SMBHB_WN_CHROMANNUAL_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMANNUAL_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMANNUAL_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMANNUAL_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMANNUAL_ER "smbhb_wn extra_chrom_annual extra_red" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMANNUAL_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi


# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN/SMBHB_WN_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN" == $(grep -w -m 1 ^${psr}_SMBHB_WN /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN "smbhb_wn" 400
#     echo "rerunning ${psr}_SMBHB_WN" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_ALL/SMBHB_WN_ALL_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_ALL" == $(grep -w -m 1 ^${psr}_SMBHB_WN_ALL /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_ALL /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_ALL "smbhb_wn_all" 400
#     echo "rerunning ${psr}_SMBHB_WN_ALL" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi



# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMBUMP_ER/SMBHB_WN_CHROMBUMP_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMBUMP_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMBUMP_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMBUMP_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMBUMP_ER "smbhb_wn extra_red extra_chrom_gauss_bump" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMBUMP_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi

# if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER/SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER_final_res.json" ]] && [[ ! "${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER" == $(grep -w -m 1 ^${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
#     sbatch -J ${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER "smbhb_wn extra_red extra_chrom_annual extra_chrom_gauss_bump" 400
#     echo "rerunning ${psr}_SMBHB_WN_CHROMBUMP_CHROMANNUAL_ER" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
# fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SMBHB_CHROMBUMP_FIXEDGW/SMBHB_CHROMBUMP_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMBUMP_FIXEDGW" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMBUMP_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMBUMP_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SMBHB_CHROMBUMP_FIXEDGW "smbhb_const_wn extra_chrom_gauss_bump" 400
    echo "rerunning ${psr}_SMBHB_CHROMBUMP_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW/SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW "smbhb_const_wn extra_chrom_annual extra_chrom_gauss_bump" 400
    echo "rerunning ${psr}_SMBHB_CHROMBUMP_CHROMANNUAL_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SMBHB_CHROMANNUAL_FIXEDGW/SMBHB_CHROMANNUAL_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_SMBHB_CHROMANNUAL_FIXEDGW" == $(grep -w -m 1 ^${psr}_SMBHB_CHROMANNUAL_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_CHROMANNUAL_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SMBHB_CHROMANNUAL_FIXEDGW "smbhb_const_wn extra_chrom_annual " 400
    echo "rerunning ${psr}_SMBHB_CHROMANNUAL_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi



if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_EFAC/SMBHB_EFAC_final_res.json" ]] && [[ ! "${psr}_SMBHB_EFAC" == $(grep -w -m 1 ^${psr}_SMBHB_EFAC /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_EFAC /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_EFAC "efac smbhb" 400
    echo "rerunning ${psr}_SMBHB_EFAC" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_EFAC_EQUAD/SMBHB_EFAC_EQUAD_final_res.json" ]] && [[ ! "${psr}_SMBHB_EFAC_EQUAD" == $(grep -w -m 1 ^${psr}_SMBHB_EFAC_EQUAD /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_EFAC_EQUAD /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_EFAC_EQUAD "efac equad smbhb" 400
    echo "rerunning ${psr}_SMBHB_EFAC_EQUAD" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_EFAC_ECORR/SMBHB_EFAC_ECORR_final_res.json" ]] && [[ ! "${psr}_SMBHB_EFAC_ECORR" == $(grep -w -m 1 ^${psr}_SMBHB_EFAC_ECORR /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_EFAC_ECORR /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_EFAC_ECORR "efac ecorr_gauss smbhb" 400
    echo "rerunning ${psr}_SMBHB_EFAC_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_SMBHB_EFAC_EQUAD_ECORR/SMBHB_EFAC_EQUAD_ECORR_final_res.json" ]] && [[ ! "${psr}_SMBHB_EFAC_EQUAD_ECORR" == $(grep -w -m 1 ^${psr}_SMBHB_EFAC_EQUAD_ECORR /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_SMBHB_EFAC_EQUAD_ECORR /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} SMBHB_EFAC_EQUAD_ECORR "efac equad ecorr_gauss smbhb" 400
    echo "rerunning ${psr}_SMBHB_EFAC_EQUAD_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}/${psr}_WN_SGWB/WN_SGWB_final_res.json" ]] && [[ ! "${psr}_WN_SGWB" == $(grep -w -m 1 ^${psr}_WN_SGWB /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list) ]]; then
    sbatch -J ${psr}_WN_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_slurm.sh ${psr} WN_SGWB "wn gw_const_gamma" 400
    echo "rerunning ${psr}_WN_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/pbilby_smbhb_wn_noise_search.list
fi