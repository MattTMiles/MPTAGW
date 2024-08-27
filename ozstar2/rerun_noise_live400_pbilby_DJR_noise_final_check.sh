#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory_fixedWN.txt | grep ${psr} | awk '{print $2}')

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/DJR_noise/final_checks/${psr}/${psr}_SMBHB_SGWB/SMBHB_SGWB_final_res.json" ]] && [[ ! "${psr}_DJR_PBILBY_SMBHB_SGWB" == $(grep -w -m 1 ^${psr}_DJR_PBILBY_SMBHB_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_DJR_PBILBY_SMBHB_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_DJR_final_check_slurm.sh ${psr} SMBHB_SGWB "efac_c equad_c ecorr_split_c smbhb"
    echo "rerunning ${psr}_DJR_PBILBY_SMBHB_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/DJR_noise/final_checks/${psr}/${psr}_SMBHB_CHROMANNUAL_SGWB/SMBHB_CHROMANNUAL_SGWB_final_res.json" ]] && [[ ! "${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_SGWB" == $(grep -w -m 1 ^${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_DJR_final_check_slurm.sh ${psr} SMBHB_CHROMANNUAL_SGWB "efac_c equad_c ecorr_split_c smbhb extra_chrom_annual"
    echo "rerunning ${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/DJR_noise/final_checks/${psr}/${psr}_SMBHB_CHROMBUMP_SGWB/SMBHB_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_DJR_PBILBY_SMBHB_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_DJR_PBILBY_SMBHB_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_DJR_PBILBY_SMBHB_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_DJR_final_check_slurm.sh ${psr} SMBHB_CHROMBUMP_SGWB "efac_c equad_c ecorr_split_c smbhb extra_chrom_gauss_bump" 
    echo "rerunning ${psr}_DJR_PBILBY_SMBHB_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/DJR_noise/final_checks/${psr}/${psr}_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB/SMBHB_CHROMANNUAL_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_DJR_final_check_slurm.sh ${psr} SMBHB_CHROMANNUAL_CHROMBUMP_SGWB "efac_c equad_c ecorr_split_c smbhb extra_chrom_annual extra_chrom_gauss_bump"
    echo "rerunning ${psr}_DJR_PBILBY_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait.list
    #((counter++))
fi