#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory_fixedWN.txt | grep ${psr} | awk '{print $2}')

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_SMBHB_CHROMANNUAL_SGWB/SMBHB_CHROMANNUAL_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} SMBHB_CHROMANNUAL_SGWB "efac_c equad_c ecorr_c smbhb_frank_psradd8 extra_chrom_annual"
    echo "rerunning ${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_SMBHB_CHROMBUMP_SGWB/SMBHB_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_SMBHB_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_SMBHB_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_SMBHB_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} SMBHB_CHROMBUMP_SGWB "efac_c equad_c ecorr_c smbhb_frank_psradd8 extra_chrom_gauss_bump" 
    echo "rerunning ${psr}_PBILBY_PSRADD8_SMBHB_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB/SMBHB_CHROMANNUAL_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} SMBHB_CHROMANNUAL_CHROMBUMP_SGWB "efac_c equad_c ecorr_c smbhb_frank_psradd8 extra_chrom_annual extra_chrom_gauss_bump"
    echo "rerunning ${psr}_PBILBY_PSRADD8_SMBHB_CHROMANNUAL_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_CHROMANNUAL_SGWB/CHROMANNUAL_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_CHROMANNUAL_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_CHROMANNUAL_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_CHROMANNUAL_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} CHROMANNUAL_SGWB "efac_c equad_c ecorr_c chrom_annual gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_PSRADD8_CHROMANNUAL_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_CHROMBUMP_SGWB/CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} CHROMBUMP_SGWB "efac_c equad_c ecorr_c chrom_gauss_bump gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_PSRADD8_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_CHROMANNUAL_CHROMBUMP_SGWB/CHROMANNUAL_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_CHROMANNUAL_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_CHROMANNUAL_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_CHROMANNUAL_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} CHROMANNUAL_CHROMBUMP_SGWB "efac_c equad_c ecorr_c chrom_annual chrom_gauss_bump gw_const_gamma"
    echo "rerunning ${psr}_PBILBY_PSRADD8_CHROMANNUAL_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_final.list
    #((counter++))
fi