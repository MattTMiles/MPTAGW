#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory.txt | grep ${psr} | awk '{print $2}')

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_PM_EFAC_SGWB/PM_EFAC_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_PM_EFAC_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_PM_EFAC_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_PM_EFAC_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} PM_EFAC_SGWB "efac smbhb_frank_psradd8"
    echo "rerunning ${psr}_PBILBY_PSRADD8_PM_EFAC_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_PM_EFAC_EQUAD_SGWB/PM_EFAC_EQUAD_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} PM_EFAC_EQUAD_SGWB "efac equad smbhb_frank_psradd8"
    echo "rerunning ${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_PM_EFAC_ECORR_SGWB/PM_EFAC_ECORR_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_PM_EFAC_ECORR_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_PM_EFAC_ECORR_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_PM_EFAC_ECORR_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} PM_EFAC_ECORR_SGWB "efac ecorr smbhb_frank_psradd8"
    echo "rerunning ${psr}_PBILBY_PSRADD8_PM_EFAC_ECORR_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_8/${psr}/${psr}_PM_EFAC_EQUAD_ECORR_SGWB/PM_EFAC_EQUAD_ECORR_SGWB_final_res.json" ]] && [[ ! "${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_ECORR_SGWB" == $(grep -w -m 1 ^${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_ECORR_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_ECORR_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD8_slurm.sh ${psr} PM_EFAC_EQUAD_ECORR_SGWB "efac equad ecorr smbhb_frank_psradd8"
    echo "rerunning ${psr}_PBILBY_PSRADD8_PM_EFAC_EQUAD_ECORR_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd8_PMWN.list
    #((counter++))
fi
