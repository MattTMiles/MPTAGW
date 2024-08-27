#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory.txt | grep ${psr} | awk '{print $2}')

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_EFAC/EFAC_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_EFAC" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_EFAC /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_EFAC /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} EFAC "efac red dm"
    echo "rerunning ${psr}_PSRADD10_PBILBY_EFAC" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_EFAC_EQUAD/EFAC_EQUAD_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_EFAC_EQUAD" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_EFAC_EQUAD /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_EFAC_EQUAD /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} EFAC_EQUAD "efac equad red dm"
    echo "rerunning ${psr}_PSRADD10_PBILBY_EFAC_EQUAD" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_EFAC_EQUAD_ECORR/EFAC_EQUAD_ECORR_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_EFAC_EQUAD_ECORR" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_EFAC_EQUAD_ECORR /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_EFAC_EQUAD_ECORR /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} EFAC_EQUAD_ECORR "efac equad ecorr red dm"
    echo "rerunning ${psr}_PSRADD10_PBILBY_EFAC_EQUAD_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_EFAC_ECORR/EFAC_ECORR_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_EFAC_ECORR" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_EFAC_ECORR /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_EFAC_ECORR /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} EFAC_ECORR "efac ecorr red dm"
    echo "rerunning ${psr}_PSRADD8_PBILBY_EFAC_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10.list
    #((counter++))
fi