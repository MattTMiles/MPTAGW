#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list | wc -l)

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_WN" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_WN /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_WN ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} WN "efac equad ecorr red dm"
    echo "rerunning ${psr}_live_200_ozstar2_WN" >> /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_WN_NO_ECORR/WN_NO_ECORR_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_WN_NO_ECORR" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_WN_NO_ECORR /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_WN_NO_ECORR ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_ECORR "efac equad red dm"
    echo "rerunning ${psr}_live_200_ozstar2_WN_NO_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_WN_NO_EQUAD/WN_NO_EQUAD_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_WN_NO_EQUAD" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_WN_NO_EQUAD /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_WN_NO_EQUAD ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_EQUAD "efac ecorr red dm"
    echo "rerunning ${psr}_live_200_ozstar2_WN_NO_EQUAD" >> /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_ozstar2/out_ppc/live_200_ozstar2/${psr}/${psr}_WN_NO_EQUAD_NO_ECORR/WN_NO_EQUAD_NO_ECORR_result.json" ]] && [[ ! "${psr}_live_200_ozstar2_WN_NO_EQUAD_NO_ECORR" == $(grep -w -m 1 ^${psr}_live_200_ozstar2_WN_NO_EQUAD_NO_ECORR /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list) ]] && [[ $(cat /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list | wc -l) -lt 11 ]] && [[ $counter -lt 11 ]]; then
    sbatch -J ${psr}_live_200_ozstar2_WN_NO_EQUAD_NO_ECORR ~/soft/GW/ozstar2/all_noise_live200_mpi_slurm.sh ${psr} WN_NO_EQUAD_NO_ECORR "efac red dm"
    echo "rerunning ${psr}_live_200_ozstar2_WN_NO_EQUAD_NO_ECORR" >> /fred/oz002/users/mmiles/MPTA_GW/WN_rerun_slurm.list
fi