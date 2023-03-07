#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished single pulsar GW models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW/SPGW_result.json" ]] && [[ ! "${psr}_live_200_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_200_SPGW) ]]; then
    sbatch -J ${psr}_live_200_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW "efac_c equad_c ecorr_c spgw"  200 DE440
    echo "rerunning ${psr}_live_200_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC/SPGWC_result.json" ]] && [[ ! "${psr}_live_200_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_200_SPGWC) ]]; then
    sbatch -J ${psr}_live_200_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC "efac_c equad_c ecorr_c spgwc" 200 DE440
    echo "rerunning ${psr}_live_200_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW400/SPGW400_result.json" ]] && [[ ! "${psr}_live_400_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_400_SPGW) ]]; then
    sbatch -J ${psr}_live_400_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW400 "efac_c equad_c ecorr_c spgw"  400 DE440
    echo "rerunning ${psr}_live_400_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC400/SPGWC400_result.json" ]] && [[ ! "${psr}_live_400_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_400_SPGWC) ]]; then
    sbatch -J ${psr}_live_400_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC400 "efac_c equad_c ecorr_c spgwc" 400 DE440
    echo "rerunning ${psr}_live_400_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW600/SPGW600_result.json" ]] && [[ ! "${psr}_live_600_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_SPGW) ]]; then
    sbatch -J ${psr}_live_600_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW600 "efac_c equad_c ecorr_c spgw"  600 DE440
    echo "rerunning ${psr}_live_600_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC600/SPGWC600_result.json" ]] && [[ ! "${psr}_live_600_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_SPGWC) ]]; then
    sbatch -J ${psr}_live_600_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC600 "efac_c equad_c ecorr_c spgwc" 600 DE440
    echo "rerunning ${psr}_live_600_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW1000/SPGW1000_result.json" ]] && [[ ! "${psr}_live_1000_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_SPGW) ]]; then
    sbatch -J ${psr}_live_1000_SPGW ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW1000 "efac_c equad_c ecorr_c spgw"  1000 DE440
    echo "rerunning ${psr}_live_1000_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC1000/SPGWC1000_result.json" ]] && [[ ! "${psr}_live_1000_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_SPGWC) ]]; then
    sbatch -J ${psr}_live_1000_SPGWC ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC1000 "efac_c equad_c ecorr_c spgwc" 1000 DE440
    echo "rerunning ${psr}_live_1000_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW600_ER/SPGW600_ER_result.json" ]] && [[ ! "${psr}_live_600_SPGW_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_SPGW) ]]; then
    sbatch -J ${psr}_live_600_SPGW_ER ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW600_ER "efac_c equad_c ecorr_c red spgw"  600 DE440
    echo "rerunning ${psr}_live_600_SPGW_ER"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC600_ER/SPGWC600_ER_result.json" ]] && [[ ! "${psr}_live_600_SPGWC_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_SPGWC) ]]; then
    sbatch -J ${psr}_live_600_SPGWC_ER ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC600_ER "efac_c equad_c ecorr_c red spgwc" 600 DE440
    echo "rerunning ${psr}_live_600_SPGWC_ER"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGW1000_ER/SPGW1000_ER_result.json" ]] && [[ ! "${psr}_live_1000_SPGW_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_SPGW) ]]; then
    sbatch -J ${psr}_live_1000_SPGW_ER ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGW1000_ER "efac_c equad_c ecorr_c red spgw"  1000 DE440
    echo "rerunning ${psr}_live_1000_SPGW_ER"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/SPGW/${psr}/${psr}_SPGWC1000_ER/SPGWC1000_ER_result.json" ]] && [[ ! "${psr}_live_1000_SPGWC_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_SPGWC) ]]; then
    sbatch -J ${psr}_live_1000_SPGWC_ER ~/soft/GW/SPGW_noise_mpi_slurm.sh ${psr} SPGWC1000_ER "efac_c equad_c ecorr_c red spgwc" 1000 DE440
    echo "rerunning ${psr}_live_1000_SPGWC_ER"
fi


if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGW600/FREE_SPGW600_result.json" ]] && [[ ! "${psr}_live_600_FREE_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_FREE_SPGW) ]]; then
    sbatch -J ${psr}_live_600_FREE_SPGW /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGW600 "efac_c equad_c ecorr_c free_spgw"  600 DE440
    echo "rerunning ${psr}_live_600_FREE_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGWC600/FREE_SPGWC600_result.json" ]] && [[ ! "${psr}_live_600_FREE_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_FREE_SPGWC) ]]; then
    sbatch -J ${psr}_live_600_FREE_SPGWC /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGWC600 "efac_c equad_c ecorr_c free_spgwc" 600 DE440
    echo "rerunning ${psr}_live_600_FREE_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGW1000/FREE_SPGW1000_result.json" ]] && [[ ! "${psr}_live_1000_FREE_SPGW" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_FREE_SPGW) ]]; then
    sbatch -J ${psr}_live_1000_FREE_SPGW /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGW1000 "efac_c equad_c ecorr_c free_spgw"  1000 DE440
    echo "rerunning ${psr}_live_1000_FREE_SPGW"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGWC1000/FREE_SPGWC1000_result.json" ]] && [[ ! "${psr}_live_1000_FREE_SPGWC" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_1000_FREE_SPGWC) ]]; then
    sbatch -J ${psr}_live_1000_FREE_SPGWC /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGWC1000 "efac_c equad_c ecorr_c free_spgwc" 1000 DE440
    echo "rerunning ${psr}_live_1000_FREE_SPGWC"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGW600_ER/FREE_SPGW600_ER_result.json" ]] && [[ ! "${psr}_live_600_FREE_SPGW_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_FREE_SPGW_ER) ]]; then
    sbatch -J ${psr}_live_600_FREE_SPGW_ER /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGW600_ER "efac_c equad_c ecorr_c red free_spgw"  600 DE440
    echo "rerunning ${psr}_live_600_FREE_SPGW_ER"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/FREE_SPGW/${psr}/${psr}_FREE_SPGWC600_ER/FREE_SPGWC600_ER_result.json" ]] && [[ ! "${psr}_live_600_FREE_SPGWC_ER" == $(cat /fred/oz002/users/mmiles/MPTA_GW/SPGW_slurm.list | grep -w ^${psr}_live_600_FREE_SPGWC_ER) ]]; then
    sbatch -J ${psr}_live_600_FREE_SPGWC_ER /home/mmiles/soft/GW/FREE_SPGW_noise_mpi_slurm.sh ${psr} FREE_SPGWC600_ER "efac_c equad_c ecorr_c red free_spgwc" 600 DE440
    echo "rerunning ${psr}_live_600_FREE_SPGWC_ER"
fi
