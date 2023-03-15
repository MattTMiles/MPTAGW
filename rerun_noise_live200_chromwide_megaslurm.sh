#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_CHROM_WIDE/RN_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_CHROM_WIDE "efac_c equad_c ecorr_c red chrom_wide"
    echo "rerunning ${psr}_live_200_RN_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_DM_CHROM_WIDE/RN_DM_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_DM_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_CHROM_WIDE "efac_c equad_c ecorr_c red dm chrom_wide"
    echo "rerunning ${psr}_live_200_RN_DM_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_DM_CHROM_WIDE/DM_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_DM_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_DM_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_DM_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROM_WIDE "efac_c equad_c ecorr_c dm chrom_wide"
    echo "rerunning ${psr}_live_200_DM_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_CHROM_WIDE/CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} CHROM_WIDE "efac_c equad_c ecorr_c chrom_wide"
    echo "rerunning ${psr}_live_200_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_CHROM_WIDE_BL/CHROM_WIDE_BL_result.json" ]] && [[ ! "${psr}_live_200_CHROM_WIDE_BL" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_CHROM_WIDE_BL) ]]; then
    sbatch -J ${psr}_live_200_CHROM_WIDE_BL ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BL "efac_c equad_c ecorr_c chrom_wide band_low"
    echo "rerunning ${psr}_live_200_CHROM_WIDE_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_CHROM_WIDE_BH/CHROM_WIDE_BH_result.json" ]] && [[ ! "${psr}_live_200_CHROM_WIDE_BH" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_CHROM_WIDE_BH) ]]; then
    sbatch -J ${psr}_live_200_CHROM_WIDE_BH ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BH "efac_c equad_c ecorr_c chrom_wide band_high"
    echo "rerunning ${psr}_live_200_CHROM_WIDE_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_CHROM_WIDE_BL_BH/CHROM_WIDE_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_CHROM_WIDE_BL_BH" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_CHROM_WIDE_BL_BH) ]]; then
    sbatch -J ${psr}_live_200_CHROM_WIDE_BL_BH ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} CHROM_WIDE_BL_BH "efac_c equad_c ecorr_c chrom_wide band_low band_high"
    echo "rerunning ${psr}_live_200_CHROM_WIDE_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_BL_CHROM_WIDE/RN_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_BL_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_BL_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_BL_CHROM_WIDE "efac_c equad_c ecorr_c red band_low chrom_wide"
    echo "rerunning ${psr}_live_200_RN_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_BH_CHROM_WIDE/RN_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_BH_CHROM_WIDE "efac_c equad_c ecorr_c red band_high chrom_wide"
    echo "rerunning ${psr}_live_200_RN_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_DM_BL_CHROM_WIDE/DM_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_DM_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} DM_BL_CHROM_WIDE "efac_c equad_c ecorr_c dm band_low chrom_wide"
    echo "rerunning ${psr}_live_200_DM_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_DM_BH_CHROM_WIDE/DM_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_DM_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_DM_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_DM_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} DM_BH_CHROM_WIDE "efac_c equad_c ecorr_c dm band_high chrom_wide"
    echo "rerunning ${psr}_live_200_DM_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_BL_BH_CHROM_WIDE/RN_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_BL_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c red band_low band_high chrom_wide"
    echo "rerunning ${psr}_live_200_RN_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_DM_BL_BH_CHROM_WIDE/DM_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_DM_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} DM_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c dm band_low band_high chrom_wide"
    echo "rerunning ${psr}_live_200_DM_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_DM_BL_CHROM_WIDE/RN_DM_BL_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_DM_BL_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_BL_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_low chrom_wide"
    echo "rerunning ${psr}_live_200_RN_DM_BL_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_DM_BL_BH_CHROM_WIDE/RN_DM_BL_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_DM_BL_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_low band_high chrom_wide"
    echo "rerunning ${psr}_live_200_RN_DM_BL_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_RN_DM_BH_CHROM_WIDE/RN_DM_BH_CHROM_WIDE_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BH_CHROM_WIDE" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_RN_DM_BH_CHROM_WIDE) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BH_CHROM_WIDE ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} RN_DM_BH_CHROM_WIDE "efac_c equad_c ecorr_c red dm band_high chrom_wide"
    echo "rerunning ${psr}_live_200_RN_DM_BH_CHROM_WIDE"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}/${psr}_DM_CHROM_WIDE_CHROM_ANNUAL/DM_CHROM_WIDE_CHROM_ANNUAL_result.json" ]] && [[ ! "${psr}_live_200_DM_CHROM_WIDE_CHROM_ANNUAL" == $(cat /fred/oz002/users/mmiles/MPTA_GW/chrom_wide_slurm.list | grep -w ^${psr}_live_200_DM_CHROM_WIDE_CHROM_ANNUAL) ]]; then
    sbatch -J ${psr}_live_200_DM_CHROM_WIDE_CHROM_ANNUAL ~/soft/GW/all_noise_live200_mpi_slurm.sh ${psr} DM_CHROM_WIDE_CHROM_ANNUAL "efac_c equad_c ecorr_c dm chrom_wide chrom_annual"
    echo "rerunning ${psr}_live_200_DM_CHROM_WIDE_CHROM_ANNUAL"
fi