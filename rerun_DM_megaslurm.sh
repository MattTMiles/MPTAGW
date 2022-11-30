#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_live_200_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM) ]]; then
    sbatch -J ${psr}_live_200_DM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_live_200_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RED_DM/RED_DM_result.json" ]] && [[ ! "${psr}_live_200_RED_DM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RED_DM) ]]; then
    sbatch -J ${psr}_live_200_RED_DM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RED_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_live_200_RED_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RED_DM_CHROM/RED_DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_RED_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RED_DM_CHROM) ]]; then
    sbatch -J ${psr}_live_200_RED_DM_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RED_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_live_200_RED_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_DM_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_CHROM) ]]; then
    sbatch -J ${psr}_live_200_DM_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_live_200_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL/DM_BL_result.json" ]] && [[ ! "${psr}_live_200_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL) ]]; then
    sbatch -J ${psr}_live_200_DM_BL ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL "efac_c equad_c ecorr_c dm band_low"
    echo "rerunning ${psr}_live_200_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BH/DM_BH_result.json" ]] && [[ ! "${psr}_live_200_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BH) ]]; then
    sbatch -J ${psr}_live_200_DM_BH ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BH "efac_c equad_c ecorr_c dm band_high"
    echo "rerunning ${psr}_live_200_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL_BH/DM_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL_BH) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_BH ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL_BH "efac_c equad_c ecorr_c dm band_low band_high"
    echo "rerunning ${psr}_live_200_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL/RN_DM_BL_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL "efac_c equad_c ecorr_c red dm band_low"
    echo "rerunning ${psr}_live_200_RN_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL_BH/RN_DM_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL_BH) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_BH ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL_BH "efac_c equad_c ecorr_c red dm band_low band_high"
    echo "rerunning ${psr}_live_200_RN_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BH/RN_DM_BH_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BH" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BH) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BH ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BH "efac_c equad_c ecorr_c red dm band_high"
    echo "rerunning ${psr}_live_200_RN_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL_CHROM/DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL_CHROM "efac_c equad_c ecorr_c dm band_low chrom"
    echo "rerunning ${psr}_live_200_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BH_CHROM/DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_live_200_DM_BH_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BH_CHROM "efac_c equad_c ecorr_c dm band_high chrom"
    echo "rerunning ${psr}_live_200_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL_BH_CHROM/DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_BH_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL_BH_CHROM "efac_c equad_c ecorr_c dm band_low band_high chrom"
    echo "rerunning ${psr}_live_200_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL_CHROM/RN_DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL_CHROM) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL_CHROM "efac_c equad_c ecorr_c red dm band_low chrom"
    echo "rerunning ${psr}_live_200_RN_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL_BH_CHROM/RN_DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL_BH_CHROM) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_BH_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM "efac_c equad_c ecorr_c red dm band_low band_high chrom"
    echo "rerunning ${psr}_live_200_RN_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BH_CHROM/RN_DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BH_CHROM" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BH_CHROM) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BH_CHROM ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BH_CHROM "efac_c equad_c ecorr_c red dm band_high chrom"
    echo "rerunning ${psr}_live_200_RN_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_live_200_RN_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL_CHROMCIDX/DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL_CHROMCIDX "efac_c equad_c ecorr_c dm band_low chrom_cidx"
    echo "rerunning ${psr}_live_200_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BH_CHROMCIDX/DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_DM_BH_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_BL_BH_CHROMCIDX/DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_DM_BL_BH_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL_CHROMCIDX/RN_DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low chrom_cidx"
    echo "rerunning ${psr}_live_200_RN_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BL_BH_CHROMCIDX/RN_DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BL_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BL_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BL_BH_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_RN_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_RN_DM_BH_CHROMCIDX/RN_DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_RN_DM_BH_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_RN_DM_BH_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_RN_DM_BH_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} RN_DM_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_RN_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_DM_CHROMCIDX" == $(squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me | awk '{print $3}' | grep -w ^${psr}_live_200_DM_CHROMCIDX) ]]; then
    sbatch -J ${psr}_live_200_DM_CHROMCIDX ~/soft/GW/all_DM_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_live_200_DM_CHROMCIDX"
fi

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
print_date=$(date)
touch "${psr} has been live_200 rerun on ${print_date}"