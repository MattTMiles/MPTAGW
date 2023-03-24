#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM/DM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"
    echo "rerunning ${psr}_live_200_portrait_tester_DM"
fi 

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN/RN_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"
    echo "rerunning ${psr}_live_200_portrait_tester_RN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_WN/WN_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_WN" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_WN /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_WN /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} WN "wn_tester"
    echo "rerunning ${psr}_live_200_portrait_tester_WN"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM/RN_DM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM "efac_c equad_c ecorr_c red dm"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_CHROM/RN_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_CHROM "efac_c equad_c ecorr_c red chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_CHROM/RN_DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_CHROM/DM_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_CHROM /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL/RN_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL "efac_c equad_c ecorr_c red band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BH/RN_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BH /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BH "efac_c equad_c ecorr_c red band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL/DM_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL "efac_c equad_c ecorr_c dm band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BH/DM_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BH "efac_c equad_c ecorr_c dm band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL_BH/RN_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL_BH "efac_c equad_c ecorr_c red band_low band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL_BH/DM_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL_BH "efac_c equad_c ecorr_c dm band_low band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROM/CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROM_BL/CHROM_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROM_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROM_BL  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROM_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROM_BL "efac_c equad_c ecorr_c chrom band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROM_BH/CHROM_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROM_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROM_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROM_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROM_BH "efac_c equad_c ecorr_c chrom band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROM_BL_BH/CHROM_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROM_BL_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROM_BL_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROM_BL_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROM_BL_BH "efac_c equad_c ecorr_c chrom band_low band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROMCIDX/CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROMCIDX_BL/CHROMCIDX_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROMCIDX_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROMCIDX_BL  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROMCIDX_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROMCIDX_BL "efac_c equad_c ecorr_c chrom_cidx band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROMCIDX_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROMCIDX_BH/CHROMCIDX_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROMCIDX_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROMCIDX_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROMCIDX_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROMCIDX_BH "efac_c equad_c ecorr_c chrom_cidx band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROMCIDX_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_CHROMCIDX_BL_BH/CHROMCIDX_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_CHROMCIDX_BL_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_CHROMCIDX_BL_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_CHROMCIDX_BL_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} CHROMCIDX_BL_BH "efac_c equad_c ecorr_c chrom_cidx band_low band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_CHROMCIDX_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL/RN_DM_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL "efac_c equad_c ecorr_c red dm band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL_BH/RN_DM_BL_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL_BH "efac_c equad_c ecorr_c red dm band_low band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BH/RN_DM_BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BH "efac_c equad_c ecorr_c red dm band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL_CHROM/RN_BL_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL_CHROM "efac_c equad_c ecorr_c red band_low chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BH_CHROM/RN_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BH_CHROM "efac_c equad_c ecorr_c red band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL_CHROM/DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL_CHROM "efac_c equad_c ecorr_c dm band_low chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BH_CHROM/DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BH_CHROM "efac_c equad_c ecorr_c dm band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL_BH_CHROM/RN_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL_BH_CHROM "efac_c equad_c ecorr_c red band_low band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL_BH_CHROM/DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL_BH_CHROM "efac_c equad_c ecorr_c dm band_low band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL_CHROM/RN_DM_BL_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL_CHROM "efac_c equad_c ecorr_c red dm band_low chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL_BH_CHROM/RN_DM_BL_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM "efac_c equad_c ecorr_c red dm band_low band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BH_CHROM/RN_DM_BH_CHROM_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BH_CHROM" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BH_CHROM  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BH_CHROM /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BH_CHROM "efac_c equad_c ecorr_c red dm band_high chrom"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BH_CHROM"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_CHROMCIDX/RN_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_CHROMCIDX/RN_DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL_CHROMCIDX/RN_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL_CHROMCIDX "efac_c equad_c ecorr_c red band_low chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BH_CHROMCIDX/RN_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL_CHROMCIDX/DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL_CHROMCIDX "efac_c equad_c ecorr_c dm band_low chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BH_CHROMCIDX/DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_BL_BH_CHROMCIDX/RN_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_BL_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_BL_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_BL_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_low band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_BL_BH_CHROMCIDX/DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_BL_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_BL_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_BL_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL_CHROMCIDX/RN_DM_BL_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BL_BH_CHROMCIDX/RN_DM_BL_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BL_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_RN_DM_BH_CHROMCIDX/RN_DM_BH_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_RN_DM_BH_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_RN_DM_BH_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_RN_DM_BH_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} RN_DM_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_high chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_RN_DM_BH_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_DM_CHROMCIDX/DM_CHROMCIDX_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_DM_CHROMCIDX" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_DM_CHROMCIDX  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_DM_CHROMCIDX /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"
    echo "rerunning ${psr}_live_200_portrait_tester_DM_CHROMCIDX"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_BH/BH_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_BH" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_BH  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_BH /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} BH "efac_c equad_c ecorr_c band_high"
    echo "rerunning ${psr}_live_200_portrait_tester_BH"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_BL/BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_BL  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} BL "efac_c equad_c ecorr_c band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_BL"
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise/out_ppc/live_200_portrait_tester/${psr}/${psr}_BH_BL/BH_BL_result.json" ]] && [[ ! "${psr}_live_200_portrait_tester_BH_BL" == $(grep -w -m 1 ^${psr}_live_200_portrait_tester_BH_BL  /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm.list) ]]; then
    sbatch -J ${psr}_live_200_portrait_tester_BH_BL /home/mmiles/soft/GW/all_noise_live200_portrait_tester_mpi_slurm.sh ${psr} BH_BL "efac_c equad_c ecorr_c band_high band_low"
    echo "rerunning ${psr}_live_200_portrait_tester_BH_BL"
fi

#cd /fred/oz002/users/mmiles/MPTA_GW/enterprise/rerun_logs/
#print_date=$(date)
#touch "${psr} has been rerun on ${print_date}"