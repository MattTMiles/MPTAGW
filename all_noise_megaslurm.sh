#!/bin/bash

# all_noise_megaslurm.sh psr
# runs entire suite of noise models being tested for the MPTA

psr=$1

sbatch -J ${psr}_DM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM "efac_c equad_c ecorr_c dm"

sbatch -J ${psr}_RN ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN "efac_c equad_c ecorr_c red"

sbatch -J ${psr}_WN ~/soft/GW/all_noise_mpi_slurm.sh ${psr} WN "efac equad ecorr"

sbatch -J ${psr}_WN_NO_ECORR ~/soft/GW/all_noise_mpi_slurm.sh ${psr} WN_NO_ECORR "efac equad"

sbatch -J ${psr}_RED_DM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RED_DM "efac_c equad_c ecorr_c red dm"

sbatch -J ${psr}_RED_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RED_CHROM "efac_c equad_c ecorr_c red chrom"

sbatch -J ${psr}_RED_DM_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RED_DM_CHROM "efac_c equad_c ecorr_c red dm chrom"

sbatch -J ${psr}_DM_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_CHROM "efac_c equad_c ecorr_c dm chrom"

sbatch -J ${psr}_RED_BL ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RED_BL "efac_c equad_c ecorr_c red band_low"

sbatch -J ${psr}_RED_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RED_BH "efac_c equad_c ecorr_c red band_high"

sbatch -J ${psr}_DM_BL ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL "efac_c equad_c ecorr_c dm band_low"

sbatch -J ${psr}_DM_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BH "efac_c equad_c ecorr_c dm band_high"

sbatch -J ${psr}_RN_BL_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BL_BH "efac_c equad_c ecorr_c red band_low band_high"

sbatch -J ${psr}_DM_BL_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL_BH "efac_c equad_c ecorr_c dm band_low band_high"

sbatch -J ${psr}_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROM "efac_c equad_c ecorr_c chrom"

sbatch -J ${psr}_CHROM_BL ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROM_BL "efac_c equad_c ecorr_c chrom band_low"

sbatch -J ${psr}_CHROM_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROM_BH "efac_c equad_c ecorr_c chrom band_high"

sbatch -J ${psr}_CHROM_BL_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROM_BL_BH "efac_c equad_c ecorr_c chrom band_low band_high"

sbatch -J ${psr}_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROMCIDX "efac_c equad_c ecorr_c chrom_cidx"

sbatch -J ${psr}_CHROMCIDX_BL ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROMCIDX_BL "efac_c equad_c ecorr_c chrom_cidx band_low"

sbatch -J ${psr}_CHROMCIDX_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROMCIDX_BH "efac_c equad_c ecorr_c chrom_cidx band_high"

sbatch -J ${psr}_CHROMCIDX_BL_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} CHROMCIDX_BL_BH "efac_c equad_c ecorr_c chrom_cidx band_low band_high"

sbatch -J ${psr}_RN_DM_BL ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL "efac_c equad_c ecorr_c red dm band_low"

sbatch -J ${psr}_RN_DM_BL_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL_BH "efac_c equad_c ecorr_c red dm band_low band_high"

sbatch -J ${psr}_RN_DM_BH ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BH "efac_c equad_c ecorr_c red dm band_high"

sbatch -J ${psr}_RN_BL_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BL_CHROM "efac_c equad_c ecorr_c red band_low chrom"

sbatch -J ${psr}_RN_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BH_CHROM "efac_c equad_c ecorr_c red band_high chrom"

sbatch -J ${psr}_DM_BL_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL_CHROM "efac_c equad_c ecorr_c dm band_low chrom"

sbatch -J ${psr}_DM_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BH_CHROM "efac_c equad_c ecorr_c dm band_high chrom"

sbatch -J ${psr}_RN_BL_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BL_BH_CHROM "efac_c equad_c ecorr_c red band_low band_high chrom"

sbatch -J ${psr}_DM_BL_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL_BH_CHROM "efac_c equad_c ecorr_c dm band_low band_high chrom"

sbatch -J ${psr}_RN_DM_BL_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL_CHROM "efac_c equad_c ecorr_c red dm band_low chrom"

sbatch -J ${psr}_RN_DM_BL_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROM "efac_c equad_c ecorr_c red dm band_low band_high chrom"

sbatch -J ${psr}_RN_DM_BH_CHROM ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BH_CHROM "efac_c equad_c ecorr_c red dm band_high chrom"

sbatch -J ${psr}_RN_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_CHROMCIDX "efac_c equad_c ecorr_c red chrom_cidx"

sbatch -J ${psr}_RN_DM_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_CHROMCIDX "efac_c equad_c ecorr_c red dm chrom_cidx"

sbatch -J ${psr}_RN_BL_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BL_CHROMCIDX "efac_c equad_c ecorr_c red band_low chrom_cidx"

sbatch -J ${psr}_RN_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_high chrom_cidx"

sbatch -J ${psr}_DM_BL_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL_CHROMCIDX "efac_c equad_c ecorr_c dm band_low chrom_cidx"

sbatch -J ${psr}_DM_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_high chrom_cidx"

sbatch -J ${psr}_RN_BL_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red band_low band_high chrom_cidx"

sbatch -J ${psr}_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c dm band_low band_high chrom_cidx"

sbatch -J ${psr}_RN_DM_BL_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low chrom_cidx"

sbatch -J ${psr}_RN_DM_BL_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BL_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_low band_high chrom_cidx"

sbatch -J ${psr}_RN_DM_BH_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} RN_DM_BH_CHROMCIDX "efac_c equad_c ecorr_c red dm band_high chrom_cidx"

sbatch -J ${psr}_DM_CHROMCIDX ~/soft/GW/all_noise_mpi_slurm.sh ${psr} DM_CHROMCIDX "efac_c equad_c ecorr_c dm chrom_cidx"

