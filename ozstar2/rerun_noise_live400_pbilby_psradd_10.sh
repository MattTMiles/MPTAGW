#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

req_mem=$(cat /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/psr_memory_fixedWN.txt | grep ${psr} | awk '{print $2}')

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

# inc. runs with a fixed gamma SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_SGWB/DM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_SGWB "efac_c equad_c ecorr_c dm gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_SGWB/RN_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_SGWB "efac_c equad_c ecorr_c red gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_SGWB/RN_DM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_SGWB "efac_c equad_c ecorr_c red dm gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROM_SGWB/RN_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROM_SGWB "efac_c equad_c ecorr_c red chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROM_SGWB/RN_DM_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROM_SGWB "efac_c equad_c ecorr_c red dm chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROM_SGWB/DM_CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROM_SGWB "efac_c equad_c ecorr_c dm chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROM_SGWB/CHROM_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROM_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROM_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROM_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROM_SGWB "efac_c equad_c ecorr_c chrom_wide gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROM_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROMCIDX_SGWB/CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROMCIDX_SGWB "efac_c equad_c ecorr_c chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROMCIDX_SGWB/RN_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROMCIDX_SGWB "efac_c equad_c ecorr_c red chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROMCIDX_SGWB/RN_DM_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROMCIDX_SGWB/DM_CHROMCIDX_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROMCIDX_SGWB "efac_c equad_c ecorr_c dm chrom_cidx gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_SW_SGWB/SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} SW_SGWB "efac_c equad_c ecorr_c sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_SWDET_SGWB/SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} SWDET_SGWB "efac_c equad_c ecorr_c swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_SW_SGWB/DM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_SW_SGWB "efac_c equad_c ecorr_c dm sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_SW_SGWB/RN_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_SW_SGWB "efac_c equad_c ecorr_c red sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_SW_SGWB/RN_DM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_SW_SGWB "efac_c equad_c ecorr_c red dm sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROM_SW_SGWB/RN_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROM_SW_SGWB "efac_c equad_c ecorr_c red chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROM_SW_SGWB/RN_DM_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROM_SW_SGWB "efac_c equad_c ecorr_c red dm chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROM_SW_SGWB/DM_CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROM_SW_SGWB "efac_c equad_c ecorr_c dm chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROM_SW_SGWB/CHROM_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROM_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROM_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROM_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROM_SW_SGWB "efac_c equad_c ecorr_c chrom_wide sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROM_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROMCIDX_SW_SGWB/CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROMCIDX_SW_SGWB/RN_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c red chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROMCIDX_SW_SGWB/RN_DM_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx sw gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROMCIDX_SW_SGWB/DM_CHROMCIDX_SW_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SW_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SW_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SW_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROMCIDX_SW_SGWB "efac_c equad_c ecorr_c dm chrom_cidx sw gw_const_gamma" 
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SW_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_SWDET_SGWB/DM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_SWDET_SGWB "efac_c equad_c ecorr_c dm swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_SWDET_SGWB/RN_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_SWDET_SGWB "efac_c equad_c ecorr_c red swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_SWDET_SGWB/RN_DM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_SWDET_SGWB "efac_c equad_c ecorr_c red dm swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROM_SWDET_SGWB/RN_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c red chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROM_SWDET_SGWB/RN_DM_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c red dm chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROM_SWDET_SGWB/DM_CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROM_SWDET_SGWB "efac_c equad_c ecorr_c dm chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROM_SWDET_SGWB/CHROM_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROM_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROM_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROM_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROM_SWDET_SGWB "efac_c equad_c ecorr_c chrom_wide swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROM_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_CHROMCIDX_SWDET_SGWB/CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_CHROMCIDX_SWDET_SGWB/RN_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c red chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROMCIDX_SWDET_SGWB/RN_DM_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c red dm chrom_cidx swdet gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_DM_CHROMCIDX_SWDET_SGWB/DM_CHROMCIDX_SWDET_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SWDET_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SWDET_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SWDET_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} DM_CHROMCIDX_SWDET_SGWB "efac_c equad_c ecorr_c dm chrom_cidx swdet gw_const_gamma" 
    echo "rerunning ${psr}_PSRADD10_PBILBY_DM_CHROMCIDX_SWDET_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_psradd_10/${psr}/${psr}_RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB/RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB_final_res.json" ]] && [[ ! "${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB" == $(grep -w -m 1 ^${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list) ]]; then
    sbatch --mem-per-cpu=${req_mem}GB -J ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_PSRADD10_slurm.sh ${psr} RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB "efac_c equad_c ecorr_c red dm chrom_wide sw chrom_annual chrom_gauss_bump gw_const_gamma"
    echo "rerunning ${psr}_PSRADD10_PBILBY_RN_DM_CHROM_SW_CHROMANNUAL_CHROMBUMP_SGWB" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby_fixedPortrait_psradd10_full.list
    #((counter++))
fi