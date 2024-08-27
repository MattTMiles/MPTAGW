#!/bin/bash

# rerun_noise_megaslurm.sh psr
# runs all unfinished noise models

psr=$1
cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata

counter=$(cat /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list | wc -l)

# inc. runs with a fixed gamma and amp SGWB

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_FIXEDGW/DM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_FIXEDGW "wn dm gw_c"
    echo "rerunning ${psr}_PBILBY_DM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_FIXEDGW/RN_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_FIXEDGW "wn red gw_c"
    echo "rerunning ${psr}_PBILBY_RN_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_FIXEDGW/RN_DM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_FIXEDGW "wn red dm gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROM_FIXEDGW/RN_CHROM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROM_FIXEDGW "wn red chrom_wide gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROM_FIXEDGW/RN_DM_CHROM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROM_FIXEDGW "wn red dm chrom_wide gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROM_FIXEDGW/DM_CHROM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROM_FIXEDGW "wn dm chrom_wide gw_c"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROM_FIXEDGW/CHROM_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROM_FIXEDGW "wn chrom_wide gw_c"
    echo "rerunning ${psr}_PBILBY_CHROM_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROMCIDX_FIXEDGW/CHROMCIDX_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROMCIDX_FIXEDGW "wn chrom_cidx gw_c"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROMCIDX_FIXEDGW/RN_CHROMCIDX_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROMCIDX_FIXEDGW "wn red chrom_cidx gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROMCIDX_FIXEDGW/RN_DM_CHROMCIDX_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROMCIDX_FIXEDGW "wn red dm chrom_cidx gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROMCIDX_FIXEDGW/DM_CHROMCIDX_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROMCIDX_FIXEDGW "wn dm chrom_cidx gw_c"
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SW_FIXEDGW/SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SW_FIXEDGW "wn sw gw_c"
    echo "rerunning ${psr}_PBILBY_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SWDET_FIXEDGW/SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SWDET_FIXEDGW "wn swdet gw_c"
    echo "rerunning ${psr}_PBILBY_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_SW_FIXEDGW/DM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_SW_FIXEDGW "wn dm sw gw_c"
    echo "rerunning ${psr}_PBILBY_DM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_SW_FIXEDGW/RN_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_SW_FIXEDGW "wn red sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_SW_FIXEDGW/RN_DM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_SW_FIXEDGW "wn red dm sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROM_SW_FIXEDGW/RN_CHROM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROM_SW_FIXEDGW "wn red chrom_wide sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROM_SW_FIXEDGW/RN_DM_CHROM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROM_SW_FIXEDGW "wn red dm chrom_wide sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROM_SW_FIXEDGW/DM_CHROM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROM_SW_FIXEDGW "wn dm chrom_wide sw gw_c"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROM_SW_FIXEDGW/CHROM_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROM_SW_FIXEDGW "wn chrom_wide sw gw_c"
    echo "rerunning ${psr}_PBILBY_CHROM_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROMCIDX_SW_FIXEDGW/CHROMCIDX_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROMCIDX_SW_FIXEDGW "wn chrom_cidx sw gw_c"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROMCIDX_SW_FIXEDGW/RN_CHROMCIDX_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROMCIDX_SW_FIXEDGW "wn red chrom_cidx sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROMCIDX_SW_FIXEDGW/RN_DM_CHROMCIDX_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROMCIDX_SW_FIXEDGW "wn red dm chrom_cidx sw gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROMCIDX_SW_FIXEDGW/DM_CHROMCIDX_SW_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SW_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SW_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SW_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROMCIDX_SW_FIXEDGW "wn dm chrom_cidx sw gw_c" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SW_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_SWDET_FIXEDGW/DM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_SWDET_FIXEDGW "wn dm swdet gw_c"
    echo "rerunning ${psr}_PBILBY_DM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_SWDET_FIXEDGW/RN_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_SWDET_FIXEDGW "wn red swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_SWDET_FIXEDGW/RN_DM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_SWDET_FIXEDGW "wn red dm swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROM_SWDET_FIXEDGW/RN_CHROM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROM_SWDET_FIXEDGW "wn red chrom_wide swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROM_SWDET_FIXEDGW/RN_DM_CHROM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROM_SWDET_FIXEDGW "wn red dm chrom_wide swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROM_SWDET_FIXEDGW/DM_CHROM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROM_SWDET_FIXEDGW "wn dm chrom_wide swdet gw_c"
    echo "rerunning ${psr}_PBILBY_DM_CHROM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROM_SWDET_FIXEDGW/CHROM_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROM_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROM_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROM_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROM_SWDET_FIXEDGW "wn chrom_wide swdet gw_c"
    echo "rerunning ${psr}_PBILBY_CHROM_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_CHROMCIDX_SWDET_FIXEDGW/CHROMCIDX_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_CHROMCIDX_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_CHROMCIDX_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CHROMCIDX_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} CHROMCIDX_SWDET_FIXEDGW "wn chrom_cidx swdet gw_c"
    echo "rerunning ${psr}_PBILBY_CHROMCIDX_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_CHROMCIDX_SWDET_FIXEDGW/RN_CHROMCIDX_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_CHROMCIDX_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_CHROMCIDX_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_CHROMCIDX_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_CHROMCIDX_SWDET_FIXEDGW "wn red chrom_cidx swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_CHROMCIDX_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_RN_DM_CHROMCIDX_SWDET_FIXEDGW/RN_DM_CHROMCIDX_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} RN_DM_CHROMCIDX_SWDET_FIXEDGW "wn red dm chrom_cidx swdet gw_c"
    echo "rerunning ${psr}_PBILBY_RN_DM_CHROMCIDX_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_DM_CHROMCIDX_SWDET_FIXEDGW/DM_CHROMCIDX_SWDET_FIXEDGW_final_res.json" ]] && [[ ! "${psr}_PBILBY_DM_CHROMCIDX_SWDET_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_DM_CHROMCIDX_SWDET_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_DM_CHROMCIDX_SWDET_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} DM_CHROMCIDX_SWDET_FIXEDGW "wn dm chrom_cidx swdet gw_c" 
    echo "rerunning ${psr}_PBILBY_DM_CHROMCIDX_SWDET_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_SMBHB_CONST_ER/SMBHB_CONST_ER_final_res.json" ]] && [[ ! "${psr}_PBILBY_CONST_ER" == $(grep -w -m 1 ^${psr}_PBILBY_CONST_ER /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_CONST_ER /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} SMBHB_CONST_ER "smbhb_const_wn extra_red" 
    echo "rerunning ${psr}_PBILBY_CONST_ER" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi

if [[ ! -f "/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby/${psr}_FIXEDGW/${psr}_WN/WN_final_res.json" ]] && [[ ! "${psr}_PBILBY_WN_FIXEDGW" == $(grep -w -m 1 ^${psr}_PBILBY_WN_FIXEDGW /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list) ]]; then
    sbatch -J ${psr}_PBILBY_WN_FIXEDGW /home/mmiles/soft/GW/ozstar2/pbilby_apptainer_FIXEDGW_slurm.sh ${psr} WN "wn gw_c" 
    echo "rerunning ${psr}_PBILBY_WN_FIXEDGW" >> /fred/oz002/users/mmiles/MPTA_GW/all_noise_slurm_pbilby.list
    #((counter++))
fi