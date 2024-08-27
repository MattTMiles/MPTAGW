#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=01:55:00
#SBATCH -o /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/job_outfiles/%x.out
#SBATCH --mem-per-cpu=10000MB

ml gcc/12.2.0 openmpi/4.1.4

ml apptainer

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata
touch "job_trackers/pbilby_${1}_${2}"
echo "job_trackers/pbilby_${1}_${2}"



i=0
while [ $i -lt 1 ]; do
    mpirun -np $SLURM_NTASKS apptainer run -B /fred,$HOME /fred/oz002/users/mmiles/apptainer_newent.sif python /home/mmiles/soft/GW/ozstar2/enterprise_run_pbilby.py -pulsar $1 -results $2 -noise_search $3 -sampler pbilby -partim /fred/oz002/dreardon/mk_gw/pipeline/djr_final_final_final_august/ -noisefile /fred/oz002/users/mmiles/MPTA_GW/SMBHB/new_models/august23data/MPTA_noise_values_SGWBWN_checked_kernelECORR.json -nlive 400 -alt_dir DJR_noise/${1}

    if [[ "$?" -eq 139 ]]; then
        echo "segfault !";
        rm /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/core*
        rm /fred/oz002/dreardon/mk_gw/pipeline/djr_final_final_final_august/core*
    else echo "no segfault !";
        ((i++));
    fi;
done


cd /fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata
rm -f "job_trackers/pbilby_${1}_${2}"

echo done