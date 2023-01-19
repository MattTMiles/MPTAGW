#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --job-name=%x.out
#SBATCH --mem=2gb
#SBATCH --tmp=2gb


source ~/.bashrc
export OMP_NUM_THREADS=1

conda activate gw

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

touch "${1}_timing_model_sampling"

echo "${1}_timing_model_sampling"

mpirun python /home/mmiles/soft/GW/enterprise_run_timing_pars.py -pulsar $1 -results timing_sample_mpi4 -noise_search efac_c ecorr_c equad_c dm red -sampler ppc -partim /fred/oz002/users/mmiles/MPTA_GW/partim -noisefile /fred/oz002/users/mmiles/MPTA_GW/enterprise/MPTA_active_noise_models/MPTA_WN_models.json -nlive 200 -alt_dir out_ppc_timing_par_sampling

cd /fred/oz002/users/mmiles/MPTA_GW/enterprise

rm -f "${1}_timing_model_sampling"


echo done
