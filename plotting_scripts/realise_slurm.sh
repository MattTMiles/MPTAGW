#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH -o %x.out
#SBATCH --mem-per-cpu=1gb



source ~/.bashrc
#export OMP_NUM_THREADS=1
ml conda


conda activate mpippcgw
cd /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1909_realise/


touch "noise_realiser_${1}"

echo "noise_realiser_${1}"


python /home/mmiles/soft/GW/plotting_scripts/bootstrap_noise_realise.py -pulsar J1909-3744 -directory /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1909_realise/ -noise dm sw red -idx ${1}


cd /fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1909_realise/

rm -f "noise_realiser_${1}"

echo done
