#!/bin/bash
# PBS -N offshore_particles_sigma0
# PBS -P e14
# PBS -q normal
# PBS -l walltime=4:00:00
# PBS -l mem=150GB
# PBS -l software=netcdf
# PBS -l ncpus=8
# PBS -l jobfs=10GB
# PBS -l storage=gdata/e14+gdata/v45+gdata/hh5+gdata/ik11+scratch/e14+scratch/v45
# PBS -j oe
# PBS -l wd

# NOTE: This script gets submitted by running ./submit_multi_job_offshore_sigma0.sh

ncpu=8

## I/O filenames
# this reads the name of the current run directory to use for output etc:
#run_name=${PWD##*/}
script_dir=/home/561/hd4873/parcel_runs/AntConn/scripts/analysis/offshore/
output_dir=/g/data/e14/hd4873/runs/parcels/output/AntConn/data/offshore/
work_dir=/scratch/v45/hd4873/AntConn/$num1/
mkdir $work_dir
#work_dir=$PBS_JOBFS
python_filename=$(ls offshore_analysis_sigma0.py)

# load conda
module use /g/data/hh5/public/modules
module load conda/analysis3

# this is needed or it saves temp output to your home dir:
cd $work_dir
cp $script_dir/$python_filename .

# run parcels:
mpirun -np $ncpu python $python_filename $num1 &>> $script_dir/logs/offshore_particles_file_$num1-sigma0_278-30_1x0-5deg.out

# remove output from working dir
# mkdir $output_dir
# rm *_analysis.py

exit
