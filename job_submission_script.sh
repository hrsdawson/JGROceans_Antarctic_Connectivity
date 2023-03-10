#!/bin/bash
#PBS -N proj01_parcels
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l mem=220GB
#PBS -l software=netcdf
#PBS -l ncpus=12
#PBS -l jobfs=20GB
#PBS -l storage=gdata/e14+gdata/v45+gdata/hh5+gdata/ik11+scratch/e14
#PBS -j oe
#PBS -l wd
#PBS -v count

## Note, run this with:
## qsub -v count=0 job_submission_script.sh

# set max number of time loops to run:
n_loops=170

# set number of cpus
ncpu=12

## I/O filenames
# this reads the name of the current run directory to use for output etc:
run_name=${PWD##*/}
script_dir=/home/561/hd4873/parcel_runs/$run_name/
output_dir=/g/data/e14/hd4873/runs/parcels/output/$run_name/
work_dir=/scratch/e14/hd4873/parcels/$run_name/
mkdir $work_dir

# find the name of the parcels python file:
# note that this assumes you only have one python file in this directory.
python_filename=$(ls *.py)

# load conda
module use /g/data/hh5/public/modules
module load conda/analysis3

# Move script to the work directory
# this is needed or it saves temp output to your home dir:
cd $work_dir
cp $script_dir/$python_filename .

# run parcels:
export OMP_NUM_THREADS=$ncpu
python $python_filename $count $run_name $n_loops &>> $script_dir/${run_name}_count$count.out
#mpirun -np $ncpu python $python_filename $count $run_name $n_loops &>> $script_dir/${run_name}_count$count.out

# copy output back to /g/data/
mkdir $output_dir
mv * $output_dir

# increment count and resubmit:
count=$((count+1))
if [ $count -lt $n_loops ]; then 
cd $script_dir
qsub -v count=${count} job_submission_script.sh
fi

exit
