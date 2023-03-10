#!/bin/bash
# PBS -N ZonalConn_shelfwest
# PBS -P e14
# PBS -q normalbw
# PBS -l walltime=4:00:00
# PBS -l mem=240GB
# PBS -l software=netcdf
# PBS -l ncpus=1
# PBS -l jobfs=50GB
# PBS -l storage=gdata/e14+gdata/v45+gdata/hh5+gdata/ik11+scratch/e14
# PBS -j oe
# PBS -l wd

## Note, run this with:
## qsub submit_python_script.sh
ncpu=1

## I/O filenames
# this reads the name of the current run directory to use for output etc:
#run_name=${PWD##*/}
script_dir=~/parcel_runs/AntConn/scripts/analysis/zonal_conn/
#output_dir=/g/data/e14/hd4873/runs/parcels/output/AntConn/data/xarray_files/time_chunked/
work_dir=/scratch/e14/hd4873/parcels/AntConn/$num1
mkdir $work_dir
#work_dir=$PBS_JOBFS
python_filename=$(ls zonal_connectivity_shelf_west.py)

# load conda
module use /g/data/hh5/public/modules/
module load conda/analysis3

# this is needed or it saves temp output to your home dir:
cd $work_dir
cp $script_dir/$python_filename .

# run analysis:
mpirun -np $ncpu python $python_filename $num1 $var &>> $script_dir/zonalconnectivity_${var}_$num1.out

# copy output back to /g/data/
# mkdir $output_dir
# mv * $output_dir

exit
