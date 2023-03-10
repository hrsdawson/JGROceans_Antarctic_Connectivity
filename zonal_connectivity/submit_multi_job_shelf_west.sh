#!/bin/bash

# # Note, run this with:
# # ./submit_multi_job_shelf_west.sh 

variables=(shelf_west)
## (asc_east asc_north asc_west shelf_west shelf_asc)
for i in ${variables[@]}; do
	start_num=0
	while [ $start_num -lt 21 ]; do
		echo $start_num $i
        	qsub -v num1=$start_num,var=$i submit_python_script_shelf_west.sh
		let start_num+=1
	done
done
