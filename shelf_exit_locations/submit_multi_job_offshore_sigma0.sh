#!/bin/csh -f

set start_num = 0
#set start_z = 0
#set end_z = 200
@ i = $start_num
#@ j = $start_z
#@ k = $end_z
while ($i <= 72)
	qsub -v num1=$i submit_python_script_offshore_sigma0.sh
	@ i = $i + 1
end
