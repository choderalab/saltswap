#!/bin/bash

nkernal=(126 512 1024 2048 4096)
nverlet=(1 2 4 8 16 32 64)

for k in ${nkernal[*]}
do
    for v in ${nverlet[*]}
    do
        if [ -a "nk${k}_nV${v}/run1.txt" ]
        then
            echo 'Simulate completed'
        else
            #mkdir "nk${k}_nV${v}"
            cd "nk${k}_nV${v}"
            sed "s/REPLACE/--npert $k --nprop $v/" ../submit_dummy > submit
            qsub submit -N "nk${k}nV${v}"
	    cd ../
        fi
    done
done
