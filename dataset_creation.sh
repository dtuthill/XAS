#!/bin/bash

#source /reg/g/psdm/etc/psconda.sh #this sets environment for psana1, LCLS 1, python 2
#source /cds/sw/ds/ana/conda2/inst/etc/profile.d/conda.sh
source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
base_path=/cds/home/d/dtuthill/LY72_Spooktroscopy/xas # changeme - where is the preproc.py file
script=$base_path/valence_sum.py # changeme
#log=$base_path/logs/run$1.log # 
log=/cds/home/d/dtuthill/LY72_Spooktroscopy/xas/logs/valence_sum.log # 

echo $log
echo $script

if [ -z "$1" ]
then
    n_nodes=1
else
    n_nodes=$1
fi

if [ -z "$2" ]
then
    tasks_per_node=12 #12 nodes on psanaq
else
    tasks_per_node=$2
fi

sbatch -p psanaq -N $n_nodes --ntasks-per-node $tasks_per_node --output $log --wrap="mpirun python $script"