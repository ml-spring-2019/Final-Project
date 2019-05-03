#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=ECPE251_Jeremy_Project03 ### Job Name
#SBATCH --time=08:00:00     ### WallTime
#SBATCH --nodes=1           ### Number of Nodes
#SBATCH --ntasks-per-node=1 ### Number of tasks (MPI processes)
##SBATCH --cpus-per-task=2  ### Number of threads per task (OMP threads). Get num threads using program arguments

echo "epoch_size, scale_width, scale_height, train_rate, loss, accuracy, io_time, arch_time, fit_time, eval_time" > results.csv

declare -a scale=("25" "50" "75" "100" "125" "150" "175" "200")
declare -a epoch=("30")
declare -a train_rate=("0.8")

for i in "${train_rate[@]}"
do
    for j in "${scale[@]}"
    do
        for k in "${epoch[@]}"
        do
            python3 main.py $i $j $k
        done
    done
done
