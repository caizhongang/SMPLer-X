#!/bin/bash
dataset=$1
USER=yinwanqi

# CSV file path
csv_file="single_data_exp_list_v2.csv"
# csv_file="small_test.csv"

# Read the CSV file and extract the first 3 columns
while IFS=',' read -r job checkpoint_path checkpoint_id _; do
    echo "$checkpoint_path,$checkpoint_id"
    ./slurm_eval_on_train_${dataset}.sh $job 1 $checkpoint_path $checkpoint_id &
    
    # while [[ $(squeue -u ${USER} --format=%j | wc -l) -gt 32 ]]; do 
    #     sleep 10
    # done 
done < "$csv_file"