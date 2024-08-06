#!/bin/bash

# Define the base path for your files
base_path="./private72.100/"
result_base_path="./private72.100/"
file_prefix="client"
result_prefix="rou"

# Loop through client numbers from 0 to 7
for client_number in {0..7}
do
  # Loop through the runs from 6 to 10
  for run_number in {10..10}
  do
    # Construct file paths
    #file_path="${base_path}${file_prefix}${client_number}_${run_number}.json"
    file_path="${base_path}${file_prefix}${client_number}_1.json"
    results_file_path="${result_base_path}${result_prefix}${client_number}_${run_number}.json"
    
    # Execute the Python script with the constructed parameters
    python ai-evaluation.py \
        --file_path "$file_path" \
        --results_file_path "$results_file_path"
    
    # Optional: Output to indicate which client and run has been processed
    echo "Processed client ${client_number}, run ${run_number}"
  done
done
