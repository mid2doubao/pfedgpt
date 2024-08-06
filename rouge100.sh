#!/bin/bash

base_model="chavinlo/alpaca-native"
lora_config_path="./lora100-shepherd-7b-task/8/"
load_8bit="--load_8bit"

# Loop through the client numbers from 0 to 19
for client_number in {0..19}
do
  # Construct paths dynamically based on the client number
  lora_weights_path="./lora100-shepherd-7b-task/8/19/local_output_${client_number}/pytorch_model.bin"
  result_save_path="./private100_task/client${client_number}_1.json"
  test_dataset_path="./lora100-shepherd-7b-task/8/test_dataset_client_${client_number}.json"
  
  # Execute the Python script with the constructed parameters
  python eva1.py $load_8bit \
      --base_model "$base_model" \
      --lora_weights_path "$lora_weights_path" \
      --lora_config_path "$lora_config_path" \
      --result_save_path "$result_save_path" \
      --test_dataset_path "$test_dataset_path"
  
  # Optional: Output to indicate which client has been processed
  echo "Processed client ${client_number}"
done