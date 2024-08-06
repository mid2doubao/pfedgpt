#!/bin/bash

base_model="chavinlo/alpaca-native"
lora_config_path="./lora50-shepherd-7b-dpa2/50/"
load_8bit="--load_8bit"

# Loop through the client numbers from 0 to 19
for client_number in {20..49}
do
  # Construct paths dynamically based on the client number
  lora_weights_path="./lora50-shepherd-7b-dpa2/50/19/local_output_${client_number}/pytorch_model.bin"
  result_save_path="./private50_dpa2/client${client_number}_1.json"
  test_dataset_path="./lora50-shepherd-7b-dpa2/50/test_dataset_client_${client_number}.json"
  
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
