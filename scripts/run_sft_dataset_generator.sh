#!/bin/bash

# Get timestamp for unique log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory if it doesn't exist
mkdir -p artifacts/log

# Set log files
LOG_FILE="artifacts/log/sft_dataset_generator_${TIMESTAMP}.log"

# Run the script with nohup
nohup python -u exp/generate_sft_data.py > "${LOG_FILE}" 2>&1 &

echo "Logs will be written to: ${LOG_FILE}"
echo "Process started with PID $!"