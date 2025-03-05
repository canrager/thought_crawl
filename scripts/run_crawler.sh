#!/bin/bash

# Script to run the thought crawler in debug mode
# Created for running exp/run_crawler.py with debug configuration

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
# Change to the project root directory
cd $PROJECT_ROOT

# Create artifacts/log directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/artifacts/log"
# Generate timestamp for log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/artifacts/log/crawler_debug_${TIMESTAMP}.log"

echo "Starting thought crawler in debug mode..."
echo "Logs will be written to: $LOG_FILE"

# Prerequisite: install spacy model
python -m spacy download en_core_web_sm
# Run the crawler script with nohup and write to the log file
nohup python exp/run_crawler.py \
    --device "cuda:0" \
    --cache_dir "/share/u/models/" \
    --debug "$@" \
    > "$LOG_FILE" 2>&1 &

# Store the process ID
PID=$!
echo "Process started with PID: $PID"
echo "To check status: tail -f $LOG_FILE"
echo "To terminate: kill $PID"

# Add any additional arguments as needed
# Example: --load_fname "path/to/saved/state" if you want to resume from a saved state

echo "Thought crawler debug run started in background" 