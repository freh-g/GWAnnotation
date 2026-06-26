#!/usr/bin/env bash
set -euo pipefail

# Start the timer
START_TIME=$SECONDS
LOG_FILE="pipeline_runtime.log"

echo "Starting analysis pipeline"

echo "STEP 1 / 6  |  Downloading the studies"
# python 1_download_studies.py
echo "STEP 1 completed"

echo "STEP 2 / 6  |  parsing the studies"
# python 2_parse_studies.py
echo "STEP 2 completed"

echo "STEP 3 / 6  |  Lifting and preparing vcfs"
# python 3_lift_and_make_vcfs.py
echo "STEP 3 completed"

echo "STEP 4 / 6  |  Annotating the vcfs"
# python 4_annotate_parallel.py
echo "STEP 4 completed"

echo "STEP 5 / 6  |  Preprocessing vcfs"
# python 5_preprocessing.py
echo "STEP 5 completed"

echo "STEP 6 / 6  |  Running models"
srun run_xgb.sh
echo "STEP 6 completed"

echo "Pipeline Completed successfully"

# Calculate elapsed time
ELAPSED_SECONDS=$(( SECONDS - START_TIME ))
MINUTES=$(( ELAPSED_SECONDS / 60 ))
SECS=$(( ELAPSED_SECONDS % 60 ))

# Format the log entry
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
LOG_ENTRY="[$TIMESTAMP] Pipeline finished successfully. Total runtime: ${MINUTES}m ${SECS}s."

# Print to console AND append to log file
echo "$LOG_ENTRY" | tee -a "$LOG_FILE"
