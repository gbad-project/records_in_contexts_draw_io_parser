#!/bin/bash

ROOT_DIR="$(pwd)"

CONDA_ENV='gbad-next'

LOG_FILE_PARSE_AUTH="$ROOT_DIR/tests/logs/test_parse_auth.log"

COMMAND_PARSE_AUTH="${ROOT_DIR}/parse-auth.sh"

LOG_FILE_MAP_SCHEMA_AUTH="$ROOT_DIR/tests/logs/test_map_schema_auth_authority_tailshuf_100.log"

COMMAND_MAP_SCHEMA_AUTH="python $ROOT_DIR/map_schema.py auth tests/test_authority_tailshuf_100.csv"

export LOG_FILE_MAP_RML_AUTH="$ROOT_DIR/tests/logs/test_map_rml_auth_authority_tailshuf_100.log"

COMMAND_MAP_RML_AUTH="python $ROOT_DIR/map_rml.py auth tests/test_authority_tailshuf_100/"

# Exit on error
set -e

# Check if CONDA_ENV is set
if [ -z "$CONDA_ENV" ]; then
    echo "Error: CONDA_ENV environment variable is not set" >&2
    exit 1
fi

# Function to deactivate conda environment
cleanup() {
    echo "Deactivating conda environment..."
    conda deactivate
}

# Set trap to ensure cleanup runs in all cases
trap cleanup EXIT

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda environment: $CONDA_ENV" >&2; exit 1; }

# Parse and convert drawio
echo "Running parse command and logging output..."
script -c "$COMMAND_PARSE_AUTH" "$LOG_FILE_PARSE_AUTH"

# Execute the script command
echo "Running command and logging output..."
script -c "$COMMAND_MAP_SCHEMA_AUTH" "$LOG_FILE_MAP_SCHEMA_AUTH"

# Execute the script command
echo "\nRunning map rml auth command and logging output..."
script -c "$COMMAND_MAP_RML_AUTH" "$LOG_FILE_MAP_RML_AUTH"

# Note: cleanup will be called automatically thanks to the trap
echo "Done!"
