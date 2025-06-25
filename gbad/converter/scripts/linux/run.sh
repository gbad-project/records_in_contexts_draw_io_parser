#!/bin/bash

ROOT_DIR="$(pwd)" && TMP_DIR="$ROOT_DIR/tmp"

CONDA_ENV='gbad-next'

CONVERT_DESCRIPTION_SCRIPT="$ROOT_DIR/gbad/converter/scripts/linux/convert_description.sh"

source "$CONVERT_DESCRIPTION_SCRIPT"

CONVERT_AUTHORITY_SCRIPT="$ROOT_DIR/gbad/converter/scripts/linux/convert_authority.sh"

source "$CONVERT_AUTHORITY_SCRIPT"

LOG_FILE_MERGE="$ROOT_DIR/gbad/converter/logs/run.log"

MERGED_GRAPH="$ROOT_DIR/gbad/data/store.nq"

COMMAND_MERGE="python $ROOT_DIR/merge_graphs.py $TMP_DIR $MERGED_GRAPH"

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



# Merge Description and Authority graphs
echo "Running merge command and logging output..."
mkdir -p "$(dirname "$LOG_FILE_MERGE")" && mkdir -p "$TMP_DIR" && mkdir -p "$(dirname "$MERGED_GRAPH")"
map_rml_auth_log_file="$ROOT_DIR/$(grep "Successfully saved postprocessed graph" $LOG_FILE_MAP_RML_AUTH | sed -E "s/.*: '(.*)'/\1/" | tr -d '\r')"
cp "$map_rml_auth_log_file" "$TMP_DIR/authority.nt"
map_rml_add_log_file="$ROOT_DIR/$(grep "Successfully saved postprocessed graph" $LOG_FILE_MAP_RML_ADD | sed -E "s/.*: '(.*)'/\1/" | tr -d '\r')"
cp "$map_rml_add_log_file" "$TMP_DIR/description.nt"
script -c "$COMMAND_MERGE" "$LOG_FILE_MERGE"

# Note: cleanup will be called automatically thanks to the trap
echo "Done!"
