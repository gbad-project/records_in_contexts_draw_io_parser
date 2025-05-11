#!/bin/sh

ROOT_DIR="$(pwd)"

CONDA_ENV='govon'

LOG_FILE="$ROOT_DIR/tests/logs/test_map_schema_add_description_rg_1-429.log"

COMMAND="python $ROOT_DIR/map_schema.py add tests/test_description_rg_1-429.csv"

LOG_FILE_2="$ROOT_DIR/tests/logs/test_map_rml_add_description_rg_1-429.log"

COMMAND_2="python $ROOT_DIR/map_rml.py add tests/test_description_rg_1-429/general_add_descriptions_and_listings_to_ric-o_model_2025-04-15_pz.rml"

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

# Execute the script command
echo "Running command and logging output..."
script "$LOG_FILE" sh -c "$COMMAND"

# Execute the script command
echo "\nRunning command 2 and logging output..."
script "$LOG_FILE_2" sh -c "$COMMAND_2"

# Note: cleanup will be called automatically thanks to the trap
echo "Done!"
