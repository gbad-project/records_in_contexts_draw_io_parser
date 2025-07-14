#!/bin/bash

set -e

# Get the dataset name from the first argument
DATASET_NAME=$1

# Define the root directory
ROOT_DIR="$(pwd)"

# Define the config file path
CONFIG_FILE="$ROOT_DIR/tests/config.json"

# Activate conda environment
# Not using conda anymore
# CONDA_ENV='gbad-next'
# eval "$(conda shell.bash hook)"
# conda activate "$CONDA_ENV" || { echo "Failed to activate conda environment: $CONDA_ENV" >&2; exit 1; }

# Install jq if not present
if ! command -v jq &> /dev/null
then
    echo "jq could not be found, installing..."
    apt-get update && apt-get install -y jq
fi

# Parse the config file to get the file paths for the given dataset
DRAWIO_FILE=$(jq -r ".${DATASET_NAME}.drawio_file" "$CONFIG_FILE")
CSV_FILE=$(jq -r ".${DATASET_NAME}.csv_file" "$CONFIG_FILE")
RML_FILE_DIR=$(dirname "$DRAWIO_FILE")
RML_FILE_NAME=$(basename "$DRAWIO_FILE" .drawio)
RML_FILE="$RML_FILE_DIR/${RML_FILE_NAME}.rml"
LOG_DIR="$ROOT_DIR/tests/logs"
mkdir -p "$LOG_DIR"

# Run the drawio parser
echo "Running draw.io parser for $DATASET_NAME..."
echo "Command: cat \"$DRAWIO_FILE\" | python draw_io_parser.py --config \"$CONFIG_FILE\" --dataset \"$DATASET_NAME\" > \"${DRAWIO_FILE%.drawio}.ttl\" 2>&1"
cat "$DRAWIO_FILE" | python draw_io_parser.py --config "$CONFIG_FILE" --dataset "$DATASET_NAME" > "${DRAWIO_FILE%.drawio}.ttl" 2>&1
echo "draw.io parser finished."

# Run the schema mapper
echo "Running schema mapper for $DATASET_NAME..."
python map_schema.py --config "$CONFIG_FILE" --dataset "$DATASET_NAME"
echo "Schema mapper finished."

# Run the RML mapper
echo "Running RML mapper for $DATASET_NAME..."
python map_rml.py --config "$CONFIG_FILE" --dataset "$DATASET_NAME" "$RML_FILE"
echo "RML mapper finished."

echo "Generic test script for $DATASET_NAME finished successfully."
