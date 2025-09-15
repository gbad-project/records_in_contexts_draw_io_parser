#!/bin/bash

# Identify relative paths within repo
MACOS_DIR="$(dirname "$0")"
RMLMAPPER_GUI_DIR="$(dirname "$MACOS_DIR")"
ROOT_DIR="$(dirname "$RMLMAPPER_GUI_DIR")"

# Load global vars from .env if it exists
ENV_FILE="$RMLMAPPER_GUI_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Register default CONDA_ENV if not already set
if [ -z "$CONDA_ENV" ]; then
    CONDA_ENV="gbad-next"
    echo "CONDA_ENV=\"$CONDA_ENV\"" >> "$ENV_FILE"
fi

eval "$(conda shell.bash hook)" >/dev/null 2>&1
conda activate "$CONDA_ENV" >/dev/null 2>&1

# Prompt for JAR file and save to .env
if [ ! -f "$RMLMAPPER" ]; then
    echo "👉 Please drag your JAR file into this window and press Enter:"
    read JAR_INPUT
    JAR_INPUT=$(echo "$JAR_INPUT" | xargs)   # trim spaces
    RMLMAPPER="$JAR_INPUT"
    echo "RMLMAPPER=\"$RMLMAPPER\"" >> "$ENV_FILE"
fi

# Prompt to enter BASE_URI and save to .env
if [ -z "$BASE_URI" ]; then
    echo "🌐 BASE_URI is not set in environment."
    read -p "Please enter a BASE_URI: " BASE_URI
    echo "BASE_URI=\"$BASE_URI\"" >> "$ENV_FILE"
fi

echo "👉 Please drag your RML file into this window and press Enter:"
read INPUT

# Trim spaces around the input (important on macOS)
INPUT=$(echo "$INPUT" | xargs)

OUTPUT="${INPUT%.*}.ttl"

java -jar "$RMLMAPPER" \
    -s turtle \
    -m "$INPUT" \
    -o "$OUTPUT" \
    -b "$BASE_URI"

echo
echo "✅ Done! Output written to: $OUTPUT"
read -p "Press Enter to close..."
