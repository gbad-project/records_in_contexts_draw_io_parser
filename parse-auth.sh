#!/bin/bash

# originally generated with Gemini Free Version on 2024-06-05,
# revised on 2024-07-10, with modifications

# Path to your original script (replace with the actual path)
ORIGINAL_SCRIPT_PATH="./parse.sh"

# Set the directory containing the graph
# script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
graph_dir="gbad/schema/authority"

# Find the first *.drawio file in the graph directory
drawio_file=$(find "$graph_dir" -type f -name "*.drawio" | head -n 1)

# Set desired args
args="-m url \
      -o http://gbad.archives.gov.on.ca/schema/authority \
      -p http://gbad.archives.gov.on.ca/schema/authority#"

# Construct the python command
python_command="\"$ORIGINAL_SCRIPT_PATH\" \"$drawio_file\" $args"

# Run the parser script
eval $python_command
